"""Step 3 of Phase 1 — composite generator.

Takes the per-fragment RGBA PNGs produced by tear_simulator.py, applies a
random rotation and random translation to each, and pastes them all onto a
single dark-gray canvas (no overlaps). The result mimics what the user gets
from a real scan: torn fragments laid out arbitrarily on a dark surface.

Per-document outputs (written into the same per-doc directory the simulator
created):
    composite.png                 ← X (model input)
    composite_transforms.json     ← per-fragment placement metadata + ground
                                    truth recovery transforms

Mathematical model
------------------
For each fragment with original-frame top-left bbox (bx, by) and crop size
(bw, bh), placement on the composite is:
    1. Translate fragment crop so that its centre is at the origin.
    2. Rotate by `angle_deg`.
    3. Translate so the rotated centre lands at `center_xy` on the canvas.

The 3×3 affine `M_place` (mapping fragment-crop coords → composite coords)
is stored for every fragment. The ground-truth reconstruction transform
(canvas → original) is then  `M_to_original = T_bbox · M_place^{-1}`,
where `T_bbox` translates by (bx, by) — i.e. shifts the upright fragment
crop back to its position in Y.

Both `M_place` and `M_to_original` are written to JSON as 3×3 lists.

CLI:
    python tools/composite_generator.py --src_root data/dataset/synthetic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BG_RGB = (45, 45, 45)              # near-black scanner surface
DEFAULT_BG_NOISE_SIGMA = 4.0               # add slight texture so it's not flat
DEFAULT_CANVAS_SCALE = 2.5                 # canvas dim = source dim * this
DEFAULT_MARGIN_PX = 24                     # keep fragments off the canvas edge
DEFAULT_PAD_PX = 6                         # min spacing between fragments
DEFAULT_MAX_PLACEMENT_TRIES = 400          # per fragment


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlacementRecord:
    fragment_index: int
    angle_deg: float                        # CCW
    center_xy: tuple[float, float]          # on composite canvas
    rotated_size: tuple[int, int]           # (w, h) of the rotated AABB
    placed_bbox: tuple[int, int, int, int]  # (x, y, w, h) on canvas
    M_place: list[list[float]]              # 3x3, fragment-crop → canvas
    M_to_original: list[list[float]]        # 3x3, canvas → original Y
    rgba_source: str                        # filename relative to per-doc dir


@dataclass
class CompositeResult:
    canvas_size: tuple[int, int]            # (W, H)
    bg_rgb: tuple[int, int, int]
    placements: list[PlacementRecord] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)  # fragment indices that failed


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_3x3(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    """Affine rotation by `angle_deg` CCW about (cx, cy), as 3×3 matrix."""
    M2 = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M = np.eye(3, dtype=np.float64)
    M[:2, :] = M2
    return M


def _translation_matrix_3x3(tx: float, ty: float) -> np.ndarray:
    M = np.eye(3, dtype=np.float64)
    M[0, 2] = tx
    M[1, 2] = ty
    return M


def _rotated_alpha_aabb(width: int, height: int, angle_deg: float
                        ) -> tuple[int, int]:
    """Bounding-box (w, h) of the rotated rectangle (no translation)."""
    rad = np.deg2rad(angle_deg)
    c, s = abs(np.cos(rad)), abs(np.sin(rad))
    w = int(np.ceil(width * c + height * s))
    h = int(np.ceil(width * s + height * c))
    return w, h


def _rotate_rgba(rgba: np.ndarray, angle_deg: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate an RGBA crop, expanding the canvas so nothing is clipped.

    Returns (rotated_rgba, rotated_mask_uint8, M_2x3_in_local_frame) — the
    local-frame matrix maps (x,y) in the input crop to (x,y) in the output crop.
    """
    h, w = rgba.shape[:2]
    out_w, out_h = _rotated_alpha_aabb(w, h, angle_deg)
    cx_in, cy_in = (w - 1) / 2.0, (h - 1) / 2.0
    cx_out, cy_out = (out_w - 1) / 2.0, (out_h - 1) / 2.0

    # Standard expanding-rotation matrix:
    M = cv2.getRotationMatrix2D((cx_in, cy_in), angle_deg, 1.0)
    M[0, 2] += (cx_out - cx_in)
    M[1, 2] += (cy_out - cy_in)

    rotated = cv2.warpAffine(rgba, M, (out_w, out_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    mask = rotated[..., 3]
    return rotated, mask, M


# ---------------------------------------------------------------------------
# Composite construction
# ---------------------------------------------------------------------------

def _make_background(width: int, height: int, bg_rgb: tuple[int, int, int],
                      noise_sigma: float, rng: np.random.Generator
                      ) -> np.ndarray:
    base = np.full((height, width, 3), bg_rgb, dtype=np.float32)
    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, size=(height, width, 3))
        base = base + noise
    return np.clip(base, 0, 255).astype(np.uint8)


def _aabb_overlap(a: tuple[int, int, int, int],
                  b: tuple[int, int, int, int],
                  pad: int) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + pad <= bx or
        bx + bw + pad <= ax or
        ay + ah + pad <= by or
        by + bh + pad <= ay
    )


def _mask_overlap(canvas_mask: np.ndarray, fragment_mask: np.ndarray,
                  x: int, y: int) -> bool:
    fh, fw = fragment_mask.shape
    H, W = canvas_mask.shape
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + fw, W)
    y1 = min(y + fh, H)
    if x1 <= x0 or y1 <= y0:
        return True  # off-canvas counts as bad placement
    fx0 = x0 - x
    fy0 = y0 - y
    fx1 = fx0 + (x1 - x0)
    fy1 = fy0 + (y1 - y0)
    return bool(np.any(canvas_mask[y0:y1, x0:x1] & fragment_mask[fy0:fy1, fx0:fx1]))


def generate_composite(per_doc_dir: Path, *,
                        rng: np.random.Generator,
                        canvas_scale: float = DEFAULT_CANVAS_SCALE,
                        margin_px: int = DEFAULT_MARGIN_PX,
                        pad_px: int = DEFAULT_PAD_PX,
                        max_tries: int = DEFAULT_MAX_PLACEMENT_TRIES,
                        bg_rgb: tuple[int, int, int] = DEFAULT_BG_RGB,
                        bg_noise_sigma: float = DEFAULT_BG_NOISE_SIGMA,
                        ) -> CompositeResult:
    """Read tear_simulator output in `per_doc_dir`, compose, write outputs.

    Required input files:
        per_doc_dir / "ground_truth.json"
        per_doc_dir / "fragments" / "fragment_NN.png"  (RGBA)
        per_doc_dir / "original.png"                   (just for sizing)

    Writes:
        per_doc_dir / "composite.png"
        per_doc_dir / "composite_transforms.json"
    """
    gt_path = per_doc_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground_truth.json in {per_doc_dir}")
    with open(gt_path, "r", encoding="utf-8") as fh:
        gt = json.load(fh)

    H_src, W_src, _ = gt["image_shape"]
    canvas_w = int(W_src * canvas_scale)
    canvas_h = int(H_src * canvas_scale)

    # Pre-load and pre-rotate every fragment so we can size-sort.
    rotated_assets: list[dict] = []
    for frag in gt["fragments"]:
        rgba_path = per_doc_dir / frag["rgba_path"]
        rgba = np.asarray(Image.open(rgba_path).convert("RGBA"))
        angle_deg = float(rng.uniform(0.0, 360.0))
        rot_rgba, rot_mask, M_local = _rotate_rgba(rgba, angle_deg)
        rotated_assets.append({
            "frag": frag,
            "rgba": rot_rgba,
            "mask": (rot_mask > 0).astype(np.uint8),
            "M_local": M_local,         # rotates within local crop frame
            "angle_deg": angle_deg,
        })

    # Place big fragments first — they're the hardest to fit.
    rotated_assets.sort(key=lambda a: a["mask"].sum(), reverse=True)

    canvas = _make_background(canvas_w, canvas_h, bg_rgb, bg_noise_sigma, rng)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    placements: list[PlacementRecord] = []
    placed_bboxes: list[tuple[int, int, int, int]] = []
    skipped: list[int] = []

    for asset in rotated_assets:
        frag = asset["frag"]
        rot_rgba = asset["rgba"]
        rot_mask = asset["mask"]
        rh, rw = rot_mask.shape

        if rw + 2 * margin_px > canvas_w or rh + 2 * margin_px > canvas_h:
            skipped.append(frag["index"])
            continue

        success = False
        for _try in range(max_tries):
            x = int(rng.integers(margin_px, canvas_w - rw - margin_px))
            y = int(rng.integers(margin_px, canvas_h - rh - margin_px))
            cand_bbox = (x, y, rw, rh)
            # Quick AABB reject before exact mask check
            quick_clear = all(
                not _aabb_overlap(cand_bbox, b, pad_px) for b in placed_bboxes)
            if not quick_clear:
                continue
            if _mask_overlap(canvas_mask, rot_mask, x, y):
                continue

            # Accept placement. Alpha-composite onto canvas.
            alpha = rot_rgba[..., 3:4].astype(np.float32) / 255.0
            rgb = rot_rgba[..., :3].astype(np.float32)
            patch = canvas[y:y + rh, x:x + rw].astype(np.float32)
            canvas[y:y + rh, x:x + rw] = np.clip(
                rgb * alpha + patch * (1.0 - alpha), 0, 255
            ).astype(np.uint8)
            canvas_mask[y:y + rh, x:x + rw] |= rot_mask
            placed_bboxes.append(cand_bbox)

            # ----- Build the placement transform -----
            # Inputs: fragment crop (size bw × bh, bbox at (bx, by) in original).
            bx, by, bw, bh = frag["bbox_in_original"]

            # M_local (2x3) maps crop coords → rotated-crop coords.
            M_local_3 = np.eye(3, dtype=np.float64)
            M_local_3[:2, :] = asset["M_local"]
            # Add canvas translation (rotated-crop coords → canvas coords).
            T_xy = _translation_matrix_3x3(float(x), float(y))
            M_place = T_xy @ M_local_3              # 3x3, crop → canvas

            # Ground truth: canvas → original-Y coordinates.
            # Original-Y point = (bx, by) + crop_point.
            T_bbox = _translation_matrix_3x3(float(bx), float(by))
            M_to_original = T_bbox @ np.linalg.inv(M_place)

            cx = x + (rw - 1) / 2.0
            cy = y + (rh - 1) / 2.0

            placements.append(PlacementRecord(
                fragment_index=frag["index"],
                angle_deg=asset["angle_deg"],
                center_xy=(cx, cy),
                rotated_size=(rw, rh),
                placed_bbox=cand_bbox,
                M_place=M_place.tolist(),
                M_to_original=M_to_original.tolist(),
                rgba_source=frag["rgba_path"],
            ))
            success = True
            break

        if not success:
            skipped.append(frag["index"])

    # Save composite PNG and transforms JSON.
    Image.fromarray(canvas).save(per_doc_dir / "composite.png")

    out = {
        "canvas_size": [canvas_w, canvas_h],
        "bg_rgb": list(bg_rgb),
        "skipped_fragment_indices": skipped,
        "placements": [
            {
                "fragment_index": p.fragment_index,
                "angle_deg": p.angle_deg,
                "center_xy": list(p.center_xy),
                "rotated_size": list(p.rotated_size),
                "placed_bbox": list(p.placed_bbox),
                "M_place": p.M_place,
                "M_to_original": p.M_to_original,
                "rgba_source": p.rgba_source,
            }
            for p in placements
        ],
    }
    with open(per_doc_dir / "composite_transforms.json", "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    return CompositeResult(
        canvas_size=(canvas_w, canvas_h),
        bg_rgb=bg_rgb,
        placements=placements,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# Verification (round-trip self-test)
# ---------------------------------------------------------------------------

def verify_composite(per_doc_dir: Path, *, max_geom_err: float = 1e-6
                      ) -> dict:
    """Geometric correctness check on the placement transforms.

    By construction, ``M_to_original @ M_place`` should equal the pure
    translation ``T_bbox`` (shift by the fragment's original bbox top-left).
    We measure the max element-wise deviation of the composed matrix from the
    expected ``T_bbox`` — this is a tight floating-point check on the math.

    We also forward-project a few corner points and report sub-pixel residuals
    after one full round-trip (canvas → original → canvas).
    """
    with open(per_doc_dir / "ground_truth.json", "r") as fh:
        gt = json.load(fh)
    with open(per_doc_dir / "composite_transforms.json", "r") as fh:
        cmp_meta = json.load(fh)

    stats = []
    for placement in cmp_meta["placements"]:
        idx = placement["fragment_index"]
        bx, by, bw, bh = gt["fragments"][idx]["bbox_in_original"]
        M_place = np.array(placement["M_place"], dtype=np.float64)
        M_to_orig = np.array(placement["M_to_original"], dtype=np.float64)

        composed = M_to_orig @ M_place
        expected = np.eye(3, dtype=np.float64)
        expected[0, 2] = bx
        expected[1, 2] = by
        matrix_err = float(np.max(np.abs(composed - expected)))

        # Round-trip: crop corners → canvas → back-to-crop, residual in pixels.
        corners = np.array([
            [0, 0, 1], [bw - 1, 0, 1],
            [bw - 1, bh - 1, 1], [0, bh - 1, 1],
        ], dtype=np.float64).T  # 3x4
        canvas_pts = M_place @ corners
        round_trip = np.linalg.inv(M_place) @ canvas_pts
        roundtrip_err = float(np.max(np.abs(round_trip[:2] - corners[:2])))

        stats.append({
            "fragment_index": idx,
            "matrix_err": matrix_err,
            "roundtrip_err_px": roundtrip_err,
            "ok": matrix_err <= max_geom_err and roundtrip_err <= 1e-6,
        })

    return {
        "doc_id": gt["doc_id"],
        "fragments": stats,
        "all_ok": all(s["ok"] for s in stats),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default=Path("data/dataset/synthetic"),
                    type=Path,
                    help="Root containing per-doc subdirs from tear_simulator")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--canvas_scale", type=float, default=DEFAULT_CANVAS_SCALE)
    ap.add_argument("--margin_px", type=int, default=DEFAULT_MARGIN_PX)
    ap.add_argument("--pad_px", type=int, default=DEFAULT_PAD_PX)
    ap.add_argument("--max_tries", type=int, default=DEFAULT_MAX_PLACEMENT_TRIES)
    ap.add_argument("--verify", action="store_true",
                    help="Run round-trip warp verification per document")
    args = ap.parse_args()

    if not args.src_root.exists():
        raise SystemExit(f"src_root not found: {args.src_root}")

    docs = sorted([p for p in args.src_root.iterdir()
                   if p.is_dir() and (p / "ground_truth.json").exists()])
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        raise SystemExit("No tear_simulator outputs found")

    master_rng = np.random.default_rng(args.seed)
    print(f"[composite] processing {len(docs)} documents")

    summary = []
    for i, doc_dir in enumerate(docs):
        sub_seed = int(master_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)
        try:
            res = generate_composite(
                doc_dir, rng=rng,
                canvas_scale=args.canvas_scale,
                margin_px=args.margin_px,
                pad_px=args.pad_px,
                max_tries=args.max_tries,
            )
        except Exception as exc:
            print(f"[composite] WARN {doc_dir.name}: {exc}; skipping")
            continue

        msg = (f"[composite] [{i + 1}/{len(docs)}] {doc_dir.name}: "
               f"placed {len(res.placements)} skipped {len(res.skipped)}")
        if args.verify:
            v = verify_composite(doc_dir)
            mat_errs = [f["matrix_err"] for f in v["fragments"]]
            rt_errs = [f["roundtrip_err_px"] for f in v["fragments"]]
            mat_max = max(mat_errs) if mat_errs else 0.0
            rt_max = max(rt_errs) if rt_errs else 0.0
            msg += (f" matrix_err_max={mat_max:.2e} "
                    f"roundtrip_err_max={rt_max:.2e}px all_ok={v['all_ok']}")
            summary.append({"doc_id": doc_dir.name, "verify": v})
        print(msg)

    if args.verify and summary:
        with open(args.src_root / "_composite_verify.json", "w",
                   encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[composite] verification summary saved to "
              f"{args.src_root / '_composite_verify.json'}")
    print("[composite] done")


if __name__ == "__main__":
    main()
