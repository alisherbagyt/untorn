"""Step 2 of Phase 1 — synthetic torn-paper generator.

Voronoi tessellation with smooth-noise-perturbed coordinates produces fragment
masks whose torn boundaries are MUTUALLY CONSISTENT (any two adjacent fragments
share the *exact* same jagged boundary, so reassembly is pixel-perfect ground
truth).

The trick: instead of jaggedying each Voronoi edge after the fact (which would
require careful matching of two sides), we perturb the *query coordinate* per
pixel:

    region(x, y) = argmin_i  || (x + nx(x,y),  y + ny(x,y)) - seed_i ||

where nx, ny are gaussian-blurred random fields. Adjacent pixels see nearly the
same perturbation → coherent jaggedness; both sides of any boundary use the
same noise → the boundary is automatically shared.

Public API:
    simulate_tear(image_rgb, *, n_fragments, rng, ...) -> TearResult

CLI:
    python tools/tear_simulator.py --src <png-or-dir> --out_root data/dataset/synthetic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_N_RANGE = (5, 12)              # uniform sample inclusive
DEFAULT_NOISE_AMPLITUDE_PX = 14.0      # how far pixels can be displaced
DEFAULT_NOISE_BLUR_SIGMA = 18.0        # smoothness of the displacement field
DEFAULT_MIN_FRAGMENT_AREA_FRAC = 0.012  # discard fragments smaller than this
SEED_MARGIN_FRAC = 0.06                # keep seeds away from image border


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FragmentRecord:
    index: int                       # 0..N-1 within this document
    region_id: int                   # original Voronoi region id
    bbox: tuple[int, int, int, int]  # (x0, y0, w, h) in source-image space
    area_px: int                     # mask area in pixels
    neighbors: list[int] = field(default_factory=list)  # other fragment.index
    rgba_path: str | None = None     # written by save()
    mask_path: str | None = None
    seed_xy: tuple[float, float] | None = None


@dataclass
class TearResult:
    image_shape: tuple[int, int, int]    # (H, W, 3) of source
    fragments: list[FragmentRecord]
    region_map: np.ndarray               # (H, W) int, fragment.index per pixel; -1 = no fragment
    rgba_per_fragment: list[np.ndarray]  # cropped RGBA arrays, parallel to fragments
    mask_per_fragment: list[np.ndarray]  # cropped binary masks, parallel to fragments


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _smooth_noise_field(shape: tuple[int, int], sigma: float, amplitude: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Return a gaussian-blurred noise field, normalized to amplitude in pixels."""
    raw = rng.standard_normal(shape).astype(np.float32)
    # GaussianBlur ksize=0 -> derived from sigma. Need odd kernel >= 1.
    blurred = cv2.GaussianBlur(raw, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT)
    std = blurred.std()
    if std < 1e-6:
        return np.zeros_like(blurred)
    return (blurred / std) * amplitude


def _build_region_map(H: int, W: int, seeds: np.ndarray, *,
                       noise_amp_px: float, noise_blur_sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Per-pixel Voronoi assignment with noise-perturbed query coordinates."""
    nx = _smooth_noise_field((H, W), noise_blur_sigma, noise_amp_px, rng)
    ny = _smooth_noise_field((H, W), noise_blur_sigma, noise_amp_px, rng)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    qx = xx + nx
    qy = yy + ny

    # Distance to each seed: shape (n_seeds, H, W)
    n = seeds.shape[0]
    sq = np.empty((n, H, W), dtype=np.float32)
    for i in range(n):
        sq[i] = (qx - seeds[i, 0]) ** 2 + (qy - seeds[i, 1]) ** 2
    return np.argmin(sq, axis=0).astype(np.int32)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected blob in a binary mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask.astype(bool)
    # stats[0] is background (label 0). Pick largest non-background.
    sizes = stats[1:, cv2.CC_STAT_AREA]
    if len(sizes) == 0:
        return np.zeros_like(mask, dtype=bool)
    biggest_label = 1 + int(np.argmax(sizes))
    return labels == biggest_label


def _compute_neighbors(region_map: np.ndarray, valid_ids: set[int]
                       ) -> dict[int, set[int]]:
    """Build adjacency graph: region i is neighbor of j if any 4-connected
    pixel pair has labels (i, j)."""
    H, W = region_map.shape
    adj: dict[int, set[int]] = {rid: set() for rid in valid_ids}

    a = region_map[:, :-1]
    b = region_map[:, 1:]
    diff_mask = a != b
    pairs_h = np.stack([a[diff_mask], b[diff_mask]], axis=1)

    a = region_map[:-1, :]
    b = region_map[1:, :]
    diff_mask = a != b
    pairs_v = np.stack([a[diff_mask], b[diff_mask]], axis=1)

    pairs = np.concatenate([pairs_h, pairs_v], axis=0)
    for u, v in pairs:
        u, v = int(u), int(v)
        if u in adj and v in adj and u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def simulate_tear(image_rgb: np.ndarray, *,
                  rng: np.random.Generator,
                  n_fragments_range: tuple[int, int] = DEFAULT_N_RANGE,
                  noise_amp_px: float = DEFAULT_NOISE_AMPLITUDE_PX,
                  noise_blur_sigma: float = DEFAULT_NOISE_BLUR_SIGMA,
                  min_area_frac: float = DEFAULT_MIN_FRAGMENT_AREA_FRAC,
                  ) -> TearResult:
    """Tear the source image into N fragments with mutually-consistent torn edges.

    Args:
        image_rgb: (H, W, 3) uint8 RGB array.
        rng: numpy Generator for all randomness.
        n_fragments_range: inclusive (low, high) for sampling fragment count.
        noise_amp_px: how jagged the tear lines are (pixel scale).
        noise_blur_sigma: smoothness of jaggedness (px).
        min_area_frac: drop fragments smaller than this fraction of image area.
    """
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3:
        raise ValueError("image_rgb must be uint8 (H, W, 3) RGB")
    H, W = image_rgb.shape[:2]
    total_area = H * W

    n_low, n_high = n_fragments_range
    n_seeds = int(rng.integers(n_low, n_high + 1))

    margin_x = int(W * SEED_MARGIN_FRAC)
    margin_y = int(H * SEED_MARGIN_FRAC)
    seeds = np.column_stack([
        rng.uniform(margin_x, W - margin_x, size=n_seeds),
        rng.uniform(margin_y, H - margin_y, size=n_seeds),
    ]).astype(np.float32)

    region_map = _build_region_map(
        H, W, seeds,
        noise_amp_px=noise_amp_px,
        noise_blur_sigma=noise_blur_sigma,
        rng=rng,
    )

    # Filter: drop tiny regions; keep largest connected component per region;
    # remap surviving regions to consecutive fragment indices.
    min_area = int(total_area * min_area_frac)
    surviving: list[tuple[int, np.ndarray]] = []
    for rid in range(n_seeds):
        raw_mask = region_map == rid
        if raw_mask.sum() < min_area:
            continue
        cc_mask = _largest_connected_component(raw_mask)
        if cc_mask.sum() < min_area:
            continue
        surviving.append((rid, cc_mask))

    if not surviving:
        raise RuntimeError("Tear simulator produced no valid fragments — "
                           "try smaller min_area_frac or fewer seeds.")

    # Build the final fragment-indexed region map.
    fragment_region_map = np.full((H, W), -1, dtype=np.int32)
    for new_idx, (_rid, mask) in enumerate(surviving):
        fragment_region_map[mask] = new_idx

    # Adjacency among fragment indices.
    valid_ids = set(range(len(surviving)))
    adj = _compute_neighbors(fragment_region_map, valid_ids)

    fragments: list[FragmentRecord] = []
    rgba_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []

    for new_idx, (rid, mask) in enumerate(surviving):
        ys, xs = np.nonzero(mask)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        bw, bh = x1 - x0, y1 - y0
        crop_img = image_rgb[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]

        rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
        rgba[..., :3] = crop_img
        rgba[..., 3] = crop_mask.astype(np.uint8) * 255

        rec = FragmentRecord(
            index=new_idx,
            region_id=int(rid),
            bbox=(x0, y0, bw, bh),
            area_px=int(mask.sum()),
            neighbors=sorted(adj[new_idx]),
            seed_xy=(float(seeds[rid, 0]), float(seeds[rid, 1])),
        )
        fragments.append(rec)
        rgba_list.append(rgba)
        mask_list.append((crop_mask.astype(np.uint8) * 255))

    return TearResult(
        image_shape=(H, W, 3),
        fragments=fragments,
        region_map=fragment_region_map,
        rgba_per_fragment=rgba_list,
        mask_per_fragment=mask_list,
    )


# ---------------------------------------------------------------------------
# Saving / debug
# ---------------------------------------------------------------------------

def save_tear_result(result: TearResult, out_dir: Path, *,
                     source_image_rgb: np.ndarray, doc_id: int | str,
                     save_debug: bool = True) -> dict:
    """Write fragments, ground_truth.json, and (optionally) debug visualization.

    Returns the ground-truth dict that was written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frag_dir = out_dir / "fragments"
    frag_dir.mkdir(exist_ok=True)

    # Save the original (Y) for convenience.
    Image.fromarray(source_image_rgb).save(out_dir / "original.png")

    # Save each fragment as RGBA.
    fragment_entries = []
    for rec, rgba in zip(result.fragments, result.rgba_per_fragment):
        fname = f"fragment_{rec.index:02d}.png"
        Image.fromarray(rgba, mode="RGBA").save(frag_dir / fname)
        rec.rgba_path = f"fragments/{fname}"
        fragment_entries.append({
            "index": rec.index,
            "region_id": rec.region_id,
            "bbox_in_original": list(rec.bbox),  # (x, y, w, h)
            "area_px": rec.area_px,
            "neighbors": rec.neighbors,
            "seed_xy": list(rec.seed_xy),
            "rgba_path": rec.rgba_path,
        })

    gt = {
        "doc_id": doc_id,
        "image_shape": list(result.image_shape),  # H, W, C
        "fragments": fragment_entries,
    }
    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=2)

    if save_debug:
        _save_debug_visualization(result, source_image_rgb, out_dir)

    return gt


def _save_debug_visualization(result: TearResult, source_rgb: np.ndarray,
                               out_dir: Path) -> None:
    """Save a colored overlay so a human can quickly see the fracture pattern."""
    H, W = result.image_shape[:2]
    rng = np.random.default_rng(42)  # deterministic colors
    palette = (rng.uniform(50, 230, size=(len(result.fragments), 3))
               .astype(np.uint8))

    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for idx in range(len(result.fragments)):
        overlay[result.region_map == idx] = palette[idx]

    blend = cv2.addWeighted(source_rgb, 0.5, overlay, 0.5, 0)

    # Outline fragments in black.
    edges = np.zeros((H, W), dtype=np.uint8)
    for idx in range(len(result.fragments)):
        m = (result.region_map == idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(edges, contours, -1, 255, 1)
    blend[edges > 0] = (0, 0, 0)

    Image.fromarray(blend).save(out_dir / "_debug_fracture.png")


# ---------------------------------------------------------------------------
# Batch driver / CLI
# ---------------------------------------------------------------------------

def _iter_source_images(src: Path) -> Iterable[Path]:
    if src.is_file():
        yield src
        return
    for p in sorted(src.glob("*.png")):
        if p.name == "index.json":
            continue
        yield p


def _doc_id_from_path(path: Path) -> str:
    return path.stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Source PNG file or directory")
    ap.add_argument("--out_root", default=Path("data/dataset/synthetic"),
                    type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of source images processed")
    ap.add_argument("--seed", type=int, default=20260426,
                    help="Master RNG seed")
    ap.add_argument("--n_min", type=int, default=DEFAULT_N_RANGE[0])
    ap.add_argument("--n_max", type=int, default=DEFAULT_N_RANGE[1])
    ap.add_argument("--noise_amp", type=float, default=DEFAULT_NOISE_AMPLITUDE_PX)
    ap.add_argument("--noise_sigma", type=float, default=DEFAULT_NOISE_BLUR_SIGMA)
    ap.add_argument("--min_area_frac", type=float, default=DEFAULT_MIN_FRAGMENT_AREA_FRAC)
    ap.add_argument("--no_debug", action="store_true",
                    help="Skip the colored fracture overlay PNG")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source not found: {args.src}")

    sources = list(_iter_source_images(args.src))
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit(f"No PNG files found at {args.src}")

    master_rng = np.random.default_rng(args.seed)
    print(f"[tear] processing {len(sources)} source images")

    for i, src_path in enumerate(sources):
        doc_id = _doc_id_from_path(src_path)
        out_dir = args.out_root / doc_id
        # Per-image RNG derived from master seed for reproducibility.
        sub_seed = int(master_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)

        pil = Image.open(src_path).convert("RGB")
        image_rgb = np.asarray(pil)
        try:
            result = simulate_tear(
                image_rgb, rng=rng,
                n_fragments_range=(args.n_min, args.n_max),
                noise_amp_px=args.noise_amp,
                noise_blur_sigma=args.noise_sigma,
                min_area_frac=args.min_area_frac,
            )
        except RuntimeError as exc:
            print(f"[tear] WARN {src_path.name}: {exc}; skipping")
            continue

        save_tear_result(result, out_dir,
                          source_image_rgb=image_rgb,
                          doc_id=doc_id,
                          save_debug=not args.no_debug)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[tear] [{i + 1}/{len(sources)}] {doc_id}: "
                  f"{len(result.fragments)} fragments")

    print("[tear] done")


if __name__ == "__main__":
    main()
