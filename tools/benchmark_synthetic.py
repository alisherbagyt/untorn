"""Step 13 of Phase 5 — synthetic puzzle ablation benchmark.

Runs the assembly stage on held-out synthetic puzzles where ground-truth
fragment placements are known, and compares four pipeline variants:

    A — baseline    (no grid filter, no Siamese)
    B — grid only   (grid filter ON, Siamese OFF)
    C — siamese     (grid filter OFF, Siamese ON)
    D — hybrid      (grid filter ON, Siamese ON)

Per-puzzle metric: fragment placement accuracy. After registering the
predicted cluster against ground truth via the seed fragment's gt pose,
each fragment's predicted (R, t) is compared to its ground-truth (R, t).
A fragment is "placed correctly" if its translation error is <= 5 px AND
its angular error is <= 2 deg (matches the plan's success criterion).

The benchmark BYPASSES SAM: it builds the `fragments` list directly from
the synthetic-dataset RGBA crops + composite_transforms.json, isolating
Phase 3 (matcher quality) from Phase 1 (segmentation quality). For
end-to-end real-data validation, use ``tools/run_benchmark.py`` against
``data/input/*``.

Output JSON schema:
    {
      "n_docs":  int,
      "variants": {
        "A": {
          "n_correct_total": int,
          "n_total": int,
          "frag_accuracy": float,
          "median_translation_err_px": float,
          "median_angle_err_deg": float,
          "mean_seconds_per_doc": float,
          "per_doc": [{...}, ...]
        },
        "B": ..., "C": ..., "D": ...
      },
      "config": {... knobs at run time ...}
    }

CLI:
    python tools/benchmark_synthetic.py
        --index data/dataset/index.json
        --out   data/eval_results/synthetic_benchmark.json
        --max_docs 50
        --variants A,B,C,D
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from untorn import config as cfg                 # noqa: E402
from untorn import edge_matcher                  # noqa: E402
from untorn.contours import analyze_fragments    # noqa: E402
from untorn.assembly import reconstruct          # noqa: E402


SUCCESS_TRANS_PX = 5.0
SUCCESS_ANGLE_DEG = 2.0


# ─── Synthetic-fragment loader ─────────────────────────────────────────────

def _load_doc_fragments(doc_dir: Path) -> dict | None:
    """Build a `fragments` list (matching segmentation.py output) from one
    synthetic doc's composite + RGBA crops + placement metadata.

    Returns a dict with keys ``composite``, ``fragments``, ``gt_transforms``,
    ``canvas_size`` or None if the doc is unusable.
    """
    try:
        with open(doc_dir / "composite_transforms.json") as fh:
            cmp_meta = json.load(fh)
        composite = np.asarray(Image.open(doc_dir / "composite.png").convert("RGB"))
    except Exception:
        return None

    canvas_w, canvas_h = cmp_meta["canvas_size"]
    fragments: list[dict] = []
    gt_transforms: dict[int, np.ndarray] = {}

    for new_id, p in enumerate(cmp_meta["placements"]):
        try:
            rgba_path = doc_dir / p["rgba_source"]
            rgba = np.asarray(Image.open(rgba_path).convert("RGBA"))
        except Exception:
            continue
        bx, by, bw, bh = p["placed_bbox"]
        if bw <= 0 or bh <= 0:
            continue
        # Place the cropped alpha into a full-canvas mask (uint8 0/255).
        full_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        # Some PIL crops may have a slightly different shape after rotation;
        # clip to the canvas.
        h_crop, w_crop = rgba.shape[:2]
        x1 = min(bx + w_crop, canvas_w)
        y1 = min(by + h_crop, canvas_h)
        full_mask[by:y1, bx:x1] = (rgba[:y1 - by, :x1 - bx, 3] > 0).astype(np.uint8) * 255
        if int(full_mask.sum()) < 100:
            continue

        contours, _ = cv2.findContours(
            full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]; cy = M["m01"] / M["m00"]
        else:
            cx = x + ww / 2; cy = y + hh / 2

        fragments.append({
            "id":       new_id,
            "mask":     full_mask,
            "bbox":     (x, y, ww, hh),
            "area":     int((full_mask > 127).sum()),
            "contour":  contour,
            "centroid": np.array([cx, cy], dtype=np.float64),
            # Stash the source fragment_index so we can recover GT later.
            "_src_fragment_index": int(p["fragment_index"]),
        })
        # M_to_original maps canvas → original. The ground-truth "ideal"
        # transform IS this matrix.
        gt_transforms[new_id] = np.array(p["M_to_original"], dtype=np.float64)

    if len(fragments) < 2:
        return None

    return {
        "composite":     composite,
        "fragments":     fragments,
        "gt_transforms": gt_transforms,
        "canvas_size":   (canvas_w, canvas_h),
    }


# ─── Pose-error computation ────────────────────────────────────────────────

def _affine_angle_translation(M: np.ndarray) -> tuple[float, float, float]:
    return (float(np.arctan2(M[1, 0], M[0, 0])),
            float(M[0, 2]), float(M[1, 2]))


def _per_fragment_errors(predicted: dict[int, np.ndarray],
                          gt: dict[int, np.ndarray],
                          seed_id: int) -> list[dict]:
    """Compare predicted (canvas → reconstructed) transforms against GT
    (canvas → original), after registering both clusters via the seed.

    Predicted seed gets identity by construction. We align clusters by
    composing predicted with gt_transforms[seed_id] (the seed's true
    canvas→original pose), then measure each fragment's residual.
    """
    if seed_id not in gt:
        return []
    M_align = gt[seed_id]
    out = []
    for fid, M_pred in predicted.items():
        if fid not in gt:
            continue
        # Register predicted into the original frame.
        M_pred_aligned = M_align @ M_pred
        ang_p, tx_p, ty_p = _affine_angle_translation(M_pred_aligned)
        ang_g, tx_g, ty_g = _affine_angle_translation(gt[fid])
        # Wrap angle difference into [-pi, pi].
        d_ang = (ang_p - ang_g + np.pi) % (2 * np.pi) - np.pi
        d_t = float(np.hypot(tx_p - tx_g, ty_p - ty_g))
        out.append({
            "fragment_id": int(fid),
            "translation_err_px": d_t,
            "angle_err_deg": float(abs(np.degrees(d_ang))),
            "moved": bool(not np.allclose(M_pred, np.eye(3), atol=1e-6)),
        })
    return out


def _seed_id_from_predicted(predicted: dict[int, np.ndarray]) -> int | None:
    """The pipeline pins one fragment at identity (the seed). Find it."""
    for fid, M in predicted.items():
        if np.allclose(M, np.eye(3), atol=1e-6):
            return int(fid)
    return None


# ─── Variant control: cfg knob overrides ───────────────────────────────────

@dataclass(frozen=True)
class Variant:
    label: str
    grid_filter: bool
    siamese: bool


VARIANTS = {
    "A": Variant("baseline",     grid_filter=False, siamese=False),
    "B": Variant("grid_only",    grid_filter=True,  siamese=False),
    "C": Variant("siamese_only", grid_filter=False, siamese=True),
    "D": Variant("hybrid",       grid_filter=True,  siamese=True),
}


def _set_variant(v: Variant) -> None:
    cfg.GRID_FILTER_ENABLED  = bool(v.grid_filter)
    cfg.EDGE_MATCHER_ENABLED = bool(v.siamese)


def _ensure_siamese_lifecycle(want_loaded: bool) -> None:
    if want_loaded and not edge_matcher.is_loaded():
        # Reset sticky failure flags so a re-attempt actually fires.
        edge_matcher._LOAD_FAILED = False
        edge_matcher._LOAD_REASON = ""
        edge_matcher.load()
    if not want_loaded and edge_matcher.is_loaded():
        edge_matcher.unload()


# ─── Per-doc, per-variant runner ───────────────────────────────────────────

def _run_one_variant(doc_data: dict, variant: Variant,
                      doc_id: str, debug_root: Path) -> dict:
    """Run one (doc, variant) trial. Returns metrics + timing."""
    composite = doc_data["composite"]
    # Deep-copy fragments so each variant starts fresh (analyze_fragments
    # mutates the dict by attaching curvature/SDF/etc).
    fragments = deepcopy(doc_data["fragments"])
    gt = doc_data["gt_transforms"]

    _set_variant(variant)
    _ensure_siamese_lifecycle(variant.siamese)

    debug_dir = debug_root / doc_id / variant.label
    debug_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        # Phase 2: feature extraction (subpixel boundary, text lines,
        # support points, color profile, SDF).
        analyze_fragments(fragments, composite, debug_dir)
        # Phase 3: assembly.
        predicted = reconstruct(fragments, composite, debug_dir)
    except Exception as exc:
        return {
            "variant":   variant.label,
            "ok":        False,
            "error":     repr(exc),
            "seconds":   time.time() - t0,
            "n_fragments": len(fragments),
        }
    elapsed = time.time() - t0

    seed_id = _seed_id_from_predicted(predicted)
    if seed_id is None:
        return {
            "variant":   variant.label,
            "ok":        False,
            "error":     "no seed fragment found",
            "seconds":   elapsed,
            "n_fragments": len(fragments),
        }

    errs = _per_fragment_errors(predicted, gt, seed_id)
    if not errs:
        return {
            "variant":   variant.label,
            "ok":        False,
            "error":     "no overlap between predicted and gt",
            "seconds":   elapsed,
            "n_fragments": len(fragments),
        }

    n_correct = sum(
        1 for e in errs
        if e["translation_err_px"] <= SUCCESS_TRANS_PX
        and e["angle_err_deg"]      <= SUCCESS_ANGLE_DEG
    )
    n_moved = sum(1 for e in errs if e["moved"])
    return {
        "variant":     variant.label,
        "ok":          True,
        "seconds":     elapsed,
        "n_fragments": len(fragments),
        "n_placed":    n_moved + 1,    # seed counts as placed
        "n_correct":   n_correct,
        "median_translation_err_px": float(np.median(
            [e["translation_err_px"] for e in errs])),
        "median_angle_err_deg":      float(np.median(
            [e["angle_err_deg"] for e in errs])),
        "p90_translation_err_px":    float(np.percentile(
            [e["translation_err_px"] for e in errs], 90)),
        "p90_angle_err_deg":         float(np.percentile(
            [e["angle_err_deg"] for e in errs], 90)),
        "seed_id":     int(seed_id),
        "frag_errors": errs,
    }


# ─── Aggregation ───────────────────────────────────────────────────────────

def _summarize(per_doc: list[dict]) -> dict:
    ok = [r for r in per_doc if r.get("ok")]
    n_total = sum(r["n_fragments"] for r in ok)
    n_correct_total = sum(r["n_correct"] for r in ok)
    n_placed_total  = sum(r["n_placed"]  for r in ok)
    if not ok:
        return {
            "n_docs": len(per_doc),
            "n_docs_ok": 0,
            "frag_accuracy": float("nan"),
            "median_translation_err_px": float("nan"),
            "median_angle_err_deg": float("nan"),
            "mean_seconds_per_doc": float(np.mean([r["seconds"]
                                                    for r in per_doc])),
            "per_doc": per_doc,
        }
    all_t = []
    all_a = []
    for r in ok:
        all_t.extend([e["translation_err_px"] for e in r["frag_errors"]])
        all_a.extend([e["angle_err_deg"]      for e in r["frag_errors"]])
    return {
        "n_docs": len(per_doc),
        "n_docs_ok": len(ok),
        "n_fragments_total": n_total,
        "n_placed_total":    n_placed_total,
        "n_correct_total":   n_correct_total,
        "frag_accuracy": float(n_correct_total / max(n_total, 1)),
        "placement_rate": float(n_placed_total / max(n_total, 1)),
        "median_translation_err_px": float(np.median(all_t)),
        "median_angle_err_deg":      float(np.median(all_a)),
        "p90_translation_err_px":    float(np.percentile(all_t, 90)),
        "p90_angle_err_deg":         float(np.percentile(all_a, 90)),
        "mean_seconds_per_doc": float(np.mean([r["seconds"]
                                                for r in per_doc])),
        "per_doc": per_doc,
    }


# ─── Driver ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path,
                     default=ROOT / "data" / "dataset" / "index.json")
    ap.add_argument("--out", type=Path,
                     default=ROOT / "data" / "eval_results"
                          / "synthetic_benchmark.json")
    ap.add_argument("--max_docs", type=int, default=20,
                     help="Cap on validation puzzles per variant.")
    ap.add_argument("--variants", type=str, default="A,B,C,D",
                     help="Comma-separated subset of {A,B,C,D}.")
    ap.add_argument("--debug_root", type=Path,
                     default=ROOT / "data" / "debug" / "synthetic_benchmark")
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--rotation_cap_deg", type=float, default=None,
                     help="Override cfg.RECON_MAX_ROTATION_DEG for this run "
                          "only. Synthetic puzzles use uniform 0-360 deg "
                          "rotation, so the production default of 30 deg "
                          "rejects most true matches; pass 180 here to "
                          "stress-test the matcher on rotation-invariant "
                          "data.")
    args = ap.parse_args()

    if not args.index.exists():
        raise SystemExit(f"index not found: {args.index}")
    with open(args.index) as fh:
        master = json.load(fh)
    synthetic_root = (ROOT / master["synthetic_root"]).resolve()
    val = master.get("val") or []
    if not val:
        raise SystemExit("val split is empty in index.json")

    rng = np.random.default_rng(args.seed)
    chosen = list(rng.permutation(len(val))[:args.max_docs])
    chosen_docs = [val[i] for i in chosen]
    print(f"[bench] {len(chosen_docs)} val docs sampled "
          f"(of {len(val)} available)")

    requested = [v.strip().upper() for v in args.variants.split(",")]
    requested = [v for v in requested if v in VARIANTS]
    if not requested:
        raise SystemExit(f"no valid variants: {args.variants}")

    args.debug_root.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Save and restore the runtime-mutable cfg knobs we tweak.
    saved_cfg = {
        "GRID_FILTER_ENABLED":   cfg.GRID_FILTER_ENABLED,
        "EDGE_MATCHER_ENABLED":  cfg.EDGE_MATCHER_ENABLED,
        "RECON_MAX_ROTATION_DEG": cfg.RECON_MAX_ROTATION_DEG,
    }
    if args.rotation_cap_deg is not None:
        cfg.RECON_MAX_ROTATION_DEG = float(args.rotation_cap_deg)
        print(f"[bench] RECON_MAX_ROTATION_DEG overridden -> "
              f"{cfg.RECON_MAX_ROTATION_DEG:.1f} deg")

    results: dict[str, dict] = {}
    try:
        for v_key in requested:
            variant = VARIANTS[v_key]
            print(f"\n[bench] === variant {v_key}: {variant.label} "
                  f"(grid={variant.grid_filter}, siamese={variant.siamese}) ===")
            per_doc: list[dict] = []
            for d_idx, entry in enumerate(chosen_docs):
                doc_id = entry["doc_id"]
                doc_dir = synthetic_root / doc_id
                doc_data = _load_doc_fragments(doc_dir)
                if doc_data is None:
                    per_doc.append({
                        "doc_id": doc_id, "ok": False,
                        "error": "load failed",
                        "seconds": 0.0, "n_fragments": 0,
                    })
                    print(f"  [{d_idx + 1}/{len(chosen_docs)}] {doc_id}: "
                          f"load failed — skipped")
                    continue
                trial = _run_one_variant(doc_data, variant, doc_id,
                                           args.debug_root)
                trial["doc_id"] = doc_id
                per_doc.append(trial)
                if trial.get("ok"):
                    print(f"  [{d_idx + 1}/{len(chosen_docs)}] {doc_id}: "
                          f"{trial['n_correct']}/{trial['n_fragments']} correct  "
                          f"med_t={trial['median_translation_err_px']:.1f}px  "
                          f"med_a={trial['median_angle_err_deg']:.2f}deg  "
                          f"{trial['seconds']:.1f}s")
                else:
                    print(f"  [{d_idx + 1}/{len(chosen_docs)}] {doc_id}: "
                          f"FAILED ({trial.get('error', '?')})")
            results[v_key] = _summarize(per_doc)
    finally:
        # Restore cfg so subsequent imports/uses see the user-configured state.
        for k, v in saved_cfg.items():
            setattr(cfg, k, v)
        edge_matcher.unload()

    # Print summary table
    print("\n" + "=" * 78)
    print(f"{'variant':<14} {'frag_acc':>9} {'med_t_px':>9} "
          f"{'med_a_deg':>10} {'p90_t_px':>9} {'p90_a_deg':>10} "
          f"{'mean_s':>8}")
    print("-" * 78)
    for v_key in requested:
        s = results[v_key]
        v = VARIANTS[v_key]
        print(f"{v_key} {v.label:<12}"
              f" {s['frag_accuracy']:>9.3f}"
              f" {s['median_translation_err_px']:>9.2f}"
              f" {s['median_angle_err_deg']:>10.2f}"
              f" {s['p90_translation_err_px']:>9.2f}"
              f" {s['p90_angle_err_deg']:>10.2f}"
              f" {s['mean_seconds_per_doc']:>8.1f}")
    print("=" * 78)

    payload = {
        "n_docs":     len(chosen_docs),
        "doc_ids":    [entry["doc_id"] for entry in chosen_docs],
        "variants":   results,
        "config": {
            "MIN_FRAGMENT_AREA_FRAC":  cfg.MIN_FRAGMENT_AREA_FRAC,
            "GRID_FILTER_TOP_K":       cfg.GRID_FILTER_TOP_K,
            "EDGE_MATCHER_MIN_SCORE":  cfg.EDGE_MATCHER_MIN_SCORE,
            "ASSEMBLY_MIN_CONFIDENCE": cfg.ASSEMBLY_MIN_CONFIDENCE,
            "RECON_MAX_ROTATION_DEG":  cfg.RECON_MAX_ROTATION_DEG,
        },
        "success_thresholds": {
            "translation_px": SUCCESS_TRANS_PX,
            "angle_deg":      SUCCESS_ANGLE_DEG,
        },
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
