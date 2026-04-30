"""
untorn.benchmark.run_suite
==========================
End-to-end benchmark runner:

    1. Generate N synthetic torn-paper cases (Voronoi + roughened tears).
    2. Run the full UNTORN pipeline on each.
    3. Extract predicted transforms + fragment centroids from the debug dirs.
    4. Score against ground truth.
    5. Aggregate + pretty-print metrics; write `benchmark_report.json`.

Typical use:

    python tools/run_benchmark.py --source data/input/clean_doc.jpg \
        --n-cases 5 --n-fragments 8 --out data/benchmark/run_01
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from .generate import generate_case, TearGeneratorConfig
from .evaluate import (
    evaluate_case, CaseMetrics, _apply_affine_pts, _umeyama_2d,
)


# ───────────────────────────────────────────────────────────────────────────
#  Predicted-transform extraction from pipeline debug output
# ───────────────────────────────────────────────────────────────────────────

def _load_predicted_transforms(pipeline_debug_dir: Path) -> dict:
    """
    Reconstruct 3x3 affines per fragment from the pipeline's debug output.

    Prefers `reconstruction/final_transforms.json` (canonical 3x3 affine
    per fragment, written verbatim by ``assembly.reconstruct``). Falls
    back to `reconstruction/final_translations.json` (angle_deg + dx + dy
    summary) for backwards compatibility with older debug dirs and for
    runs where only the summary is present.
    """
    transforms_path = pipeline_debug_dir / "reconstruction" / "final_transforms.json"
    if transforms_path.exists():
        try:
            data = json.loads(transforms_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            out: dict[int, np.ndarray] = {}
            for fid_str, M_list in data.items():
                try:
                    M = np.asarray(M_list, dtype=np.float64)
                except Exception:
                    continue
                if M.shape == (3, 3):
                    out[int(fid_str)] = M
            if out:
                return out

    p = pipeline_debug_dir / "reconstruction" / "final_translations.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for fid_str, rec in data.items():
        ang = np.deg2rad(float(rec.get("angle_deg", 0.0)))
        dx = float(rec.get("dx", 0.0))
        dy = float(rec.get("dy", 0.0))
        M = np.array([
            [np.cos(ang), -np.sin(ang), dx],
            [np.sin(ang),  np.cos(ang), dy],
            [0,            0,           1],
        ], dtype=np.float64)
        out[int(fid_str)] = M
    return out


def _load_predicted_centroids(pipeline_debug_dir: Path) -> dict:
    """
    Centroid of each segmented fragment in the INPUT image frame
    (working-resolution coordinates, same frame as final_translations.json).

    Reads from segmentation/fragments_meta.json.
    """
    meta = pipeline_debug_dir / "segmentation" / "fragments_meta.json"
    if meta.exists():
        data = json.loads(meta.read_text(encoding="utf-8"))
        out = {}
        for row in data:
            fid = int(row["id"])
            if "centroid" in row:
                out[fid] = tuple(row["centroid"])
            elif "bbox_xywh" in row:
                x, y, w, h = row["bbox_xywh"]
                out[fid] = (float(x + w / 2), float(y + h / 2))
        if out:
            return out

    # Fallback: scan 05_mask_final_*.png debug artefacts
    centroids = {}
    mask_dir = pipeline_debug_dir / "segmentation"
    if mask_dir.exists():
        for mask_file in sorted(mask_dir.glob("05_mask_final_*.png")):
            try:
                fid = int(mask_file.stem.split("_")[-1])
            except ValueError:
                continue
            m = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            M = cv2.moments(m > 127, binaryImage=True)
            if M["m00"] == 0:
                continue
            centroids[fid] = (float(M["m10"] / M["m00"]),
                              float(M["m01"] / M["m00"]))
    return centroids


def _load_pipeline_scale(pipeline_debug_dir: Path) -> float:
    """Read the work→full scale factor written by the pipeline."""
    meta_path = pipeline_debug_dir / "pipeline_meta.json"
    if not meta_path.exists():
        return 1.0
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return float(data.get("scale_factor", 1.0))


# ───────────────────────────────────────────────────────────────────────────
#  Per-case diagnostic render
# ───────────────────────────────────────────────────────────────────────────

def _render_diagnostic(truth_path: Path,
                       predicted_transforms: dict,
                       predicted_centroids: dict,
                       working_scale_factor: float,
                       out_path: Path) -> None:
    """
    Three-pane diagnostic:
        LEFT  — ground-truth source image with GT fragment centroids.
        MID   — synthetic scan with predicted fragment centroids (raw).
        RIGHT — ground-truth source image with predicted centroids AFTER
                the global Umeyama alignment; coloured lines connect each
                predicted→matched-GT centroid. Short lines = good fit.
    """
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    src_path  = Path(truth["source_path"])
    scan_path = Path(truth["scan_path"])
    src  = cv2.imread(str(src_path))
    scan = cv2.imread(str(scan_path))
    if src is None or scan is None:
        raise FileNotFoundError("source/scan missing for diagnostic")

    s = float(working_scale_factor)

    # GT centroids in both frames
    gt_src_cs, gt_can_cs = [], []
    for g in truth["fragments"]:
        M = np.array(g["affine_source_to_canvas"], dtype=np.float64)
        sc = np.array(g["centroid_in_source"], dtype=np.float64)
        cc = _apply_affine_pts(np.vstack([M, [0, 0, 1]]),
                               sc.reshape(1, 2))[0]
        gt_src_cs.append(sc)
        gt_can_cs.append(cc)
    gt_src_cs = np.asarray(gt_src_cs)
    gt_can_cs = np.asarray(gt_can_cs)

    pred_ids = sorted(predicted_centroids.keys())
    # Predicted centroids in INPUT/working-res scan
    pred_in_work = np.asarray([predicted_centroids[i] for i in pred_ids])
    pred_in_full = pred_in_work * s

    # Greedy match predicted → GT by full-res scan centroid
    used = set()
    match = {}
    for pi, pf in zip(pred_ids, pred_in_full):
        d = np.linalg.norm(gt_can_cs - pf, axis=1)
        for gi in np.argsort(d):
            if int(gi) not in used:
                match[pi] = int(gi)
                used.add(int(gi))
                break
        else:
            match[pi] = None

    # Pipeline output canvas centroid for each matched frag
    pred_out = []
    gt_src_pair = []
    for pi in pred_ids:
        gi = match.get(pi)
        if gi is None:
            continue
        T = np.asarray(predicted_transforms[pi], dtype=np.float64)
        if T.shape == (2, 3):
            T = np.vstack([T, [0, 0, 1]])
        pc = _apply_affine_pts(T, pred_in_work[pred_ids.index(pi)].reshape(1, 2))[0]
        pred_out.append(pc)
        gt_src_pair.append(gt_src_cs[gi])

    if len(pred_out) < 2:
        # Fallback: just stack the raw scan + source
        h = max(src.shape[0], scan.shape[0])
        w = src.shape[1] + scan.shape[1] + 20
        panel = np.full((h, w, 3), 30, dtype=np.uint8)
        panel[:src.shape[0], :src.shape[1]] = src
        panel[:scan.shape[0], src.shape[1] + 20:] = scan
        cv2.imwrite(str(out_path), panel)
        return

    pred_out = np.asarray(pred_out)
    gt_src_pair = np.asarray(gt_src_pair)
    M_align = _umeyama_2d(pred_out, gt_src_pair)
    pred_aligned = _apply_affine_pts(M_align, pred_out)

    # Compose three equally-wide panels at src height
    H = src.shape[0]
    def _fit(img, target_h):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(round(w * scale)), target_h))
    src1 = src.copy()
    scan1 = _fit(scan, H)
    src2 = src.copy()

    # LEFT: GT source centroids
    for i, c in enumerate(gt_src_cs):
        cv2.circle(src1, (int(c[0]), int(c[1])), 7, (0, 255, 0), -1)
        cv2.putText(src1, f"{i}", (int(c[0]) + 8, int(c[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # MIDDLE: scan with predicted centroids
    scan_scale_y = scan1.shape[0] / scan.shape[0]
    scan_scale_x = scan1.shape[1] / scan.shape[1]
    for pi, pf in zip(pred_ids, pred_in_full):
        cv2.circle(scan1,
                   (int(pf[0] * scan_scale_x), int(pf[1] * scan_scale_y)),
                   7, (0, 200, 255), -1)
        cv2.putText(scan1, f"{pi}",
                    (int(pf[0] * scan_scale_x) + 8,
                     int(pf[1] * scan_scale_y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # RIGHT: source with predicted-aligned centroids + link to GT
    for pa, gt_c in zip(pred_aligned, gt_src_pair):
        pa_pt = (int(pa[0]), int(pa[1]))
        g_pt  = (int(gt_c[0]), int(gt_c[1]))
        cv2.line(src2, pa_pt, g_pt, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(src2, g_pt, 6, (0, 255, 0), 2)
        cv2.circle(src2, pa_pt, 6, (0, 0, 255), -1)

    # Legends
    def _label(img, text):
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (15, 15, 15), -1)
        cv2.putText(img, text, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    _label(src1, "GT source + GT centroids")
    _label(scan1, "Input scan + predicted centroids")
    _label(src2, "Source + predicted (aligned) vs GT (green)")

    pad = 20
    panel = np.full(
        (H, src1.shape[1] + scan1.shape[1] + src2.shape[1] + 2 * pad, 3),
        20, dtype=np.uint8)
    x = 0
    for p in (src1, scan1, src2):
        panel[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


# ───────────────────────────────────────────────────────────────────────────
#  Suite runner
# ───────────────────────────────────────────────────────────────────────────

def run_suite(source_image_path: str | Path,
              output_dir: str | Path,
              n_cases: int = 3,
              n_fragments: int = 8,
              seed: int = 2026,
              roughen_amplitude: float = 6.0,
              max_rotation_deg: float = 25.0,
              skip_pipeline: bool = False,
              ) -> dict:
    """
    Generate and evaluate a benchmark suite.

    If `skip_pipeline` is True, only generates the cases (useful for
    producing test inputs without tying up the GPU).

    Returns a dict with per-case CaseMetrics + aggregate statistics.
    """
    # Import lazily so `generate_case` alone doesn't pull in SAM weights
    from ..pipeline import run as run_pipeline
    from .. import config as cfg

    source_image_path = Path(source_image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = cv2.imread(str(source_image_path))
    if source is None:
        raise FileNotFoundError(f"Source image not found: {source_image_path}")
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    rng = np.random.default_rng(seed)

    gen_cfg_base = TearGeneratorConfig(
        n_fragments=n_fragments,
        roughen_amplitude=roughen_amplitude,
        max_rotation_rad=np.deg2rad(max_rotation_deg),
    )

    all_metrics = []
    suite_start = time.time()
    for case_idx in range(n_cases):
        case_id = f"case_{case_idx:03d}"
        gen_cfg = TearGeneratorConfig(**{**asdict(gen_cfg_base),
                                          "seed": int(rng.integers(0, 2**31 - 1))})

        print(f"\n{'=' * 65}")
        print(f"  BENCHMARK CASE {case_idx + 1}/{n_cases}: {case_id}")
        print(f"{'=' * 65}")
        print(f"  Generating synthetic tear (N={n_fragments}) ...")
        truth = generate_case(source, output_dir, case_id, gen_cfg)
        print(f"  Wrote {truth['scan_path']}")

        if skip_pipeline:
            continue

        scan_path = Path(truth["scan_path"])
        print(f"\n  Running UNTORN pipeline on {scan_path.name} ...")
        out_path = output_dir / case_id / f"{case_id}_reconstructed.png"
        try:
            run_pipeline(str(scan_path), str(out_path))
        except Exception as exc:
            print(f"  PIPELINE FAILED: {exc}")
            all_metrics.append({
                "case_id": case_id,
                "status": "PIPELINE_ERROR",
                "error": str(exc),
            })
            continue

        # Pipeline wrote debug data to cfg.DEBUG_DIR / stem
        debug_dir = cfg.DEBUG_DIR / scan_path.stem

        pred_T = _load_predicted_transforms(debug_dir)
        pred_C = _load_predicted_centroids(debug_dir)

        if not pred_T or not pred_C:
            print(f"  Could not extract predictions from {debug_dir}")
            all_metrics.append({
                "case_id": case_id,
                "status": "NO_PREDICTIONS",
                "debug_dir": str(debug_dir),
            })
            continue

        scale = _load_pipeline_scale(debug_dir)
        metrics = evaluate_case(
            output_dir / case_id / "truth.json",
            pred_T, pred_C,
            predicted_fragment_polygons=None,
            working_scale_factor=scale)
        print(f"  -> placed={metrics.n_fragments_placed}/"
              f"{metrics.n_fragments_truth}, "
              f"median_pos_err={metrics.median_centroid_error_px:.1f}px, "
              f"median_rot_err={metrics.median_rotation_error_deg:.2f}deg, "
              f"align_rms={metrics.alignment_residual_px:.2f} "
              f"(working_scale={scale:.2f})")

        # Save a side-by-side diagnostic so regressions are visually obvious.
        try:
            _render_diagnostic(
                output_dir / case_id / "truth.json",
                pred_T, pred_C, scale,
                output_dir / case_id / f"{case_id}_diagnostic.png",
            )
        except Exception as exc:
            print(f"  (diagnostic render failed: {exc})")

        all_metrics.append({"status": "OK", **metrics.to_dict()})

    suite_time = time.time() - suite_start

    # ── Aggregate ───────────────────────────────────────────────────────
    ok = [m for m in all_metrics if m.get("status") == "OK"]
    agg = {
        "n_cases":                    len(all_metrics),
        "n_ok":                       len(ok),
        "suite_seconds":              round(suite_time, 1),
        "aggregate": {},
    }
    if ok:
        def agg_stat(key):
            vals = [m[key] for m in ok
                    if isinstance(m.get(key), (int, float))
                    and np.isfinite(m[key])]
            if not vals:
                return {}
            return {
                "mean":   float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p90":    float(np.percentile(vals, 90)),
            }
        agg["aggregate"] = {
            "placement_rate":              agg_stat("placement_rate"),
            "median_centroid_error_px":    agg_stat("median_centroid_error_px"),
            "p90_centroid_error_px":       agg_stat("p90_centroid_error_px"),
            "median_rotation_error_deg":   agg_stat("median_rotation_error_deg"),
            "p90_rotation_error_deg":      agg_stat("p90_rotation_error_deg"),
            "alignment_residual_px":       agg_stat("alignment_residual_px"),
        }

    report = {
        "source_image_path": str(source_image_path),
        "output_dir":        str(output_dir),
        "n_fragments":       n_fragments,
        "seed":              seed,
        "cases":             all_metrics,
        **agg,
    }
    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_aggregate(agg)
    print(f"\nReport: {report_path}")
    return report


def _print_aggregate(agg: dict) -> None:
    print("\n" + "=" * 65)
    print("  BENCHMARK AGGREGATE")
    print("=" * 65)
    if not agg.get("aggregate"):
        print("  (no successful cases)")
        return
    for key, stats in agg["aggregate"].items():
        if not stats:
            continue
        print(f"  {key:30s}  median={stats['median']:.3f}  "
              f"mean={stats['mean']:.3f}  p90={stats['p90']:.3f}")
    print(f"  cases OK: {agg['n_ok']}/{agg['n_cases']}")
    print(f"  suite time: {agg['suite_seconds']}s")
