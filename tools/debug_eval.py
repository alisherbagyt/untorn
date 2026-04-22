"""
Debug the evaluator by printing every intermediate value for a given case.
Useful for diagnosing "why is rotation error huge when pipeline angles look fine".
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from untorn.benchmark.evaluate import (
    _apply_affine_pts, _umeyama_2d, _decompose_affine,
)
from untorn.benchmark.run_suite import (
    _load_predicted_transforms, _load_predicted_centroids,
    _load_pipeline_scale,
)


def main():
    case_dir = Path(r"C:\dev\untorn\data\benchmark\run_04\case_000")
    debug_dir = Path(r"C:\dev\untorn\data\debug\case_000_scan")

    truth = json.loads((case_dir / "truth.json").read_text(encoding="utf-8"))
    pred_T = _load_predicted_transforms(debug_dir)
    pred_C = _load_predicted_centroids(debug_dir)
    s = _load_pipeline_scale(debug_dir)

    print(f"working_scale = {s:.4f}")
    print(f"\nGT fragments:")
    gt_src_cs = []
    gt_can_cs = []
    for g in truth["fragments"]:
        M = np.array(g["affine_source_to_canvas"], dtype=np.float64)
        M3 = np.vstack([M, [0, 0, 1]])
        sc = np.array(g["centroid_in_source"], dtype=np.float64)
        cc = _apply_affine_pts(M3, sc.reshape(1, 2))[0]
        gt_src_cs.append(sc)
        gt_can_cs.append(cc)
        rot_deg = g["applied_rotation_deg"]
        print(f"  GT {g['id']}: scatter={rot_deg:+6.2f}deg  "
              f"src={sc}  canvas={cc.round(1).tolist()}")
    gt_src_cs = np.asarray(gt_src_cs)
    gt_can_cs = np.asarray(gt_can_cs)

    print(f"\nPred centroids (working) vs full-res scan frame:")
    for pid in sorted(pred_C):
        cw = np.array(pred_C[pid])
        cf = cw * s
        print(f"  pred {pid}: work={cw.round(1).tolist()}  "
              f"full_scan={cf.round(1).tolist()}")

    print(f"\nPred transforms (pipeline canvas):")
    for pid in sorted(pred_T):
        T = pred_T[pid]
        a, _ = _decompose_affine(T)
        print(f"  T[{pid}] angle={math.degrees(a):+7.2f}deg  "
              f"tx={T[0,2]:+8.1f} ty={T[1,2]:+8.1f}")

    # Match by full-res scan centroid
    pred_ids = sorted(pred_C)
    used = set()
    matches = {}
    print(f"\nMatching pred -> GT (nearest on full-res scan):")
    for pid in pred_ids:
        pc_full = np.array(pred_C[pid]) * s
        dists = np.linalg.norm(gt_can_cs - pc_full, axis=1)
        order = np.argsort(dists)
        for gi in order:
            if int(gi) not in used:
                matches[pid] = int(gi)
                used.add(int(gi))
                print(f"  pred {pid} -> GT {gi} (dist {dists[gi]:.1f})")
                break

    # Predicted output-canvas centroid
    pred_out = []
    gt_src_list = []
    for pid, gi in matches.items():
        T = pred_T[pid]
        pc = _apply_affine_pts(T, np.array(pred_C[pid]).reshape(1, 2))[0]
        pred_out.append(pc)
        gt_src_list.append(gt_src_cs[gi])
    pred_out = np.asarray(pred_out)
    gt_src_list = np.asarray(gt_src_list)

    print(f"\nPred canvas centroids vs GT source centroids:")
    for pid, po, gs in zip(matches.keys(), pred_out, gt_src_list):
        print(f"  pred {pid}: pipeline_canvas={po.round(1).tolist()}  "
              f"gt_source={gs.round(1).tolist()}")

    M_align = _umeyama_2d(pred_out, gt_src_list)
    ang_align, _ = _decompose_affine(M_align)
    scale_align = np.linalg.norm(M_align[:2, :2], ord=2)
    print(f"\nUmeyama: rotation={math.degrees(ang_align):+.2f}deg  "
          f"scale={scale_align:.4f}  "
          f"t=({M_align[0,2]:+.1f}, {M_align[1,2]:+.1f})")

    # Per-fragment rotation error
    print(f"\nPer-fragment rotation errors:")
    for pid, gi in matches.items():
        T = pred_T[pid]
        T_aligned = M_align @ T
        pred_angle, _ = _decompose_affine(T_aligned)
        gt_M = np.eye(3)
        gt_M[:2, :] = np.array(truth["fragments"][gi]["affine_source_to_canvas"])
        gt_angle, _ = _decompose_affine(gt_M)
        da = (pred_angle + gt_angle + math.pi) % (2 * math.pi) - math.pi
        rot_err = math.degrees(abs(da))
        print(f"  pred {pid} -> GT {gi}: "
              f"pred_ang={math.degrees(pred_angle):+7.2f}  "
              f"gt_ang={math.degrees(gt_angle):+7.2f}  "
              f"sum={math.degrees(pred_angle + gt_angle):+7.2f}  "
              f"err={rot_err:6.2f}deg")


if __name__ == "__main__":
    main()
