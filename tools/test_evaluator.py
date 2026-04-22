"""
tools/test_evaluator.py
=======================
Sanity-check the benchmark evaluator by constructing an "oracle" prediction
that is the exact analytic inverse of the ground-truth scatter applied to
each fragment. A correct evaluator should report near-zero centroid error,
near-zero rotation error, and sub-pixel alignment residual.

Run:
    C:\\Users\\alish\\miniconda3\\envs\\untorn\\python.exe tools/test_evaluator.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from untorn.benchmark.synth_doc import synth_document
from untorn.benchmark.generate import generate_case, TearGeneratorConfig
from untorn.benchmark.evaluate import evaluate_case


def _oracle_prediction(truth: dict, working_scale: float = 1.0):
    """
    Build the exact analytic inverse of every GT scatter as the 'predicted'
    transforms. Feed the evaluator these — a correct evaluator should return
    near-zero errors.
    """
    pred_T = {}
    pred_C = {}
    for g in truth["fragments"]:
        fid = int(g["id"])
        M = np.asarray(g["affine_source_to_canvas"], dtype=np.float64)
        M3 = np.vstack([M, [0, 0, 1]])
        M_inv = np.linalg.inv(M3)  # canvas→source

        # Pipeline transforms map INPUT (working-res scan) coords → pipeline
        # canvas coords. The scan is the FULL-RES canvas, so to express
        # M_inv in working-res input coords we pre-scale by working_scale
        # (multiply input coord by working_scale → full-res coord → apply
        # M_inv → source coord).
        S = np.array([[working_scale, 0, 0],
                      [0, working_scale, 0],
                      [0, 0, 1]], dtype=np.float64)
        T_pred = M_inv @ S
        pred_T[fid] = T_pred

        # Predicted centroid in INPUT (working-res scan) frame: that's the
        # fragment's scatter-canvas centroid divided by working_scale.
        sc = np.asarray(g["centroid_in_source"], dtype=np.float64)
        cc = (M3 @ np.array([sc[0], sc[1], 1.0]))[:2]
        pred_C[fid] = (cc[0] / working_scale, cc[1] / working_scale)

    return pred_T, pred_C


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = synth_document(width=600, height=800, seed=42)
        cfg = TearGeneratorConfig(n_fragments=5, seed=123,
                                  resize_source_longest_edge=None)
        truth = generate_case(src, tmp_path, "case_oracle", cfg)

        for scale in (1.0, 1.3):
            pred_T, pred_C = _oracle_prediction(truth, working_scale=scale)
            metrics = evaluate_case(
                tmp_path / "case_oracle" / "truth.json",
                pred_T, pred_C, working_scale_factor=scale,
            )
            ok = (metrics.median_centroid_error_px < 0.5
                  and metrics.median_rotation_error_deg < 0.1
                  and metrics.alignment_residual_px < 0.5)
            tag = "PASS" if ok else "FAIL"
            print(f"[{tag}] scale={scale:.2f}  "
                  f"pos={metrics.median_centroid_error_px:.4f}px  "
                  f"rot={metrics.median_rotation_error_deg:.4f}deg  "
                  f"align_rms={metrics.alignment_residual_px:.4f}  "
                  f"placed={metrics.n_fragments_placed}/"
                  f"{metrics.n_fragments_truth}")
            if not ok:
                return 1
    print("\nEvaluator oracle test PASSED — math is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
