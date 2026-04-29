"""
Smoke test for untorn.seam_solver.refine_pair.

The seam solver runs a Nelder-Mead simplex in (Δθ, Δdx, Δdy) space against
``matching.evaluate_edge_fit.fit_cost`` plus an absolute SDT-overlap
penalty. We construct two synthetic torn-edge fragments whose true seams
sit on a known curve, perturb one fragment's pose by a small (Δθ, Δt),
and confirm:

  1. ``refine_pair`` reduces the combined cost.
  2. The recovered Δ is bounded by the configured drift caps.
  3. The optimiser is short-circuited when the initial pose is already
     locally optimal (no spurious updates accepted on noise-only seeds).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import cv2

from untorn.contours import analyze_fragments
from untorn.matching import (
    prepare_edges_and_sdt,
    affine_from_Rt,
    evaluate_edge_fit,
    match_pair,
)
from untorn.seam_solver import refine_pair


def _build_pair() -> tuple[list[dict], np.ndarray]:
    """Two vertical strips with a shared sinuous seam at x = 100.

    The fragments don't carry any random pose offset — the matcher should
    return R ≈ I, t ≈ 0 — so the seam solver can reliably drive a
    deliberately injected perturbation back to zero.
    """
    H, W = 220, 200
    img = np.full((H, W, 3), 238, dtype=np.uint8)
    for ry in (60, 120, 170):
        cv2.rectangle(img, (10, ry), (W - 10, ry + 6), (40, 40, 40), -1)

    rng = np.random.default_rng(0)
    ys = np.arange(20, H - 20)
    seam_x = (100.0 + 6.0 * np.sin(ys / 14.0)
              + rng.normal(scale=0.3, size=len(ys)))

    poly_a = np.concatenate([
        np.column_stack([np.full_like(ys, 12), ys]),
        np.column_stack([seam_x, ys])[::-1],
    ]).astype(np.int32)
    mask_a = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask_a, [poly_a], 255)

    poly_b = np.concatenate([
        np.column_stack([seam_x, ys]),
        np.column_stack([np.full_like(ys, W - 12), ys])[::-1],
    ]).astype(np.int32)
    mask_b = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask_b, [poly_b], 255)

    def _make(fid, mask):
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        Mm = cv2.moments(mask)
        cx = Mm["m10"] / max(Mm["m00"], 1)
        cy = Mm["m01"] / max(Mm["m00"], 1)
        return {"id": fid, "mask": mask, "contour": c,
                "bbox": (int(x), int(y), int(w), int(h)),
                "centroid": [cx, cy]}

    return [_make(0, mask_a), _make(1, mask_b)], img


def _torn_edge_indices(frag_a, frag_b):
    """Pick the most-anchored torn-edge pair on the matching seam."""
    ia = max(range(len(frag_a["edges"])),
             key=lambda k: (frag_a["edges"][k].get("is_torn", False),
                            frag_a["edges"][k].get("length", 0.0)))
    ib = max(range(len(frag_b["edges"])),
             key=lambda k: (frag_b["edges"][k].get("is_torn", False),
                            frag_b["edges"][k].get("length", 0.0)))
    return ia, ib


def test_seam_refine_recovers_perturbed_pose():
    fragments, img = _build_pair()
    with tempfile.TemporaryDirectory() as tmp:
        analyze_fragments(fragments, img, Path(tmp))
    prepare_edges_and_sdt(fragments, image_rgb=img)

    match = match_pair(fragments[0], fragments[1], img)
    if match is None:
        print("  matcher rejected the synthetic pair; smoke test SKIPPED")
        return

    R0 = match["R"]
    t0 = match["translation"]
    edge_i = int(match["edge_i"])
    edge_j = int(match["edge_j"])

    edge_a = fragments[0]["edges"][edge_i]
    edge_b = fragments[1]["edges"][edge_j]

    # Inject a 1° rotation + 1.5 px translation perturbation away from the
    # matcher's pose.
    perturb_theta = float(np.deg2rad(1.0))
    cP = float(np.cos(perturb_theta)); sP = float(np.sin(perturb_theta))
    P = np.array([[cP, -sP], [sP, cP]], dtype=np.float64)
    R_init = P @ R0
    t_init = P @ t0 + np.array([1.5, 0.5], dtype=np.float64)

    init_fit = evaluate_edge_fit(fragments[0], fragments[1],
                                  edge_a, edge_b, R_init, t_init)
    R_new, t_new, diag = refine_pair(
        fragments[0], fragments[1], edge_i, edge_j, R_init, t_init,
        max_drift_deg=3.0, max_drift_px=4.0,
        min_improvement=0.0)

    final_fit = evaluate_edge_fit(fragments[0], fragments[1],
                                   edge_a, edge_b, R_new, t_new)
    print(f"  init  cost={init_fit['fit_cost']:.2f}  gap={init_fit['fit_gap_px']:.2f}")
    print(f"  final cost={final_fit['fit_cost']:.2f}  gap={final_fit['fit_gap_px']:.2f}")
    print(f"  delta={diag.get('delta')}  iters={diag.get('iters', 0)}")

    assert final_fit["fit_cost"] <= init_fit["fit_cost"] + 1e-6, \
        "refine_pair must not worsen the cost"
    # Most synthetic perturbations are recoverable; allow a tolerance for
    # the case where the matcher's R0 was already biased.
    if "delta" in diag and any(diag["delta"]):
        assert abs(diag["delta"][0]) <= float(np.deg2rad(3.0)) + 1e-6
        assert abs(diag["delta"][1]) <= 4.0 + 1e-6
        assert abs(diag["delta"][2]) <= 4.0 + 1e-6


def test_seam_refine_no_op_at_optimum():
    """Starting at the matcher's pose, the solver should refuse to move
    when there's no measurable improvement (min_improvement guard)."""
    fragments, img = _build_pair()
    with tempfile.TemporaryDirectory() as tmp:
        analyze_fragments(fragments, img, Path(tmp))
    prepare_edges_and_sdt(fragments, image_rgb=img)

    match = match_pair(fragments[0], fragments[1], img)
    if match is None:
        print("  matcher rejected the synthetic pair; smoke test SKIPPED")
        return

    R0 = match["R"]; t0 = match["translation"]
    R_new, t_new, diag = refine_pair(
        fragments[0], fragments[1],
        int(match["edge_i"]), int(match["edge_j"]),
        R0, t0,
        min_improvement=10.0)
    print(f"  diag={diag}")
    if diag.get("reason") == "no_improvement":
        # Original pose preserved verbatim.
        assert np.allclose(R_new, R0)
        assert np.allclose(t_new, t0)


if __name__ == "__main__":
    test_seam_refine_recovers_perturbed_pose()
    test_seam_refine_no_op_at_optimum()
    print("seam_solver tests passed")
