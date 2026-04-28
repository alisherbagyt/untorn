"""
untorn.seam_solver
==================
Step 4 — post-MST edge-contact pose refinement.

For every adjacent placed pair (anchor, attached), the matcher produces a
relative SE(2) (R, t) that aligns a sub-arc of the matched torn edges.
ICP tightens that further but optimises against the matched-correspondence
sub-arc, not against the full edge polylines and not under an explicit
non-overlap constraint. The seam solver is the final per-pair polish that
runs the search in the right space:

    cost(Δθ, Δdx, Δdy) =
        FIT_W_OVERLAP   * overlap_norm
      + FIT_W_GAP       * mean_nearest_neighbour_gap
      + FIT_W_UNCOVERED * (1 − arc_coverage)
      + λ_overlap       * total_SDT_penetration_px

The first three terms come straight from `matching.evaluate_edge_fit` —
already the project's standard "do these edges meet along their whole
length" measurement. The fourth is an explicit anti-overlap term so the
optimiser cannot trade gap for interpenetration when both moves cost the
same in the original `fit_cost`.

The optimiser is Nelder-Mead simplex (gradient-free) — the cost is mildly
non-smooth because of integer SDT lookups, which break L-BFGS-B but
Nelder-Mead handles cleanly.

Public API
----------
    refine_pair(frag_a, frag_b, edge_a_idx, edge_b_idx, R0, t0)
        -> (R*, t*, diag)

The caller is `assembly._refine_seams_post_mst`, which walks the merge log
and applies one refinement per attach.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from . import config as cfg
from .matching import evaluate_edge_fit


def _R_from_theta(theta: float) -> np.ndarray:
    c = float(np.cos(theta)); s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _compose_left(theta0: float, t0: np.ndarray,
                  d_theta: float, d_t: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Apply (Δθ, Δt) as a LEFT pre-composition on the original SE(2):

        new_pose(p) = ΔR · (R0·p + t0) + Δt
                    = (ΔR R0)·p + (ΔR t0 + Δt)

    Pre-composition lets us reason about Δθ and Δt as canvas-frame moves
    of the warped polyline, which is what we want — Δt and Δθ are
    interpretable directly in the seam frame.
    """
    dR = _R_from_theta(d_theta)
    R0 = _R_from_theta(theta0)
    new_R = dR @ R0
    new_t = dR @ np.asarray(t0, dtype=np.float64).reshape(2) + np.asarray(d_t, dtype=np.float64).reshape(2)
    return new_R, new_t


def refine_pair(frag_a: dict, frag_b: dict,
                edge_a_idx: int, edge_b_idx: int,
                R0: np.ndarray, t0: np.ndarray,
                *,
                lambda_overlap: float | None = None,
                max_iter: int | None = None,
                max_drift_deg: float | None = None,
                max_drift_px: float | None = None,
                min_improvement: float | None = None
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Refine (R, t) so frag_b's edge meets frag_a's edge tightly without
    interpenetration. Returns the new (R*, t*) plus a diagnostics dict.

    Args:
        frag_a, frag_b:    fragment dicts as built by ``fragment_io``.
        edge_a_idx, edge_b_idx: torn-edge indices (the matched pair).
        R0, t0:            initial rigid SE(2) mapping B's local coords
                           into A's local coords (as stored on a match dict).
        lambda_overlap:    extra penalty weight on absolute overlap pixels
                           (cfg.SEAM_SOLVER_LAMBDA_OVERLAP).
        max_iter:          simplex iteration cap (cfg.SEAM_SOLVER_MAX_ITER).
        max_drift_deg:     hard cap on rotation move (deg).
        max_drift_px:      hard cap on translation move per axis (px).
        min_improvement:   accept refinement only when cost drops by
                           at least this much (cfg.SEAM_SOLVER_MIN_IMPROVEMENT).

    Returns:
        (R*, t*, diag) where diag has init_cost, final_cost, delta and
        either the iteration count on success or a "reason" string when
        the refinement is rejected.
    """
    if lambda_overlap is None:
        lambda_overlap = float(getattr(cfg, "SEAM_SOLVER_LAMBDA_OVERLAP", 0.6))
    if max_iter is None:
        max_iter = int(getattr(cfg, "SEAM_SOLVER_MAX_ITER", 40))
    if max_drift_deg is None:
        max_drift_deg = float(getattr(cfg, "SEAM_SOLVER_MAX_DRIFT_DEG", 2.0))
    if max_drift_px is None:
        max_drift_px = float(getattr(cfg, "SEAM_SOLVER_MAX_DRIFT_PX", 5.0))
    if min_improvement is None:
        min_improvement = float(getattr(
            cfg, "SEAM_SOLVER_MIN_IMPROVEMENT", 0.5))

    edges_a = frag_a.get("edges") or []
    edges_b = frag_b.get("edges") or []
    if not (0 <= edge_a_idx < len(edges_a)) or not (0 <= edge_b_idx < len(edges_b)):
        return R0, t0, {"reason": "invalid_edge_idx"}
    edge_a = edges_a[edge_a_idx]
    edge_b = edges_b[edge_b_idx]
    if not edge_a.get("is_torn") or not edge_b.get("is_torn"):
        return R0, t0, {"reason": "non_torn_edge"}

    theta0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
    t0_arr = np.asarray(t0, dtype=np.float64).reshape(2)

    max_drift_rad = float(np.deg2rad(max_drift_deg))
    bounds = [(-max_drift_rad, max_drift_rad),
              (-max_drift_px, max_drift_px),
              (-max_drift_px, max_drift_px)]

    def _cost(x: np.ndarray) -> float:
        R_n, t_n = _compose_left(theta0, t0_arr,
                                 float(x[0]),
                                 np.array([float(x[1]), float(x[2])]))
        fit = evaluate_edge_fit(frag_a, frag_b, edge_a, edge_b, R_n, t_n)
        return float(fit["fit_cost"]) + lambda_overlap * float(fit.get("fit_overlap_px", 0.0))

    init_cost = _cost(np.zeros(3, dtype=np.float64))

    # Nelder-Mead — gradient-free, handles the integer-rounded SDT lookups
    # in evaluate_edge_fit. `bounds` are honoured by recent scipy releases.
    try:
        result = minimize(
            _cost,
            x0=np.zeros(3, dtype=np.float64),
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": int(max_iter),
                "xatol": 1e-3,
                "fatol": 1e-3,
                "adaptive": True,
            },
        )
    except Exception as exc:
        return R0, t0, {"reason": f"solver_error: {exc}",
                        "init_cost": init_cost, "final_cost": init_cost}

    final_cost = float(result.fun)
    if final_cost > init_cost - min_improvement:
        return R0, t0, {
            "reason": "no_improvement",
            "init_cost": round(init_cost, 3),
            "final_cost": round(final_cost, 3),
            "delta": [0.0, 0.0, 0.0],
        }

    # Defensive bounds clamp — Nelder-Mead can drift outside `bounds`
    # in some scipy versions if the initial simplex straddles the edge.
    d_theta = float(np.clip(result.x[0], -max_drift_rad, max_drift_rad))
    d_tx    = float(np.clip(result.x[1], -max_drift_px, max_drift_px))
    d_ty    = float(np.clip(result.x[2], -max_drift_px, max_drift_px))

    R_new, t_new = _compose_left(theta0, t0_arr, d_theta,
                                 np.array([d_tx, d_ty]))
    return R_new, t_new, {
        "init_cost":   round(init_cost, 3),
        "final_cost":  round(final_cost, 3),
        "delta":       [round(d_theta, 5),
                        round(d_tx, 3),
                        round(d_ty, 3)],
        "iters":       int(getattr(result, "nit", 0)),
    }


__all__ = ["refine_pair"]
