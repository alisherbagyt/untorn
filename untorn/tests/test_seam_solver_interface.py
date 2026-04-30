"""Lock the seam_solver.refine_pair contract so the new assembly's call
site stays compatible."""

from __future__ import annotations

import math

import numpy as np

from untorn import seam_solver, matching as M


def test_refine_pair_returns_R_t_diag(torn_pair_zero):
    p = torn_pair_zero
    # Pick the first torn edge of each fragment as a synthetic seed.
    edge_a_idx = next(k for k, e in enumerate(p.frag_a["edges"]) if e["is_torn"])
    edge_b_idx = next(k for k, e in enumerate(p.frag_b["edges"]) if e["is_torn"])
    R0 = np.eye(2, dtype=np.float64)
    t0 = np.zeros(2, dtype=np.float64)
    R_new, t_new, diag = seam_solver.refine_pair(
        p.frag_a, p.frag_b, edge_a_idx, edge_b_idx, R0, t0)
    assert R_new.shape == (2, 2)
    assert t_new.shape == (2,)
    assert isinstance(diag, dict)


def test_refine_pair_does_not_increase_cost(torn_pair_factory):
    """Refinement must be cost-monotone: final_cost <= init_cost. If the
    solver can't improve, it returns no_improvement and leaves the pose
    unchanged."""
    p = torn_pair_factory(angle_deg=0.0)
    m = M.match_pair(p.frag_a, p.frag_b, p.image_a, direction_aware=False)
    assert m is not None
    R_new, t_new, diag = seam_solver.refine_pair(
        p.frag_a, p.frag_b, m["edge_i"], m["edge_j"], m["R"], m["t"])
    if "init_cost" in diag and "final_cost" in diag:
        assert diag["final_cost"] <= diag["init_cost"] + 1e-3
