"""Curvature sign convention.

``compute_curvature_string`` (Wolfson turning function) must produce a
curvature signal that is rotation- and translation-invariant. This test
locks the convention against a synthetic L-shape so any future change
that flips the sign or breaks invariance is caught immediately.
"""

from __future__ import annotations

import math

import numpy as np

from untorn.contours import (compute_curvature_string,
                             resample_arc_length)


def _l_shape_polyline(n: int = 200) -> np.ndarray:
    """A right-angle L: (0, 0) -> (50, 0) -> (50, 50)."""
    half = n // 2
    seg1 = np.column_stack([np.linspace(0, 50, half), np.zeros(half)])
    seg2 = np.column_stack([np.full(half, 50.0), np.linspace(0, 50, half)])
    return np.vstack([seg1, seg2])


def test_curvature_string_for_straight_line_is_near_zero():
    pts = np.column_stack([np.linspace(0, 100, 200), np.zeros(200)])
    _, curv = compute_curvature_string(pts)
    assert curv.size > 0
    assert float(np.max(np.abs(curv))) < 1e-3


def test_curvature_string_for_l_shape_has_a_spike():
    pts = _l_shape_polyline()
    _, curv = compute_curvature_string(pts)
    # The L has a 90 deg turn in the middle; it should produce one strong
    # peak in |curvature|.
    assert curv.size > 0
    assert float(np.max(np.abs(curv))) > 0.1


def test_curvature_string_is_rotation_invariant():
    pts = _l_shape_polyline()
    _, curv0 = compute_curvature_string(pts)
    # Rotate by 47 deg.
    theta = math.radians(47.0)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    _, curv_rot = compute_curvature_string(pts @ R.T)
    assert curv0.size == curv_rot.size
    # Allow a tiny tolerance for floating-point + resampling jitter.
    assert float(np.max(np.abs(curv0 - curv_rot))) < 1e-2


def test_curvature_string_is_translation_invariant():
    pts = _l_shape_polyline()
    _, curv0 = compute_curvature_string(pts)
    _, curv1 = compute_curvature_string(pts + np.array([1234.0, -567.0]))
    assert float(np.max(np.abs(curv0 - curv1))) < 1e-9


def test_resample_arc_length_uniform_spacing():
    """Resampled points should be uniformly spaced along arc length."""
    pts = _l_shape_polyline()
    out = resample_arc_length(pts, n_samples=50)
    diffs = np.linalg.norm(np.diff(out, axis=0), axis=1)
    # Uniform spacing => all diffs equal
    assert diffs.std() / diffs.mean() < 0.05
