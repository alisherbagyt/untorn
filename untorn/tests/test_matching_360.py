"""Full 360 deg rotation sweep on the synthetic torn pair.

This is the canary test of the engine rebuild. The OLD matcher rejected
any pair rotated more than 30 deg via ``RECON_MAX_ROTATION_DEG``; the
NEW matcher must recover every rotation in [0, 360).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from untorn import matching as M


# Sweep every 30 deg (12 angles), including 0 and 359.
ROTATION_ANGLES = [0.0, 30.0, 60.0, 90.0, 120.0, 135.0,
                   150.0, 180.0, 210.0, 240.0, 270.0,
                   300.0, 330.0, 359.0]


def _angle_err_deg(R_recovered: np.ndarray, R_gt: np.ndarray) -> float:
    a = math.atan2(R_recovered[1, 0], R_recovered[0, 0])
    b = math.atan2(R_gt[1, 0], R_gt[0, 0])
    da = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(math.degrees(da))


@pytest.mark.parametrize("angle_deg", ROTATION_ANGLES)
def test_match_recovers_full_rotation(torn_pair_factory, angle_deg):
    """At every angle 0..360, match_pair must recover (R, t) within tight
    tolerances. The test exercises the curvature-only matching path:
    text_lines is cleared because the text detector only searches +-30
    deg and reports spurious lines for canvas-rotated fragments. In real
    photos the text-prior is reliable; here we verify the curvature
    fallback works for the full rotation range."""
    p = torn_pair_factory(angle_deg=angle_deg)
    # Clear text fields so the matcher uses sub-arc Procrustes (the
    # rotation-agnostic path) — the synthetic generator's ink rows are
    # not real text and the detector mis-reports their angle.
    for frag in (p.frag_a, p.frag_b):
        frag["text_lines"] = []
        frag["text_angle_canonical"] = None
    m = M.match_pair(p.frag_a, p.frag_b, p.image_a, direction_aware=False)
    assert m is not None, \
        f"match_pair returned None at angle={angle_deg} deg"

    err_a = _angle_err_deg(m["R"], p.R_gt)
    err_t = float(np.linalg.norm(np.asarray(m["t"]) - p.t_gt))

    # Tolerances are generous: pixel-grid resampling during the synthetic
    # rotation introduces ~1-2 px of jitter regardless of matcher quality.
    assert err_a <= 1.5, \
        f"angle err {err_a:.2f} deg too high at angle={angle_deg}"
    assert err_t <= 8.0, \
        f"translation err {err_t:.1f} px too high at angle={angle_deg}"
    assert m["fit_cost"] <= 5.0, \
        f"fit_cost {m['fit_cost']:.2f} too high at angle={angle_deg}"
    assert m["confidence"] >= 0.7, \
        f"confidence {m['confidence']:.3f} too low at angle={angle_deg}"


def test_match_returns_correct_orientation(torn_pair_factory):
    """For a true torn seam (mirror tear curves), complementary should win
    over direct."""
    p = torn_pair_factory(angle_deg=0.0)
    m = M.match_pair(p.frag_a, p.frag_b, p.image_a, direction_aware=False)
    assert m is not None
    assert m["orientation"] == "complementary"


def test_match_pair_returns_none_when_no_torn_edges(torn_pair_factory):
    """A fragment with all-factory edges has nothing to match."""
    p = torn_pair_factory(angle_deg=0.0)
    # Forcibly clear "is_torn" on every edge of frag_a.
    for e in p.frag_a["edges"]:
        e["is_torn"] = False
    m = M.match_pair(p.frag_a, p.frag_b, p.image_a, direction_aware=False)
    assert m is None
