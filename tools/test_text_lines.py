"""
Sanity check untorn.text_lines.

Renders a synthetic fragment-shaped crop with 3 horizontal "text" lines
(dark bars on a bright paper background), wraps it in UNTORN's fragment
dict shape, and verifies the detector returns ~3 baselines with roughly
the expected y positions. Also verifies a blank fragment returns zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import cv2

from untorn.text_lines import (
    detect_text_lines,
    text_line_continuity,
)


def _synth_fragment(h=220, w=260, tilt_deg=0.0, with_text=True):
    """A rectangular fragment with 3 horizontal 'lines' of dark bars."""
    img = np.full((h + 40, w + 40, 3), 240, dtype=np.uint8)  # paper-ish
    mask = np.zeros((h + 40, w + 40), dtype=np.uint8)
    # fragment rectangle centered in image
    x0, y0 = 20, 20
    cv2.rectangle(mask, (x0, y0), (x0 + w, y0 + h), 255, -1)
    if with_text:
        for row_y in (50, 100, 150):
            for col_x in range(40, w - 20, 14):
                cv2.rectangle(img,
                              (x0 + col_x, y0 + row_y),
                              (x0 + col_x + 9, y0 + row_y + 10),
                              (30, 30, 30), -1)
    # optional rotation for tilted-text test
    if abs(tilt_deg) > 1e-4:
        center = ((img.shape[1]) / 2, (img.shape[0]) / 2)
        M = cv2.getRotationMatrix2D(center, tilt_deg, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             borderValue=(240, 240, 240))
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                              borderValue=0)
    frag = {
        "id": 0,
        "mask": mask,
        "bbox": (0, 0, mask.shape[1], mask.shape[0]),
        "contour": np.array([[x0, y0], [x0 + w, y0],
                             [x0 + w, y0 + h], [x0, y0 + h]],
                            dtype=np.int32).reshape(-1, 1, 2),
        "centroid": [img.shape[1] / 2, img.shape[0] / 2],
    }
    return frag, img


def test_detects_three_baselines_horizontal():
    frag, img = _synth_fragment()
    lines = detect_text_lines(frag, img)
    assert len(lines) >= 3, f"expected >=3, got {len(lines)}"
    ys = sorted(float(l["center"][1]) for l in lines[:3])
    # the three synthetic baselines sit at y=20+55, 105, 155 in image coords
    # (row_y + bar half-height). Tolerance generous for projection peak offset.
    expected = [75.0, 125.0, 175.0]
    for got, want in zip(ys, expected):
        assert abs(got - want) < 8.0, f"baseline y off: got {got}, want ~{want}"
    for l in lines[:3]:
        assert abs(l["angle"]) < np.deg2rad(5), \
            f"baseline angle should be near 0, got {np.rad2deg(l['angle']):.1f} deg"


def test_detects_tilted_baselines():
    frag, img = _synth_fragment(tilt_deg=10.0)
    lines = detect_text_lines(frag, img)
    assert len(lines) >= 2, f"expected >=2 tilted baselines, got {len(lines)}"
    angles_deg = sorted(np.rad2deg(l["angle"]) for l in lines)
    # At least one should be near +10 deg (ignoring flip ambiguity).
    got = min(abs(a - 10.0) for a in angles_deg)
    got = min(got, min(abs(a + 170.0) for a in angles_deg))  # 180-flip
    assert got < 4.0, f"no baseline matched +10 deg; angles={angles_deg}"


def test_blank_fragment_returns_empty():
    frag, img = _synth_fragment(with_text=False)
    lines = detect_text_lines(frag, img)
    assert lines == [], f"blank fragment should return empty, got {len(lines)}"


def test_continuity_rewards_matching_pair():
    frag_a, img = _synth_fragment()
    lines_a = detect_text_lines(frag_a, img)
    assert len(lines_a) >= 3
    # Fragment B: shift baselines so they continue just past A across a seam.
    # A-lines span roughly x=60..269 in frag_a's frame. Shift B by 215 px so
    # B-lines run from x=275..484 — right next to A's end with a small gap.
    shift = np.array([215.0, 0.0])
    lines_b = []
    for l in lines_a:
        lines_b.append({
            "p0": l["p0"] + shift,
            "p1": l["p1"] + shift,
            "center": l["center"] + shift,
            "angle": l["angle"],
            "length": l["length"],
            "confidence": l["confidence"],
        })
    I = np.eye(3)
    M_a = I
    M_b = I  # already in their "post-placement" coordinates
    # Seam sits between A's right edge (x=269) and B's left edge (x=275).
    seam_point = np.array([272.0, 120.0])
    seam_normal = np.array([1.0, 0.0])
    score, expected = text_line_continuity(lines_a, lines_b, M_a, M_b,
                                           seam_point, seam_normal,
                                           search_radius_px=40.0)
    assert expected >= 2, f"expected >=2 near-seam lines, got {expected}"
    assert score >= 0.8, f"continuity score too low: {score:.2f}"


def test_continuity_penalizes_broken_pair():
    frag_a, img = _synth_fragment()
    lines_a = detect_text_lines(frag_a, img)
    # Fragment B: shift by ~half the inter-line spacing (50 px) so B's
    # baselines fall between A's — no accidental Y-alignment within 3 px tol.
    shift = np.array([10.0, 25.0])
    lines_b = []
    for l in lines_a:
        lines_b.append({
            "p0": l["p0"] + shift,
            "p1": l["p1"] + shift,
            "center": l["center"] + shift,
            "angle": l["angle"],
            "length": l["length"],
            "confidence": l["confidence"],
        })
    I = np.eye(3)
    seam_point = np.array([272.0, 120.0])
    seam_normal = np.array([1.0, 0.0])
    score, expected = text_line_continuity(lines_a, lines_b, I, I,
                                           seam_point, seam_normal,
                                           search_radius_px=40.0)
    assert score < 0.2, f"broken pair should score low, got {score:.2f}"


if __name__ == "__main__":
    test_detects_three_baselines_horizontal()
    test_detects_tilted_baselines()
    test_blank_fragment_returns_empty()
    test_continuity_rewards_matching_pair()
    test_continuity_penalizes_broken_pair()
    print("text_lines tests passed")
