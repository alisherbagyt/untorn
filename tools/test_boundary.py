"""
Sanity check untorn.boundary.refine_boundary_subpixel.

Generates a synthetic image with a smooth disc whose true edge is at a
known sub-pixel radius; compares the sub-pixel-refined boundary to the
raw integer mask boundary. The refined points should be closer to the
true circle than the integer ones in RMS terms.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import cv2

from untorn.boundary import (
    refine_boundary_subpixel,
    _extract_ordered_contour,
)


def _render_disc(h: int, w: int, cx: float, cy: float, radius: float):
    """
    Anti-aliased disc at sub-pixel (cx, cy) with given radius.
    Returns (image_gray uint8, binary_mask uint8) where mask > 127 inside.
    """
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    # smooth transition 1px wide at the true edge
    alpha = np.clip(radius + 0.5 - dist, 0.0, 1.0)
    # Paper = 240, background = 30
    img = (30 + 210 * alpha).astype(np.uint8)
    mask = (dist <= radius).astype(np.uint8) * 255
    return img, mask


def _radial_rms_error(pts: np.ndarray, cx: float, cy: float, r: float) -> float:
    if len(pts) == 0:
        return float("inf")
    d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(np.sqrt(np.mean((d - r) ** 2)))


def test_subpixel_beats_integer():
    h, w = 200, 200
    cx, cy, r = 100.37, 99.62, 40.0   # deliberately non-integer center
    img, mask = _render_disc(h, w, cx, cy, r)

    raw = _extract_ordered_contour(mask)
    refined = refine_boundary_subpixel(mask, img, band_px=3.0,
                                        smooth_sigma=0.8, step_px=0.25)

    rms_raw = _radial_rms_error(raw, cx, cy, r)
    rms_ref = _radial_rms_error(refined, cx, cy, r)

    print(f"  raw     RMS radial error = {rms_raw:.3f} px")
    print(f"  refined RMS radial error = {rms_ref:.3f} px")

    assert rms_ref < rms_raw, "sub-pixel refinement did not improve on integer"
    assert rms_ref < 0.3, f"refined RMS too high: {rms_ref:.3f}"


def test_empty_mask_returns_empty():
    mask = np.zeros((50, 50), dtype=np.uint8)
    img = np.full((50, 50), 30, dtype=np.uint8)
    out = refine_boundary_subpixel(mask, img)
    assert len(out) == 0


def test_preserves_point_count():
    h, w = 120, 120
    _, mask = _render_disc(h, w, 60.0, 60.0, 30.0)
    img = np.full((h, w), 30, dtype=np.uint8)
    img[(mask > 0)] = 240
    raw = _extract_ordered_contour(mask)
    refined = refine_boundary_subpixel(mask, img)
    assert len(refined) == len(raw), "point count must be preserved"


def test_rotated_square_stays_sub_pixel():
    """The real use case: an irregular (square here) mask with a sharp edge."""
    h, w = 200, 200
    img = np.full((h, w), 30, dtype=np.uint8)
    pts = np.array([[60.3, 50.7], [150.2, 60.4],
                    [140.1, 150.9], [50.8, 140.3]], dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    cv2.fillPoly(img, [pts.astype(np.int32)], 240)

    raw = _extract_ordered_contour(mask)
    refined = refine_boundary_subpixel(mask, img, band_px=3.0, smooth_sigma=0.8)
    # Should differ from the integer contour in at least some points.
    assert len(refined) == len(raw)
    diff = np.linalg.norm(refined - raw, axis=1)
    assert diff.max() > 0.01, "refinement should move some points"
    # No single refinement should teleport; should stay within band.
    assert diff.max() < 4.0, f"refined point wandered {diff.max():.2f} px"


if __name__ == "__main__":
    test_subpixel_beats_integer()
    test_empty_mask_returns_empty()
    test_preserves_point_count()
    test_rotated_square_stays_sub_pixel()
    print("boundary tests passed")
