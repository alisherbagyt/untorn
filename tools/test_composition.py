"""
Regression test for untorn.composition (polygon-clip / Step 6).

The polygon-clip composition uses a winner-take-all by interior SDT and a
Voronoi seam-fill. The legacy `_feathered_alpha` helper was retired with
the rewrite, so this test exercises the public surface that's actually
shipping:

  1. `_fragment_paper_lab` excludes ink pixels and returns a clean paper
     LAB mean; an all-ink fragment returns None.
  2. `_apply_lab_shift` brightens paper pixels by the requested ΔL while
     leaving ink (L < ink_thresh) essentially unchanged.
  3. `compose_final` runs end-to-end on a 2-fragment synthetic layout,
     returns the documented keys, covers most of the placed pixels, and
     emits the debug artefacts the frontend depends on.
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

from untorn.composition import (
    compose_final,
    _fragment_paper_lab,
    _apply_lab_shift,
)


def test_fragment_paper_lab_excludes_ink():
    """LAB mean should reflect paper, not the black ink bars on top."""
    H, W = 100, 100
    rgb = np.full((H, W, 3), (238, 232, 214), dtype=np.uint8)   # cream paper
    for ry in (20, 45, 70):
        cv2.rectangle(rgb, (10, ry), (90, ry + 6), (20, 20, 20), -1)
    mask = np.full((H, W), 255, dtype=np.uint8)
    lab = _fragment_paper_lab(rgb, mask, ink_thresh=140, safe_margin_px=3)
    assert lab is not None, "expected paper LAB to be computable"
    # Cream paper has L ~ 230+, roughly zero-centred a/b.
    assert lab[0] > 200.0, f"paper L should be bright, got {lab[0]:.1f}"


def test_fragment_paper_lab_returns_none_when_no_paper():
    """An all-ink fragment has no paper pixels — should return None."""
    H, W = 80, 80
    rgb = np.full((H, W, 3), 20, dtype=np.uint8)    # all "ink"
    mask = np.full((H, W), 255, dtype=np.uint8)
    lab = _fragment_paper_lab(rgb, mask, ink_thresh=140, safe_margin_px=3)
    assert lab is None, f"all-ink fragment should yield None, got {lab}"


def test_apply_lab_shift_spares_ink():
    """L+20 shift: paper brightens, ink stays dark."""
    H, W = 64, 64
    rgb = np.full((H, W, 3), 220, dtype=np.uint8)       # paper
    rgb[30:40, 10:54] = (25, 25, 25)                    # ink stripe
    mask = np.full((H, W), 255, dtype=np.uint8)
    out = _apply_lab_shift(rgb, mask, delta_lab=(20.0, 0.0, 0.0),
                           ink_thresh=140)
    paper_before = rgb[5, 5].astype(np.float32).mean()
    paper_after  = out[5, 5].astype(np.float32).mean()
    ink_before   = rgb[35, 30].astype(np.float32).mean()
    ink_after    = out[35, 30].astype(np.float32).mean()
    print(f"  paper {paper_before:.1f}->{paper_after:.1f}  "
          f"ink {ink_before:.1f}->{ink_after:.1f}")
    assert paper_after > paper_before + 3.0, \
        f"paper should brighten, got {paper_before:.1f}->{paper_after:.1f}"
    assert abs(ink_after - ink_before) < 3.0, \
        f"ink should barely move, got {ink_before:.1f}->{ink_after:.1f}"


def _make_fragment(mask: np.ndarray, fid: int) -> dict:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    M_ = cv2.moments(mask)
    cx = M_["m10"] / max(M_["m00"], 1)
    cy = M_["m01"] / max(M_["m00"], 1)
    return {"id": fid, "mask": mask, "contour": c,
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": [cx, cy],
            "area": int((mask > 127).sum())}


def test_compose_final_end_to_end():
    """Two fragments side-by-side, identity transforms, Voronoi seam-fill, no crash."""
    H, W = 120, 240
    img = np.full((H, W, 3), 238, dtype=np.uint8)
    for ry in (30, 60, 90):
        cv2.rectangle(img, (10, ry), (W - 10, ry + 4), (40, 40, 40), -1)
    mask_a = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask_a, (10, 10), (110, H - 10), 255, -1)
    mask_b = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask_b, (130, 10), (W - 10, H - 10), 255, -1)
    frags = [_make_fragment(mask_a, 0), _make_fragment(mask_b, 1)]
    transforms = {0: np.eye(3, dtype=np.float64),
                  1: np.eye(3, dtype=np.float64)}

    with tempfile.TemporaryDirectory() as tmp:
        debug_dir = Path(tmp)
        res = compose_final(img, frags, transforms, debug_dir)

        # Documented keys
        for k in ("canvas", "coverage", "gap_mask", "crop_bbox"):
            assert k in res, f"missing key {k}"
        canvas = res["canvas"]
        coverage = res["coverage"]
        assert canvas.ndim == 3 and canvas.shape[2] == 3
        assert coverage.shape[:2] == canvas.shape[:2]

        # Coverage pixels should exceed half of both source masks' area
        area_src = int((mask_a > 127).sum()) + int((mask_b > 127).sum())
        area_cov = int((coverage > 127).sum())
        print(f"  covered {area_cov} / {area_src} src px")
        assert area_cov > area_src * 0.5, \
            f"coverage should capture most placed pixels; "\
            f"got {area_cov}/{area_src}"

        # Debug artefacts persisted under the new (post-rewrite) names.
        comp_dir = debug_dir / "composition"
        for name in ("01_raw_composite.png",
                     "02_coverage_pre_voronoi.png",
                     "03_coverage_post_voronoi.png",
                     "composition_meta.json"):
            assert (comp_dir / name).exists(), f"missing debug artefact {name}"


if __name__ == "__main__":
    test_fragment_paper_lab_excludes_ink()
    test_fragment_paper_lab_returns_none_when_no_paper()
    test_apply_lab_shift_spares_ink()
    test_compose_final_end_to_end()
    print("composition tests passed")
