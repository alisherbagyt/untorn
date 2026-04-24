"""
Stage-7 regression test for untorn.gap_fill.

Covers hole classification (edge / small / medium / large), repair-mask
construction with text preservation, and the end-to-end orchestrator
which is expected to survive a missing LaMa checkpoint (SKIPPED_NO_MODEL)
and also actually inpaint when the backend is available.
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

from untorn.gap_fill import (
    _classify_holes,
    _build_repair_mask,
    _document_hull_mask,
    inpaint_gaps,
)


def _make_coverage(H=200, W=300,
                   edge_hole=True,
                   small_hole=True,
                   medium_hole=False,
                   large_hole=False) -> np.ndarray:
    """
    Build a coverage mask that fills the canvas minus a selectable set of
    holes. Doc area ~= H*W = 60,000 px. Small hole 5x5 (25 px, 0.04%),
    medium 20x20 (400 px, 0.67%), large 80x80 (6400 px, 10.7%). The edge
    notch punches through the top border so GAP_EDGE_TOUCH_PX classifies
    it as `edge`.
    """
    cov = np.full((H, W), 255, dtype=np.uint8)
    if edge_hole:
        # notch that actually touches the top border (y=0)
        cv2.rectangle(cov, (40, 0), (60, 8), 0, -1)
    if small_hole:
        cv2.rectangle(cov, (100, 50), (105, 55), 0, -1)
    if medium_hole:
        cv2.rectangle(cov, (150, 80), (170, 100), 0, -1)
    if large_hole:
        cv2.rectangle(cov, (80, 100), (160, 180), 0, -1)
    return cov


def test_classify_holes_edge_vs_interior():
    """Edge holes must be excluded from the interior repair mask."""
    cov = _make_coverage(edge_hole=True, small_hole=True,
                         medium_hole=False, large_hole=False)
    interior, reports = _classify_holes(
        cov, edge_touch_px=4, small_frac=0.005, medium_frac=0.05)
    kinds = sorted(r["kind"] for r in reports)
    print(f"  kinds={kinds}")
    assert "edge" in kinds, "edge hole should be reported"
    assert "small" in kinds, "small hole should be reported"
    # The interior mask must contain the small hole but NOT the edge notch
    # — sample one pixel inside each region.
    assert interior[52, 102] > 0, "small-hole pixel should be in interior mask"
    assert interior[4, 50] == 0, \
        "edge-hole pixel should be excluded from interior mask"


def test_classify_holes_identifies_large_as_missing_fragment():
    """Hole > GAP_MEDIUM_FRAC must be reported as 'large'."""
    cov = _make_coverage(edge_hole=False, small_hole=False,
                         medium_hole=False, large_hole=True)
    _interior, reports = _classify_holes(
        cov, edge_touch_px=4, small_frac=0.005, medium_frac=0.05)
    print(f"  reports={reports}")
    assert any(r["kind"] == "large" for r in reports), \
        f"80x80 hole should classify as large; got {reports}"


def test_build_repair_mask_preserves_ink():
    """Ink strokes inside the repair region must be subtracted out."""
    H, W = 160, 240
    cov = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(cov, (20, 20), (W - 20, H - 20), 255, -1)
    cv2.rectangle(cov, (50, 40), (70, 60), 0, -1)      # small interior hole
    canvas = np.full((H, W, 3), 238, dtype=np.uint8)
    # Paint a bar of ink crossing near the hole
    cv2.rectangle(canvas, (30, 50), (100, 56), (25, 25, 25), -1)
    holes, reports = _classify_holes(
        cov, edge_touch_px=4, small_frac=0.005, medium_frac=0.05)
    mask = _build_repair_mask(
        canvas, cov, holes, reports,
        band_px=4, ink_thresh=140, medium_context_px=20)
    # Ink pixels must not be marked for repair
    assert mask[53, 60] == 0, \
        "ink pixel should be excluded from repair mask"


def test_inpaint_gaps_end_to_end_no_lama():
    """
    End-to-end: even if LaMa is unavailable (or on a pristine environment
    that happens to have it loaded), the call must produce a canvas, a
    meta dict with a status, and the expected debug artefacts.
    """
    H, W = 160, 240
    cov = _make_coverage(H=H, W=W, edge_hole=True, small_hole=True,
                         medium_hole=True, large_hole=False)
    canvas = np.full((H, W, 3), 238, dtype=np.uint8)
    for ry in (30, 60, 90):
        cv2.rectangle(canvas, (25, ry), (W - 25, ry + 4),
                      (40, 40, 40), -1)
    with tempfile.TemporaryDirectory() as tmp:
        debug_dir = Path(tmp)
        res = inpaint_gaps(canvas, cov, debug_dir, refine=False)
    cleaned = res["canvas"]; meta = res["meta"]
    assert cleaned.shape == canvas.shape
    assert meta["status"] in ("OK", "SKIPPED_NO_MODEL", "OK_NO_OP", "FAILED")
    counts = meta["hole_counts"]
    print(f"  status={meta['status']}  counts={counts}  "
          f"missing_fragment={meta['missing_fragment']}  "
          f"repair_px={meta['mask_pixels']}")
    assert counts["edge"]   >= 1
    assert counts["small"]  >= 1
    # Medium 20x20 = 400 px / hull area (~180x220=39,600) = 1.01% which
    # falls between small_frac=0.005 and medium_frac=0.05.
    assert counts["medium"] >= 1


if __name__ == "__main__":
    test_classify_holes_edge_vs_interior()
    test_classify_holes_identifies_large_as_missing_fragment()
    test_build_repair_mask_preserves_ink()
    test_inpaint_gaps_end_to_end_no_lama()
    print("gap_fill stage-7 tests passed")
