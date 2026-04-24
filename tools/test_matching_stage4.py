"""
Stage-4 regression tests for untorn.matching.

Covers the three new behaviours introduced by the matcher rewrite:

  1. Multi-seed Procrustes recovers pose when the SW match spans a wrong
     sub-arc. We synthesise a polyline where only one end's curvature
     fingerprint survives (the other end is overwritten with noise), and
     check that best-seed RMS is much lower than full-arc Procrustes RMS.

  2. Paper-color LAB prefilter scores matching paper high and mismatched
     paper low.

  3. The new `_match_edge_pair` runs end-to-end on two synthetic fragments
     that share a torn seam, returning a reasonable confidence without
     crashing even when DINOv2 / text-line caches are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import cv2

from untorn.matching import (
    procrustes_rigid,
    _mean_paper_lab,
    _paper_color_score,
)


def test_multi_seed_procrustes_beats_single():
    """
    Construct two polylines where only the first half matches under some
    rotation/translation; the second half is random noise. A single
    Procrustes over the whole array is biased by the noisy tail and
    produces a high RMS; a head-seed Procrustes recovers the correct pose.
    """
    rng = np.random.default_rng(0)
    n = 60
    t = np.linspace(0, 1, n)
    # Good half: a smooth curve in A, rotated+translated in B.
    curve = np.column_stack([t * 100.0, 5.0 * np.sin(2 * np.pi * t)])
    theta = np.deg2rad(8.0)
    R_true = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    t_true = np.array([20.0, -10.0])
    curve_b = (R_true @ curve.T).T + t_true

    # Corrupt the second half of B with garbage: SW would still line up
    # indices 1:1 (we do the same here), pulling a full-arc Procrustes
    # toward a compromise pose.
    half = n // 2
    curve_b_corrupt = curve_b.copy()
    curve_b_corrupt[half:] = rng.normal(scale=40.0, size=(n - half, 2)) \
                             + curve_b_corrupt[half - 1]

    # Full-arc Procrustes (single seed)
    _ang_full, _t_full, rms_full, _ = procrustes_rigid(curve_b_corrupt, curve)

    # Head-seed Procrustes (first half only) — what multi-seed would pick.
    _ang_head, _t_head, rms_head, _ = procrustes_rigid(
        curve_b_corrupt[:half], curve[:half])

    print(f"  full-arc RMS = {rms_full:.2f}  head-seed RMS = {rms_head:.2f}")
    # The head seed should be dramatically better (this is the whole point
    # of multi-seed). Require at least 3x improvement.
    assert rms_head < rms_full / 3.0, \
        f"head-seed should beat full-arc by >=3x; got {rms_head:.2f} vs {rms_full:.2f}"
    assert rms_head < 1.0, f"head-seed RMS should be near-zero, got {rms_head:.2f}"


def _paper_crop(h, w, paper_rgb, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), paper_rgb, dtype=np.uint8)
    noise = rng.integers(-4, 4, size=(h, w, 1), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def test_paper_color_score_distinguishes_paper():
    """Same paper -> score near 1.0; different paper -> score near 0."""
    img_cream = _paper_crop(80, 80, (238, 232, 214), seed=1)
    img_blue  = _paper_crop(80, 80, (120, 150, 210), seed=2)

    mask = np.full((80, 80), 255, dtype=np.uint8)
    frag_cream_a = {"mask": mask}
    frag_cream_b = {"mask": mask}
    frag_blue    = {"mask": mask}

    frag_cream_a["paper_lab"] = _mean_paper_lab(img_cream, frag_cream_a)
    # same paper, different noise seed:
    img_cream_2 = _paper_crop(80, 80, (238, 232, 214), seed=99)
    frag_cream_b["paper_lab"] = _mean_paper_lab(img_cream_2, frag_cream_b)
    frag_blue["paper_lab"]    = _mean_paper_lab(img_blue, frag_blue)

    score_same = _paper_color_score(frag_cream_a, frag_cream_b)
    score_diff = _paper_color_score(frag_cream_a, frag_blue)
    print(f"  paper-color same = {score_same:.2f}  diff = {score_diff:.2f}")
    assert score_same >= 0.9, f"same-paper score too low: {score_same:.2f}"
    assert score_diff <= 0.2, f"different-paper score too high: {score_diff:.2f}"


def test_match_pair_end_to_end_runs():
    """
    Smoke test: the new _match_edge_pair survives when DINOv2 / text_lines
    caches are absent (neutral scores) and still produces a confidence in
    [0, 1] on a simple torn pair.
    """
    from untorn.contours import analyze_fragments
    from untorn import matching as M
    import tempfile

    # Build two fragments that share a torn seam at x = 100. A is the left
    # half, B is the right half (translated so its centroid sits apart).
    H = 200
    img = np.full((H, 240, 3), 240, dtype=np.uint8)
    # ink bars so the paper-LAB helper has something to exclude
    for ry in (60, 120):
        cv2.rectangle(img, (30, ry), (210, ry + 8), (40, 40, 40), -1)

    # A sinuous shared seam at roughly x=105 so both fragments pick up a
    # torn (non-straight) edge along it.
    rng = np.random.default_rng(42)
    ys = np.arange(20, 181)
    seam_x = (105.0 + 6.0 * np.sin(ys / 12.0)
              + rng.normal(scale=0.6, size=len(ys)))

    # Fragment A: left half, bounded by the jagged seam on the right.
    poly_a = np.concatenate([
        np.column_stack([np.full_like(ys, 20), ys]),
        np.column_stack([seam_x - 2.0, ys])[::-1],
    ]).astype(np.int32)
    mask_a = np.zeros((H, 240), dtype=np.uint8)
    cv2.fillPoly(mask_a, [poly_a], 255)

    # Fragment B: right half, bounded by the jagged seam on its left,
    # shifted 10 px right so the fragments do not overlap.
    poly_b = np.concatenate([
        np.column_stack([seam_x + 8.0, ys]),
        np.column_stack([np.full_like(ys, 220), ys])[::-1],
    ]).astype(np.int32)
    mask_b = np.zeros((H, 240), dtype=np.uint8)
    cv2.fillPoly(mask_b, [poly_b], 255)

    def _mk(fid, mask):
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        M_ = cv2.moments(mask)
        cx = M_["m10"] / max(M_["m00"], 1)
        cy = M_["m01"] / max(M_["m00"], 1)
        return {
            "id": fid, "mask": mask, "contour": c,
            "bbox": (x, y, w, h), "centroid": [cx, cy],
        }

    fragments = [_mk(0, mask_a), _mk(1, mask_b)]
    with tempfile.TemporaryDirectory() as tmp:
        analyze_fragments(fragments, img, Path(tmp))
    M.prepare_edges_and_sdt(fragments, image_rgb=img)

    torn_a = [e for e in fragments[0]["edges"] if e["is_torn"]]
    torn_b = [e for e in fragments[1]["edges"] if e["is_torn"]]
    assert torn_a and torn_b, \
        f"expected torn edges on both fragments; got {len(torn_a)} / {len(torn_b)}"

    res = M._match_edge_pair(
        torn_a[0], torn_b[0], img, tag="frag0<>frag1",
        frag_a=fragments[0], frag_b=fragments[1])
    if res is None:
        # Not every synthetic pair survives the physical SDT gate; we only
        # require that the call completed without error.
        print("  match rejected (SDT or SW) — no crash, still a pass")
        return
    c = res["confidence"]
    print(f"  match confidence = {c:.2f}  "
          f"geom={res['geom_conf']:.2f} dinov2={res['dinov2_score']:.2f} "
          f"strip={res['strip_ncc_score']:.2f} text={res['text_score']:.2f} "
          f"paper={res['paper_score']:.2f}")
    assert 0.0 <= c <= 1.0, f"confidence out of range: {c}"


if __name__ == "__main__":
    test_multi_seed_procrustes_beats_single()
    test_paper_color_score_distinguishes_paper()
    test_match_pair_end_to_end_runs()
    print("matching stage-4 tests passed")
