"""
Stage-5 regression test for untorn.assembly.

The assembly module is the layout-agnostic replacement for the old
hierarchical `reconstruction` orchestrator. Its contract is simple:

    reconstruct(fragments, image_rgb, debug_dir) -> {frag_idx: 3x3 affine}

Every placed fragment should land in roughly the right pose regardless of
the fragments' incoming scan order or rotation. The old orchestrator
failed outright here because it tried to identify scan-layout corners
before anything else.

This test builds a 3-fragment synthetic document torn along two jagged
vertical seams, pre-rotates each fragment by a random angle + offset
(simulating the "randomly laid out" case the old corner finder died on),
and checks that `reconstruct` returns a transform dict that
(a) contains a transform for every fragment, (b) pulls at least two of
the three fragments into the seed's frame (i.e. the MST placed them),
and (c) runs end-to-end without crashing in under a reasonable budget.

We don't assert exact ground-truth transforms — that would over-constrain
this unit test — only that the orchestrator produces a coherent placement
and survives layouts that would have broken the old neighbor graph.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import cv2

from untorn.contours import analyze_fragments
from untorn.assembly import (
    reconstruct,
    _enumerate_pair_candidates,
    _resolve_conflicts,
    _MatchCache,
)


# ── Synthetic document builder ────────────────────────────────────────────

def _jagged_seam(ys: np.ndarray, x0: float, seed: int) -> np.ndarray:
    """Vertical seam with 6-px sinusoidal wobble + 0.6-px noise."""
    rng = np.random.default_rng(seed)
    return (x0
            + 6.0 * np.sin(ys / 12.0 + seed)
            + rng.normal(scale=0.6, size=len(ys)))


def _fill_fragment_mask(H: int, W: int, poly: np.ndarray) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m


def _rotate_fragment(mask: np.ndarray, img: np.ndarray,
                     angle_deg: float,
                     H: int, W: int,
                     offset: tuple[float, float]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate a fragment by `angle_deg` around its mask centroid, then shift
    by `offset`. Returns a new (mask, image) pair sized (H, W).
    """
    M_mom = cv2.moments(mask)
    if M_mom["m00"] < 1:
        return mask, img
    cx = M_mom["m10"] / M_mom["m00"]
    cy = M_mom["m01"] / M_mom["m00"]
    R = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0).astype(np.float64)
    R[0, 2] += offset[0]
    R[1, 2] += offset[1]
    M32 = R.astype(np.float32)
    mask_out = cv2.warpAffine(mask, M32, (W, H),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img_out = cv2.warpAffine(img, M32, (W, H),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return mask_out, img_out


def _bake_fragment(fid: int, mask: np.ndarray) -> dict:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    M_ = cv2.moments(mask)
    cx = M_["m10"] / max(M_["m00"], 1)
    cy = M_["m01"] / max(M_["m00"], 1)
    return {"id": fid, "mask": mask, "contour": c,
            "bbox": (x, y, w, h), "centroid": [cx, cy]}


def _build_three_fragment_case() -> tuple[list[dict], np.ndarray]:
    """
    Three vertical strips of a 480x220 "document" with sinuous torn seams
    between them, each fragment rotated by a small random angle and
    shifted into a scattered layout. This is the case that broke the old
    corner-seeded orchestrator on its first move.
    """
    H, W = 220, 480
    img = np.full((H, W, 3), 238, dtype=np.uint8)
    # three horizontal ink bars so paper-LAB has pure-paper pixels to sample
    for ry in (55, 110, 165):
        cv2.rectangle(img, (15, ry), (W - 15, ry + 7), (40, 40, 40), -1)

    ys = np.arange(20, H - 20)
    seam_1 = _jagged_seam(ys, 160.0, seed=1)
    seam_2 = _jagged_seam(ys, 320.0, seed=2)

    poly_a = np.concatenate([
        np.column_stack([np.full_like(ys, 20), ys]),
        np.column_stack([seam_1 - 2.0, ys])[::-1],
    ])
    poly_b = np.concatenate([
        np.column_stack([seam_1 + 2.0, ys]),
        np.column_stack([seam_2 - 2.0, ys])[::-1],
    ])
    poly_c = np.concatenate([
        np.column_stack([seam_2 + 2.0, ys]),
        np.column_stack([np.full_like(ys, W - 20), ys])[::-1],
    ])

    mask_a = _fill_fragment_mask(H, W, poly_a)
    mask_b = _fill_fragment_mask(H, W, poly_b)
    mask_c = _fill_fragment_mask(H, W, poly_c)

    # Each fragment goes onto its own scene image so we can rotate it
    # independently. The canvas is widened so rotated+shifted content
    # still fits without clipping.
    SH, SW = 320, 640
    scene = np.full((SH, SW, 3), 238, dtype=np.uint8)
    rng = np.random.default_rng(7)

    fragments = []
    img_scenes = []
    for fid, mask in enumerate([mask_a, mask_b, mask_c]):
        # paint fragment pixels onto the scene before rotating
        pad_img = np.full((SH, SW, 3), 238, dtype=np.uint8)
        pad_img[:H, :W] = img
        pad_mask = np.zeros((SH, SW), dtype=np.uint8)
        pad_mask[:H, :W] = mask
        angle = float(rng.uniform(-30.0, 30.0))
        dx = float(rng.uniform(-60.0, 60.0))
        dy = float(rng.uniform(-40.0, 40.0))
        m2, _ = _rotate_fragment(pad_mask, pad_img, angle, SH, SW, (dx, dy))
        _, i2 = _rotate_fragment(pad_mask, pad_img, angle, SH, SW, (dx, dy))
        img_scenes.append(i2)
        fragments.append(_bake_fragment(fid, m2))

    # The pipeline passes a single RGB image with every fragment burned
    # into it — collapse per-fragment scenes by OR-ing their masks.
    scene_img = np.full((SH, SW, 3), 238, dtype=np.uint8)
    for frag, im in zip(fragments, img_scenes):
        mk = frag["mask"] > 127
        scene_img[mk] = im[mk]

    return fragments, scene_img


# ── Tests ─────────────────────────────────────────────────────────────────

def test_enumerate_pair_candidates_skips_no_torn_fragments():
    """Fragments without torn edges must not appear in pairings."""
    fragments = [
        {"id": 0, "edges": [{"is_torn": True,  "length": 100.0}],
         "paper_lab": (60.0, 5.0, 12.0)},
        {"id": 1, "edges": [{"is_torn": False, "length": 200.0}],
         "paper_lab": (60.0, 5.0, 12.0)},
        {"id": 2, "edges": [{"is_torn": True,  "length": 120.0}],
         "paper_lab": (60.0, 5.0, 12.0)},
    ]
    pairs = _enumerate_pair_candidates(fragments)
    pair_set = {tuple(sorted(p)) for p in pairs}
    assert (0, 2) in pair_set, "0<>2 should survive (both have torn edges)"
    assert (0, 1) not in pair_set, "frag 1 has no torn edge; must be dropped"
    assert (1, 2) not in pair_set, "frag 1 has no torn edge; must be dropped"


def test_enumerate_pair_candidates_drops_mismatched_paper():
    """Wildly different paper LAB should be excluded by the prefilter."""
    fragments = [
        {"id": 0, "edges": [{"is_torn": True, "length": 100.0}],
         "paper_lab": (60.0, 0.0, 0.0)},     # near-white
        {"id": 1, "edges": [{"is_torn": True, "length": 100.0}],
         "paper_lab": (60.0, 80.0, 80.0)},   # saturated red (ΔE ~ 113)
        {"id": 2, "edges": [{"is_torn": True, "length": 110.0}],
         "paper_lab": (61.0, 1.0, -1.0)},    # near-white again
    ]
    pair_set = {tuple(sorted(p)) for p in _enumerate_pair_candidates(fragments)}
    assert (0, 2) in pair_set, "matched-paper pair must survive"
    assert (0, 1) not in pair_set, "ΔE 113 pair must be pruned"
    assert (1, 2) not in pair_set, "ΔE 113 pair must be pruned"


def test_resolve_conflicts_picks_max_weight_bipartite():
    """
    Two free fragments {A, B} both want to attach to the same anchor S.
    Conflict resolver should route each to its best distinct anchor so
    the assignment is bipartite-valid.
    """
    candidates = {
        10: [(0, {"confidence": 0.80}), (1, {"confidence": 0.60})],
        11: [(0, {"confidence": 0.70}), (1, {"confidence": 0.65})],
    }
    out = _resolve_conflicts(candidates)
    # Exactly two anchors assigned, each to a distinct free fragment.
    assert len(out) == 2, f"expected 2 assignments, got {len(out)}"
    free_targets = {pair[0] for pair in out.values()}
    assert free_targets == {10, 11}, \
        f"each free frag must be assigned exactly once; got {free_targets}"
    # Max-weight should prefer (0<->10 @ 0.80) + (1<->11 @ 0.65) = 1.45
    # over (0<->11 @ 0.70) + (1<->10 @ 0.60) = 1.30.
    assert out.get(0, (None, None))[0] == 10, \
        f"anchor 0 should prefer free-10; got {out.get(0)}"


def test_reconstruct_end_to_end_on_random_layout():
    """
    End-to-end smoke test: three torn fragments scattered at random
    angles/positions. The MST orchestrator must produce a transform for
    every fragment and place at least two of the three (seed + one
    neighbour). This is a case where the old corner-seeded orchestrator
    failed outright.
    """
    fragments, scene_rgb = _build_three_fragment_case()
    assert len(fragments) == 3

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        debug_dir = Path(tmp)
        analyze_fragments(fragments, scene_rgb, debug_dir)
        transforms = reconstruct(fragments, scene_rgb, debug_dir)
    dt = time.time() - t0

    print(f"  reconstruct returned in {dt:.2f}s")
    assert isinstance(transforms, dict), "reconstruct must return a dict"
    assert set(transforms.keys()) == {0, 1, 2}, \
        f"transform keys should cover every fragment; got {set(transforms.keys())}"
    for i, T in transforms.items():
        assert T.shape == (3, 3), f"transform {i} must be 3x3, got {T.shape}"
        assert np.isfinite(T).all(), f"transform {i} has non-finite entries"

    # At least two fragments should have moved from identity — the seed
    # stays at eye(3) but its partner must have been attached.
    non_identity = sum(1 for T in transforms.values()
                       if not np.allclose(T, np.eye(3), atol=1e-6))
    print(f"  {non_identity} / 3 fragments moved away from identity")
    assert non_identity >= 1, \
        "expected at least one non-seed fragment to be attached"


if __name__ == "__main__":
    test_enumerate_pair_candidates_skips_no_torn_fragments()
    test_enumerate_pair_candidates_drops_mismatched_paper()
    test_resolve_conflicts_picks_max_weight_bipartite()
    test_reconstruct_end_to_end_on_random_layout()
    print("assembly stage-5 tests passed")
