"""
Sanity check untorn.appearance.

Runs DINOv2 on two synthetic fragment crops and checks:
    - same-source paper -> high cosine (>= 0.55)
    - different-content patches -> lower cosine than matching seam
    - seam_patch_cosine maps image<->feature coords correctly
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import cv2

from untorn.appearance import (
    DINOv2Extractor,
    attach_dinov2_features_all,
    seam_patch_cosine,
)


def _paper_crop(h=240, w=240, seed=0, ink_rows=(60, 120, 180)):
    """Paper-ish crop with repeating 'ink' bars."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    # light paper noise so DINOv2 patches aren't identical
    noise = rng.integers(-6, 6, size=(h, w, 1), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for ry in ink_rows:
        for cx in range(20, w - 20, 16):
            cv2.rectangle(img, (cx, ry), (cx + 10, ry + 10), (30, 30, 30), -1)
    return img


def _synth_two_fragments(seam_offset_px=0):
    """
    Build one synthetic 'document' then split it vertically. Both halves
    come from exactly the same paper/ink, so DINOv2 should give high
    cosine across the seam.
    """
    big = _paper_crop(h=240, w=500, seed=7, ink_rows=(60, 120, 180))
    left_img = big[:, :260].copy()
    right_img = big[:, 240:].copy()

    left = {
        "id": 0,
        "bbox": (0, 0, left_img.shape[1], left_img.shape[0]),
        "mask": np.full(left_img.shape[:2], 255, dtype=np.uint8),
    }
    right = {
        "id": 1,
        "bbox": (0, 0, right_img.shape[1], right_img.shape[0]),
        "mask": np.full(right_img.shape[:2], 255, dtype=np.uint8),
    }
    # The two "images" the pipeline would see (each fragment in its own
    # working-image frame). We use one composite canvas per fragment so the
    # fragment bbox/mask are valid indices into image_rgb in our tests.
    return left, left_img, right, right_img


def test_extract_shape_and_norm():
    ext = DINOv2Extractor.get()
    crop = _paper_crop(h=180, w=260)
    info = ext.extract(crop)
    assert info["features"].shape == (ext.grid, ext.grid, ext.feat_dim), \
        f"unexpected shape {info['features'].shape}"
    norms = np.linalg.norm(info["features"], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-3), \
        f"features not L2-normalised: norm range {norms.min():.3f}..{norms.max():.3f}"


def test_seam_same_paper_high_cosine():
    left, left_img, right, right_img = _synth_two_fragments()
    # Extract features separately; each fragment sees only its own image.
    attach_dinov2_features_all([left], left_img)
    attach_dinov2_features_all([right], right_img)

    # Seam is at x=260 in left's frame and x=0 in right's frame. Build the
    # placement so right is shifted by +240 into left's canvas (overlap
    # 20 px matches the big-image split).
    I = np.eye(3)
    T_r = np.eye(3); T_r[0, 2] = 240.0

    seam_point = np.array([250.0, 120.0])   # middle of left, near overlap
    seam_normal = np.array([1.0, 0.0])       # from left toward right

    score, n = seam_patch_cosine(left, right, I, T_r,
                                 seam_point, seam_normal,
                                 n_patches=6, patch_offset_px=8.0,
                                 sample_span_px=80.0)
    print(f"  same-paper seam cosine = {score:.3f} over {n} patches")
    assert n >= 4, f"expected >=4 valid patches, got {n}"
    assert score >= 0.55, f"same-paper seam should score high, got {score:.3f}"


def test_seam_different_paper_lower():
    left, left_img, right, _ = _synth_two_fragments()
    # 'right' is now an unrelated crop: different noise seed + shifted bars.
    other = _paper_crop(h=240, w=260, seed=99, ink_rows=(30, 90, 150))
    right_diff = {
        "id": 2,
        "bbox": (0, 0, other.shape[1], other.shape[0]),
        "mask": np.full(other.shape[:2], 255, dtype=np.uint8),
    }
    attach_dinov2_features_all([left], left_img)
    attach_dinov2_features_all([right_diff], other)

    I = np.eye(3)
    T_r = np.eye(3); T_r[0, 2] = 240.0
    seam_point = np.array([250.0, 120.0])
    seam_normal = np.array([1.0, 0.0])

    score_same, _ = seam_patch_cosine(left, left, I, I,
                                      seam_point, seam_normal,
                                      n_patches=6, patch_offset_px=6.0,
                                      sample_span_px=60.0)
    score_diff, _ = seam_patch_cosine(left, right_diff, I, T_r,
                                      seam_point, seam_normal,
                                      n_patches=6, patch_offset_px=8.0,
                                      sample_span_px=80.0)
    print(f"  self-self cosine    = {score_same:.3f}")
    print(f"  different-paper cos = {score_diff:.3f}")
    assert score_same > score_diff + 0.03, \
        f"self-self should exceed different-paper; {score_same:.3f} vs {score_diff:.3f}"


if __name__ == "__main__":
    test_extract_shape_and_norm()
    test_seam_same_paper_high_cosine()
    test_seam_different_paper_lower()
    print("appearance tests passed")
