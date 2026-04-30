"""Pytest fixtures for the UNTORN engine tests.

Builds minimal synthetic torn-paper fragments that satisfy the descriptor
contract enforced by ``untorn.fragment_io.build_descriptor``. Each fixture
yields fragments already in ITS OWN local canvas (A in A's canvas, B in
B's canvas) so ``matching.match_pair`` is exercised exactly the way it is
called from production: input two independent fragment dicts, return the
SE(2) pose that aligns them.

Pose convention
---------------
``match_pair`` returns ``(R, t)`` such that for any point ``p`` in B's
local canvas, ``R @ p + t`` is the same physical point expressed in A's
local canvas. The synthetic generator below produces fragments and a
known ground-truth ``(R_gt, t_gt)`` consistent with that contract.

Why the math works the way it does
----------------------------------
Generating B at angle θ rotates B's canvas pixels by θ around B's
centroid ``c_B``. The matcher must therefore return the *inverse*
rotation ``R(-θ)`` (so that points in the rotated canvas, when warped,
land back where the un-rotated B placed them). The translation accounts
for the fact that rotating around ``c_B`` shifts the canvas if ``c_B``
is not the canvas origin: solve

    p_in_A = R(-θ) @ p_in_B_rotated + t_gt
           = p_in_B_unrotated + (bx_b - bx_a, by_b - by_a)

and ``t_gt = c_B - R(-θ) @ c_B + (bx_b - bx_a, by_b - by_a)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

# Synthetic masks have flat-color page interiors with no gradient at the
# axis-aligned edges. Sub-pixel ridge-snapping has nothing to lock onto and
# can wiggle straight contours into ~1.7x-longer paths, which corrupts the
# torn/factory edge classifier. Disable for the test fixtures — production
# (real photos) keeps it on via the default in untorn/config.py.
from untorn import config as _cfg
_cfg.BOUNDARY_REFINE_ENABLED = False

from untorn import fragment_io


# ---------------------------------------------------------------------------
# Synthetic page generation
# ---------------------------------------------------------------------------

def _draw_synthetic_page(h: int = 600, w: int = 800,
                         seed: int = 12345) -> np.ndarray:
    """White page with text-like ink rows so appearance gates have signal."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    for row in range(60, h - 60, 50):
        start = int(rng.integers(40, 120))
        end = int(rng.integers(w - 120, w - 40))
        cv2.line(img, (start, row), (end, row), (40, 35, 30), 2)
        for gap_x in range(start + 60, end - 60, 80):
            gw = int(rng.integers(8, 14))
            cv2.rectangle(img, (gap_x, row - 5), (gap_x + gw, row + 5),
                          (245, 245, 245), thickness=-1)
    noise = rng.normal(0, 3, size=img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _tear_curve(h: int, w: int, seed: int = 7) -> np.ndarray:
    """Smooth-but-non-trivial vertical tear from y=0 to y=h.

    We deliberately keep the curve ASYMMETRIC and DISTINCTIVE (one big
    lobe and a small bump) so the matcher can lock onto a unique
    correspondence — multi-lobed tears create many ambiguous local
    matches and turn the test into a "find the global optimum among
    near-duplicates" problem rather than a basic correctness check.
    """
    n = max(40, h // 6)
    ys = np.linspace(0, h - 1, n)
    # One big sine lobe (period = h), one small bump (period = h/3).
    big = 28.0 * np.sin(np.linspace(0, math.pi, n))
    small = 6.0 * np.sin(np.linspace(0, 3 * math.pi, n) + 0.7)
    xs = w / 2.0 + big + small
    return np.column_stack([xs, ys]).astype(np.int32)


def _split_along_tear(page: np.ndarray, tear_xy: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    h, w = page.shape[:2]
    pts = tear_xy.astype(np.float64)
    pts = pts[np.argsort(pts[:, 1])]
    x_at_y = np.full(h, w / 2.0, dtype=np.float64)
    for k in range(len(pts) - 1):
        y0, y1 = int(pts[k, 1]), int(pts[k + 1, 1])
        x0, x1 = float(pts[k, 0]), float(pts[k + 1, 0])
        if y1 == y0:
            x_at_y[y0:y0 + 1] = (x0 + x1) / 2.0
            continue
        rows = np.arange(max(0, y0), min(h, y1 + 1))
        ts = (rows - y0) / (y1 - y0)
        x_at_y[rows] = x0 + ts * (x1 - x0)
    cols = np.arange(w)
    left = np.zeros((h, w), dtype=np.uint8)
    right = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        left[y, :] = (cols < x_at_y[y]).astype(np.uint8) * 255
        right[y, :] = (cols >= x_at_y[y]).astype(np.uint8) * 255
    return left, right


# ---------------------------------------------------------------------------
# Fragment construction
# ---------------------------------------------------------------------------

def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 127)
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1))


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.int32)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)


def _frag_from_mask(image: np.ndarray, mask: np.ndarray, frag_id: int) -> dict:
    bx, by, bw, bh = _bbox(mask)
    contour = _largest_contour(mask)
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        cx, cy = bx + bw / 2.0, by + bh / 2.0
    return {
        "id": int(frag_id),
        "mask": mask,
        "bbox": (bx, by, bw, bh),
        "area": int((mask > 127).sum()),
        "contour": contour,
        "centroid": (float(cx), float(cy)),
    }


def _isolate(mask: np.ndarray, image: np.ndarray, frag_id: int,
             pad: int = 40) -> tuple[dict, np.ndarray, int, int]:
    """Crop a fragment into its own (PAD-padded) canvas.

    Returns ``(frag_dict, canvas_image, src_origin_x, src_origin_y)`` where
    ``src_origin`` is the (x, y) of the source-page pixel that ended up at
    canvas (PAD, PAD) — used to compute ground-truth translations.
    """
    bx, by, bw, bh = _bbox(mask)
    out_h = bh + 2 * pad
    out_w = bw + 2 * pad
    canvas = np.full((out_h, out_w, 3), 245, dtype=np.uint8)
    cmask = np.zeros((out_h, out_w), dtype=np.uint8)
    m_crop = mask[by:by + bh, bx:bx + bw] > 127
    i_crop = image[by:by + bh, bx:bx + bw]
    cmask[pad:pad + bh, pad:pad + bw][m_crop] = 255
    canvas[pad:pad + bh, pad:pad + bw][m_crop] = i_crop[m_crop]
    return _frag_from_mask(canvas, cmask, frag_id), canvas, bx - pad, by - pad


# ---------------------------------------------------------------------------
# Public dataclass + factory
# ---------------------------------------------------------------------------

@dataclass
class TornPair:
    """Two ingested fragment dicts with a known ground-truth pose.

    The ground-truth pose maps points in B's local frame to A's local frame:
    ``p_in_A = R_gt @ p_in_B + t_gt``.
    """
    frag_a: dict
    frag_b: dict
    image_a: np.ndarray
    image_b: np.ndarray
    R_gt: np.ndarray
    t_gt: np.ndarray
    angle_gt_deg: float


def _rotate_canvas(image: np.ndarray, mask: np.ndarray,
                   angle_deg: float, center: tuple[float, float],
                   ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Rotate image+mask by ``angle_deg`` around ``center`` in canvas frame.

    Returns ``(rot_image, rot_mask, new_center)`` where new_center is where
    the original ``center`` lands in the new canvas (used for ground-truth
    pose math).
    """
    h, w = mask.shape
    cx, cy = float(center[0]), float(center[1])
    Rcv = cv2.getRotationMatrix2D((cx, cy), -float(angle_deg), 1.0)

    # Compute the rotated bbox of the canvas corners
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    rotated = (Rcv[:, :2] @ corners.T).T + Rcv[:, 2]
    pad = 8
    rx0 = float(rotated[:, 0].min()) - pad
    ry0 = float(rotated[:, 1].min()) - pad
    rx1 = float(rotated[:, 0].max()) + pad
    ry1 = float(rotated[:, 1].max()) + pad
    rw = int(math.ceil(rx1 - rx0))
    rh = int(math.ceil(ry1 - ry0))
    Rcv[0, 2] -= rx0
    Rcv[1, 2] -= ry0

    rot_mask = cv2.warpAffine(mask, Rcv, (rw, rh),
                              flags=cv2.INTER_NEAREST, borderValue=0)
    rot_img = cv2.warpAffine(image, Rcv, (rw, rh),
                             flags=cv2.INTER_LINEAR,
                             borderValue=(245, 245, 245))
    new_center = (cx - rx0, cy - ry0)
    return rot_img, rot_mask, new_center


def make_torn_pair(angle_deg: float = 0.0, *,
                   page_h: int = 600, page_w: int = 800,
                   seed: int = 12345) -> TornPair:
    """Build a synthetic torn pair where B has been rotated by ``angle_deg``.

    Math (derived from the matcher's contract):

        p_in_A = R(-theta) @ p_in_B_rotated + t_gt
        t_gt   = c_B_post - R(-theta) @ c_B_post + (bx_b - bx_a,
                                                     by_b - by_a)

    where c_B_post is B's centroid in its CURRENT (post-rotation) canvas
    and (bx_b - bx_a, by_b - by_a) is the inter-canvas offset that places
    B's tear back onto A's tear when both are un-rotated.
    """
    page = _draw_synthetic_page(page_h, page_w, seed=seed)
    tear = _tear_curve(page_h, page_w, seed=seed + 1)
    left, right = _split_along_tear(page, tear)

    frag_a, image_a, ax_off, ay_off = _isolate(left, page, frag_id=0)
    frag_b, image_b, bx_off, by_off = _isolate(right, page, frag_id=1)

    base_t = np.array([bx_off - ax_off, by_off - ay_off], dtype=np.float64)

    if angle_deg == 0.0:
        R_gt = np.eye(2, dtype=np.float64)
        t_gt = base_t.copy()
    else:
        c_pre = np.array(frag_b["centroid"], dtype=np.float64)
        rot_img, rot_mask, c_post = _rotate_canvas(
            image_b, frag_b["mask"], angle_deg, c_pre)
        frag_b = _frag_from_mask(rot_img, rot_mask, frag_id=1)
        image_b = rot_img
        c_post_arr = np.array(c_post, dtype=np.float64)
        theta = math.radians(angle_deg)
        c, s = math.cos(-theta), math.sin(-theta)
        R_gt = np.array([[c, -s], [s, c]], dtype=np.float64)
        # The matcher returns (R, t) such that R @ p_post + t = p_pre + base_t.
        # Plug in c_post -> c_pre: t = c_pre + base_t - R_gt @ c_post.
        t_gt = c_pre + base_t - R_gt @ c_post_arr

    fragment_io.build_all([frag_a], image_a)
    fragment_io.build_all([frag_b], image_b)

    return TornPair(
        frag_a=frag_a, frag_b=frag_b,
        image_a=image_a, image_b=image_b,
        R_gt=R_gt, t_gt=t_gt,
        angle_gt_deg=float(angle_deg),
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def torn_pair_factory():
    """Factory: ``torn_pair_factory(angle_deg=...)`` returns a TornPair."""
    return make_torn_pair


@pytest.fixture
def torn_pair_zero():
    """Trivial torn pair with no rotation."""
    return make_torn_pair(angle_deg=0.0)


# ---------------------------------------------------------------------------
# Multi-fragment scene for assembly tests (shared canvas)
# ---------------------------------------------------------------------------

@dataclass
class TornScene:
    """N fragments built by tearing a synthetic page along multiple curves.

    All fragments live in the SAME ``image_rgb`` canvas and share its
    coordinate frame — which is what production looks like after Phase 1
    segmentation. Ground-truth ``transforms_gt`` is the identity for every
    fragment (each fragment's contour is already at its true global
    position). To make ``reconstruct`` work it must IGNORE the canvas
    layout and place fragments by curvature matching alone.
    """
    fragments: list[dict]
    image_rgb: np.ndarray
    transforms_gt: dict[int, np.ndarray]


def _multi_tear_curves(h: int, w: int, n_pieces: int = 3,
                        seed: int = 31) -> list[np.ndarray]:
    """Generate ``n_pieces - 1`` smooth vertical-ish tear curves dividing
    the page into ``n_pieces`` strips."""
    rng = np.random.default_rng(seed)
    curves = []
    for k in range(1, n_pieces):
        n = max(40, h // 6)
        ys = np.linspace(0, h - 1, n)
        center_x = w * k / n_pieces
        big = (rng.uniform(20, 32)
               * np.sin(np.linspace(0, math.pi, n) + rng.uniform(0, 1.0)))
        small = (rng.uniform(4, 8)
                 * np.sin(np.linspace(0, 3 * math.pi, n) + rng.uniform(0, 1.0)))
        xs = center_x + big + small
        curves.append(np.column_stack([xs, ys]).astype(np.int32))
    return curves


def _split_into_strips(page: np.ndarray, tears: list[np.ndarray]) -> list[np.ndarray]:
    """Cut the page into n_pieces vertical strips by the given tears."""
    h, w = page.shape[:2]
    # Per-row x thresholds for each tear.
    thresholds = []
    for tear in tears:
        pts = tear[np.argsort(tear[:, 1])]
        x_at_y = np.full(h, w / 2.0, dtype=np.float64)
        for k in range(len(pts) - 1):
            y0, y1 = int(pts[k, 1]), int(pts[k + 1, 1])
            x0, x1 = float(pts[k, 0]), float(pts[k + 1, 0])
            if y1 == y0:
                x_at_y[y0:y0 + 1] = (x0 + x1) / 2.0
                continue
            rows = np.arange(max(0, y0), min(h, y1 + 1))
            ts = (rows - y0) / (y1 - y0)
            x_at_y[rows] = x0 + ts * (x1 - x0)
        thresholds.append(x_at_y)

    cols = np.arange(w)
    n_strips = len(tears) + 1
    masks = []
    for s in range(n_strips):
        mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            lo_x = float(thresholds[s - 1][y]) if s > 0 else -1.0
            hi_x = float(thresholds[s][y]) if s < len(tears) else w + 1.0
            mask[y, :] = ((cols >= lo_x) & (cols < hi_x)).astype(np.uint8) * 255
        masks.append(mask)
    return masks


def make_torn_scene(n_pieces: int = 3, *,
                    page_h: int = 600, page_w: int = 800,
                    seed: int = 42) -> TornScene:
    """Build a scene of ``n_pieces`` fragments laid out in their original
    page positions (no perturbation). Used by assembly tests to verify
    reconstruct() can re-discover the correct cluster from curvature
    matching alone."""
    page = _draw_synthetic_page(page_h, page_w, seed=seed)
    tears = _multi_tear_curves(page_h, page_w,
                                n_pieces=n_pieces, seed=seed + 1)
    masks = _split_into_strips(page, tears)
    fragments = [_frag_from_mask(page, m, frag_id=i)
                 for i, m in enumerate(masks)]
    fragment_io.build_all(fragments, page)
    return TornScene(
        fragments=fragments,
        image_rgb=page,
        transforms_gt={i: np.eye(3, dtype=np.float64) for i in range(n_pieces)},
    )


@pytest.fixture
def torn_scene_factory():
    """Factory: ``torn_scene_factory(n_pieces=N)`` returns a TornScene."""
    return make_torn_scene


# Selfcheck for the multi-fragment fixture.
def _selfcheck_scene() -> None:
    s = make_torn_scene(n_pieces=3)
    assert len(s.fragments) == 3
    for f in s.fragments:
        assert f.get("edges"), f"fragment {f['id']} has no edges"
        torn = sum(1 for e in f["edges"] if e["is_torn"])
        assert torn >= 1, f"fragment {f['id']} has no torn edges"


_selfcheck_scene()


# ---------------------------------------------------------------------------
# Self-check at import time
# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    p = make_torn_pair(angle_deg=0.0)
    assert p.frag_a.get("edges"), "frag_a has no edges"
    assert p.frag_b.get("edges"), "frag_b has no edges"
    a_torn = sum(1 for e in p.frag_a["edges"] if e["is_torn"])
    b_torn = sum(1 for e in p.frag_b["edges"] if e["is_torn"])
    assert a_torn >= 1 and b_torn >= 1, \
        "synthetic generator produced no torn edges"


_selfcheck()
