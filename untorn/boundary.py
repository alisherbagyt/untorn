"""
untorn.boundary
===============
Sub-pixel boundary refinement.

SAM 2.1 returns integer-pixel masks. Their raw contours carry +-1 px
jitter that dominates the Wolfson curvature signal for gentle tears,
and downstream composition shows this jitter as visible white seams
even when fragments are algorithmically aligned correctly.

This module snaps each boundary pixel to the local gradient-magnitude
maximum along its inward normal, within a small search band, refining
the contour to genuinely sub-pixel (x, y) coordinates. The refined
contour is then used by contours.py for curvature sampling and by
composition.py for seam-aware rendering.

Algorithm (per fragment):
    1. Extract the largest external contour as an ordered pixel polyline.
    2. Compute gradient magnitude of the working-image grayscale, optionally
       smoothed with a small Gaussian so micro-texture noise does not
       hijack the snap.
    3. For each contour point, compute an inward-pointing unit normal from
       the local tangent (the contour is CCW by OpenCV convention, so the
       -90 deg tangent rotation points into the mask).
    4. Sample the gradient magnitude along the normal at offsets
       [-band_px, +band_px] with a small sub-pixel step via bilinear
       interpolation.
    5. Find the offset that maximises gradient magnitude; refine the peak
       with a 3-point parabolic fit for true sub-pixel accuracy.
    6. Move the point along the normal by that offset. Output Nx2 float64.

Complexity: O(N * M) bilinear samples per fragment, where N is the number
of boundary points (~perimeter in px) and M is the number of offset
samples (~20 at default settings). On CPU this is roughly 10-30 ms per
fragment at 1500 px working resolution.
"""

from __future__ import annotations

import numpy as np
import cv2

from . import config as cfg


# ---------------------------------------------------------------------------
# Gradient magnitude
# ---------------------------------------------------------------------------

def _gradient_magnitude(image_gray: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smoothed Sobel gradient magnitude, float32."""
    gray = image_gray
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    if sigma and sigma > 0:
        # kernel size: roughly 6*sigma, forced odd
        ksize = max(3, int(round(sigma * 6)) | 1)
        gray = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


# ---------------------------------------------------------------------------
# Bilinear sampling of a float32 image at arbitrary float coordinates
# ---------------------------------------------------------------------------

def _bilinear_sample(img: np.ndarray,
                     xs: np.ndarray,
                     ys: np.ndarray) -> np.ndarray:
    """
    Vectorised bilinear sample of `img` at (xs, ys). Shapes of xs/ys match,
    and the returned array has the same shape. Out-of-bounds coordinates
    clamp to the nearest edge.
    """
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    # fractional parts BEFORE clamping so weights match the true coords
    wx1 = xs - x0
    wy1 = ys - y0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    return (img[y0, x0] * wx0 * wy0 +
            img[y0, x1] * wx1 * wy0 +
            img[y1, x0] * wx0 * wy1 +
            img[y1, x1] * wx1 * wy1)


# ---------------------------------------------------------------------------
# Ordered contour extraction
# ---------------------------------------------------------------------------

def _extract_ordered_contour(mask: np.ndarray) -> np.ndarray:
    """
    Largest external contour as an ordered Nx2 (x, y) float64 polyline.
    Uses CHAIN_APPROX_NONE so every boundary pixel is preserved; the
    curvature string will downsample later.
    """
    m = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float64)


# ---------------------------------------------------------------------------
# Inward-normal estimation from ordered polyline
# ---------------------------------------------------------------------------

def _inward_normals(pts: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
    """
    For each point on a closed contour, unit normal pointing into the mask.
    We compute a tangent from the neighbours, rotate it -90 deg to get a
    candidate normal (correct for CCW contours, which is OpenCV's default
    for external contours), then verify by probing the mask a few pixels
    in that direction and flipping per-point if needed.
    """
    n = len(pts)
    if n < 3:
        return np.zeros_like(pts)

    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    tang = nxt - prev
    tlen = np.linalg.norm(tang, axis=1, keepdims=True)
    tlen = np.where(tlen < 1e-8, 1.0, tlen)
    tang = tang / tlen

    # rotate -90 deg: (tx, ty) -> (ty, -tx). For a CCW external contour
    # this points INWARD.
    normals = np.stack([tang[:, 1], -tang[:, 0]], axis=1)

    # Per-point sign check: probe 2 px along the candidate normal; if the
    # mask sample is 0, flip. Handles any edge case where a contour winds
    # the other way (e.g. tiny isolated blobs).
    h, w = mask.shape[:2]
    probe = pts + 2.0 * normals
    px = np.clip(np.round(probe[:, 0]).astype(np.int32), 0, w - 1)
    py = np.clip(np.round(probe[:, 1]).astype(np.int32), 0, h - 1)
    inside = mask[py, px] > 127
    # Flip normals where probe landed OUTSIDE the mask.
    flip = np.where(inside, 1.0, -1.0)[:, None]
    return normals * flip


# ---------------------------------------------------------------------------
# Public: sub-pixel refinement of a single mask boundary
# ---------------------------------------------------------------------------

def refine_boundary_subpixel(mask: np.ndarray,
                             image_gray: np.ndarray,
                             band_px: float | None = None,
                             smooth_sigma: float | None = None,
                             step_px: float | None = None) -> np.ndarray:
    """
    Refine a binary-mask boundary to sub-pixel (x, y) by gradient-ridge
    snapping along inward normals.

    Args:
        mask: HxW uint8 binary mask (>127 foreground).
        image_gray: HxW uint8 or float32 single-channel or RGB image. If
            RGB is given it is converted to grayscale.
        band_px: Half-width of the search band along each normal. Default
            cfg.BOUNDARY_GRADIENT_BAND_PX.
        smooth_sigma: Gaussian sigma applied before Sobel. Default
            cfg.BOUNDARY_SMOOTH_SIGMA.
        step_px: Sub-pixel sampling step along the normal. Default
            cfg.BOUNDARY_STEP_PX.

    Returns:
        Nx2 float64 array of refined (x, y) coordinates, ordered along the
        contour. Empty array if the mask has no foreground.
    """
    if band_px is None:
        band_px = cfg.BOUNDARY_GRADIENT_BAND_PX
    if smooth_sigma is None:
        smooth_sigma = cfg.BOUNDARY_SMOOTH_SIGMA
    if step_px is None:
        step_px = cfg.BOUNDARY_STEP_PX

    pts = _extract_ordered_contour(mask)
    if len(pts) < 3:
        return pts

    if image_gray.ndim == 3:
        gray = cv2.cvtColor(image_gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_gray
    gmag = _gradient_magnitude(gray, smooth_sigma)

    normals = _inward_normals(pts, (mask > 127).astype(np.uint8) * 255)

    # Offsets to probe along each normal: -band .. +band at step_px spacing.
    # Positive offset = inward, negative = outward. The true paper edge sits
    # essentially at offset 0 +- 1 px; the band just needs to be wide enough
    # to catch mask erosion/dilation jitter.
    offsets = np.arange(-band_px, band_px + step_px * 0.5, step_px,
                        dtype=np.float64)
    if len(offsets) < 3:
        return pts

    # Sample grid: (N, M)
    sample_xs = pts[:, 0][:, None] + offsets[None, :] * normals[:, 0][:, None]
    sample_ys = pts[:, 1][:, None] + offsets[None, :] * normals[:, 1][:, None]
    grads = _bilinear_sample(gmag, sample_xs, sample_ys)

    # Per-point argmax -> integer offset
    n = len(pts)
    k_max = np.argmax(grads, axis=1)

    # Parabolic sub-pixel refinement around each argmax.
    safe_k = np.clip(k_max, 1, len(offsets) - 2)
    rng = np.arange(n)
    g_m = grads[rng, safe_k - 1]
    g_0 = grads[rng, safe_k]
    g_p = grads[rng, safe_k + 1]

    denom = g_m - 2.0 * g_0 + g_p
    # Parabola has a max only when denom < 0; otherwise keep integer argmax.
    valid = denom < -1e-8
    safe_denom = np.where(valid, denom, -1.0)
    delta = 0.5 * (g_m - g_p) / safe_denom
    delta = np.where(valid, delta, 0.0)
    # safety clamp: parabolic fit can extrapolate wildly if the peak is
    # nearly flat. Bound to one sample step.
    delta = np.clip(delta, -1.0, 1.0)

    refined_offset = offsets[safe_k] + delta * step_px
    refined_pts = pts + refined_offset[:, None] * normals
    return refined_pts


# ---------------------------------------------------------------------------
# Helpers that operate on UNTORN's fragment dicts
# ---------------------------------------------------------------------------

def attach_subpixel_contour(fragment: dict,
                            image_rgb: np.ndarray) -> dict:
    """
    Refine a fragment's boundary in-place.

    Adds:
        fragment["contour_subpixel"]: Nx2 float64 refined polyline.

    Leaves the original integer `fragment["contour"]` untouched so existing
    downstream code paths keep working. Callers that want the refined
    contour read `contour_subpixel` explicitly.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    refined = refine_boundary_subpixel(fragment["mask"], gray)
    fragment["contour_subpixel"] = refined
    return fragment


def attach_subpixel_contours_all(fragments: list,
                                 image_rgb: np.ndarray) -> list:
    """Batch variant: refine every fragment's contour in-place."""
    if not cfg.BOUNDARY_REFINE_ENABLED:
        for frag in fragments:
            # fall back: store integer contour as float for a uniform type
            c = frag["contour"].reshape(-1, 2).astype(np.float64)
            frag["contour_subpixel"] = c
        return fragments
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    for frag in fragments:
        frag["contour_subpixel"] = refine_boundary_subpixel(
            frag["mask"], gray
        )
    return fragments
