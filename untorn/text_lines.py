"""
untorn.text_lines
=================
Lightweight per-fragment text-line detection.

UNTORN reconstructs mixed printed + handwritten documents, which means
text baselines are a much stronger signal than edge curvature alone:
lines should be horizontal after a correct global rotation, and they
must continue across matched seams. This module exposes that signal to
the matcher and to the global rotation pass in assembly.

Approach (no learned model):
    1. Binarise the fragment interior to ink pixels (grayscale < 140,
       masked).
    2. Search candidate baseline angles in a +-30 deg window by rotating
       the ink image and computing a horizontal projection profile at
       each candidate. The angle whose profile has the sharpest peaks is
       taken as the fragment's text orientation.
    3. Peak-find the best profile to locate baselines. For each peak,
       estimate the baseline extent from the ink row profile and back-
       project the endpoints to the working-image coordinate frame.

Each detected baseline is stored as:
    {
        "p0":        (x0, y0),       start point, working-image coords
        "p1":        (x1, y1),       end   point, working-image coords
        "center":    (cx, cy),       midpoint
        "angle":     float,          baseline angle in radians (from +x)
        "length":    float,          length in px
        "confidence":float,          0-1 peak prominence normalized
    }

The points are in the SAME coordinate frame as `frag["contour"]` - i.e.
working-image / scan space. Applying the fragment's placement transform
(3x3 affine) moves the baseline with the fragment.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import rotate as nd_rotate, gaussian_filter1d
from scipy.signal import find_peaks

from . import config as cfg


# ---------------------------------------------------------------------------
# Ink mask extraction
# ---------------------------------------------------------------------------

def _ink_mask(image_gray_crop: np.ndarray,
              fragment_mask_crop: np.ndarray,
              ink_grayscale_max: int = 140) -> np.ndarray:
    """Binary uint8 ink mask: inside the fragment AND dark enough to be ink."""
    inside = fragment_mask_crop > 127
    dark = image_gray_crop < ink_grayscale_max
    return (inside & dark).astype(np.uint8)


# ---------------------------------------------------------------------------
# Rotation + horizontal projection profile
# ---------------------------------------------------------------------------

def _projection_profile(ink_crop: np.ndarray, angle_deg: float,
                        smooth_sigma: float = 2.0) -> np.ndarray:
    """
    Sum-of-ink per row after rotating the crop by `angle_deg`, smoothed
    with a 1-D Gaussian so a thick line's projection plateau becomes one
    peak rather than several sub-peaks at the plateau edges.
    """
    if abs(angle_deg) < 1e-4:
        rot = ink_crop
    else:
        rot = nd_rotate(ink_crop, angle_deg, reshape=True,
                        order=0, mode="constant", cval=0)
    profile = rot.sum(axis=1).astype(np.float32)
    if smooth_sigma > 0 and len(profile) > 3:
        profile = gaussian_filter1d(profile, sigma=smooth_sigma)
    return profile


def _line_score(profile: np.ndarray, min_peak_height: float) -> float:
    """
    Heuristic 'peakiness' of a 1-D projection profile. Returns roughly the
    summed prominence of all peaks above `min_peak_height`. Documents with
    crisp text lines score highly; noisy/textureless fragments score near 0.
    """
    if len(profile) < 5:
        return 0.0
    peaks, props = find_peaks(profile,
                              height=min_peak_height,
                              distance=8,
                              prominence=min_peak_height * 0.4)
    if len(peaks) == 0:
        return 0.0
    return float(np.sum(props.get("prominences", 0.0)))


# ---------------------------------------------------------------------------
# Back-projection from rotated-crop coords to working-image coords
# ---------------------------------------------------------------------------

def _rotated_to_original(xs_rot: np.ndarray,
                         ys_rot: np.ndarray,
                         crop_shape: tuple,
                         rot_shape: tuple,
                         angle_deg: float,
                         bbox_origin: tuple) -> np.ndarray:
    """
    Inverse of scipy.ndimage.rotate(reshape=True). Given coordinates in the
    rotated (and potentially larger) frame, return their location in the
    original crop frame, then shifted by the bbox origin into working-image
    space.
    """
    ch, cw = crop_shape[:2]
    rh, rw = rot_shape[:2]
    bx, by = bbox_origin

    # scipy rotates around the ORIGINAL center and reshapes to fit.
    # Forward: q = R * (p - c_orig) + c_rot   (c_rot = center of rot frame)
    # Inverse: p = R^T * (q - c_rot) + c_orig
    c_orig = np.array([(cw - 1) / 2.0, (ch - 1) / 2.0])
    c_rot = np.array([(rw - 1) / 2.0, (rh - 1) / 2.0])

    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # scipy rotates COUNTERCLOCKWISE when angle is positive.
    # Forward R =  [ cos, -sin; sin, cos ]
    # Inverse R^T = [ cos, sin; -sin, cos ]
    q = np.stack([xs_rot - c_rot[0], ys_rot - c_rot[1]], axis=-1)
    x_orig = cos_t * q[..., 0] + sin_t * q[..., 1] + c_orig[0]
    y_orig = -sin_t * q[..., 0] + cos_t * q[..., 1] + c_orig[1]

    return np.stack([x_orig + bx, y_orig + by], axis=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_text_lines(fragment: dict,
                      image_rgb: np.ndarray,
                      max_angle_deg: float | None = None,
                      angle_step_deg: float = 2.0,
                      min_ink_frac: float | None = None) -> list[dict]:
    """
    Detect text baselines inside a fragment. Returns [] if the fragment
    has essentially no ink or no detectable text structure.

    Coordinates in the returned baselines are in working-image space,
    same frame as fragment['contour'].
    """
    if max_angle_deg is None:
        max_angle_deg = getattr(cfg, "TEXT_LINE_ANGLE_SEARCH_DEG", 30.0)
    if min_ink_frac is None:
        min_ink_frac = getattr(cfg, "TEXT_LINE_MIN_INK_FRAC", 0.02)

    bbox = fragment["bbox"]
    bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h_img, w_img = fragment["mask"].shape[:2]
    bx_e = min(bx + bw, w_img)
    by_e = min(by + bh, h_img)

    if image_rgb.ndim == 3:
        gray_full = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray_full = image_rgb
    gray = gray_full[by:by_e, bx:bx_e]
    mask = fragment["mask"][by:by_e, bx:bx_e]

    ink = _ink_mask(gray, mask)
    if ink.sum() < max(50, 0.002 * ink.size):
        return []

    # Sweep candidate angles. min_peak_height is a fraction of max possible
    # per-row ink count (= crop width). Adapted to rotated width below.
    best_angle = 0.0
    best_score = -1.0
    best_profile: np.ndarray | None = None
    best_rot_shape = ink.shape

    angles = np.arange(-max_angle_deg, max_angle_deg + 1e-6,
                       angle_step_deg, dtype=np.float64)
    for a in angles:
        profile = _projection_profile(ink, float(a))
        row_w = ink.shape[1]
        # after rotation the row width equals rot_shape[1], which we pass in
        # with the profile via its length indirectly; approximate by ink.shape
        min_peak_h = min_ink_frac * row_w
        score = _line_score(profile, min_peak_h)
        if score > best_score:
            best_score = score
            best_angle = float(a)
            best_profile = profile
            # recompute rotated shape for back-projection
            if abs(a) < 1e-4:
                best_rot_shape = ink.shape
            else:
                # scipy.ndimage.rotate reshape=True yields a predictable bbox
                rot_h = int(round(abs(ink.shape[0] * np.cos(np.deg2rad(a))) +
                                  abs(ink.shape[1] * np.sin(np.deg2rad(a)))))
                rot_w = int(round(abs(ink.shape[1] * np.cos(np.deg2rad(a))) +
                                  abs(ink.shape[0] * np.sin(np.deg2rad(a)))))
                best_rot_shape = (rot_h, rot_w)

    if best_profile is None or best_score <= 0:
        return []

    # Peaks in the winning profile.
    row_w = best_rot_shape[1]
    min_peak_h = min_ink_frac * row_w
    peaks, props = find_peaks(best_profile,
                              height=min_peak_h,
                              distance=8,
                              prominence=min_peak_h * 0.4)
    if len(peaks) == 0:
        return []

    # For each peak row, measure ink extent in that row and back-project the
    # two endpoints into working-image coords.
    baselines: list[dict] = []
    rotated_ink = (nd_rotate(ink, best_angle, reshape=True, order=0,
                             mode="constant", cval=0)
                   if abs(best_angle) > 1e-4 else ink)

    max_prom = max(float(props["prominences"].max()), 1e-6)
    for i, py in enumerate(peaks):
        row = rotated_ink[py, :]
        xs = np.flatnonzero(row)
        if len(xs) < 8:
            continue
        x_start = float(xs[0])
        x_end = float(xs[-1])
        if x_end - x_start < 20:
            continue
        # Back-project the two endpoints (and midpoint for convenience).
        pts_rot_x = np.array([x_start, x_end], dtype=np.float64)
        pts_rot_y = np.array([py, py], dtype=np.float64)
        pts_orig = _rotated_to_original(pts_rot_x, pts_rot_y,
                                        ink.shape, best_rot_shape,
                                        best_angle, (bx, by))
        p0 = pts_orig[0]
        p1 = pts_orig[1]
        d = p1 - p0
        length = float(np.hypot(d[0], d[1]))
        if length < 20:
            continue
        angle = float(np.arctan2(d[1], d[0]))
        conf = float(props["prominences"][i]) / max_prom
        baselines.append({
            "p0": p0,
            "p1": p1,
            "center": (p0 + p1) * 0.5,
            "angle": angle,
            "length": length,
            "confidence": conf,
        })

    return baselines


# ---------------------------------------------------------------------------
# Batch helper + matching-gate utility
# ---------------------------------------------------------------------------

def attach_text_lines_all(fragments: list[dict],
                          image_rgb: np.ndarray) -> list[dict]:
    """Run detect_text_lines on every fragment; result lives in frag['text_lines']."""
    for frag in fragments:
        frag["text_lines"] = detect_text_lines(frag, image_rgb)
    return fragments


def _apply_affine(M: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous affine to a 2-D point (or batch)."""
    pt = np.atleast_2d(pt).astype(np.float64)
    ones = np.ones((pt.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([pt, ones], axis=1)
    out = homog @ M.T
    return out[:, :2] if pt.shape[0] > 1 else out[0, :2]


def text_line_continuity(lines_a: list[dict],
                         lines_b: list[dict],
                         M_a: np.ndarray,
                         M_b: np.ndarray,
                         seam_point: np.ndarray,
                         seam_normal: np.ndarray,
                         max_y_disc_px: float = 3.0,
                         max_angle_disc_deg: float = 8.0,
                         search_radius_px: float = 20.0) -> tuple[float, int]:
    """
    Score how well fragment-A's text baselines continue into fragment-B.

    Both fragments come with a proposed placement affine (M_a, M_b). We
    transform every baseline into the shared canvas frame and then, for
    each baseline of A that runs close to the seam, look for a baseline
    of B whose post-placement y-coord and angle match within the caps.

    Args:
        lines_a, lines_b: baselines as produced by detect_text_lines.
        M_a, M_b: 3x3 affines placing each fragment into the canvas.
        seam_point: (x, y) a point on the proposed seam, in canvas coords.
        seam_normal: 2-D unit normal of the seam (canvas coords).
        max_y_disc_px: vertical discontinuity allowed for a "continued" line.
        max_angle_disc_deg: angular discontinuity allowed.
        search_radius_px: how close to the seam a line must run on BOTH
            sides to be considered as "at the seam".

    Returns:
        (continuity_score, n_expected_lines)
            continuity_score in [0, 1]: fraction of near-seam A-lines that
            found a continuation on B.
            n_expected_lines: count of A-lines that ran near enough to the
            seam to participate. If this is 0, the caller should skip the
            text-line gate for this pair.
    """
    if not lines_a or not lines_b:
        return 0.0, 0

    def _near_seam(line: dict, M: np.ndarray) -> bool:
        p0 = _apply_affine(M, line["p0"])
        p1 = _apply_affine(M, line["p1"])
        # project endpoints onto seam normal
        d0 = abs(float((p0 - seam_point) @ seam_normal))
        d1 = abs(float((p1 - seam_point) @ seam_normal))
        return min(d0, d1) < search_radius_px

    def _transform(line: dict, M: np.ndarray):
        p0 = _apply_affine(M, line["p0"])
        p1 = _apply_affine(M, line["p1"])
        direction = p1 - p0
        angle = np.arctan2(direction[1], direction[0])
        return p0, p1, angle

    # Pre-compute transformed B baselines once.
    b_t = [(*_transform(lb, M_b), lb) for lb in lines_b]

    expected = 0
    continued = 0
    max_angle_rad = np.deg2rad(max_angle_disc_deg)

    for la in lines_a:
        if not _near_seam(la, M_a):
            continue
        expected += 1
        a_p0, a_p1, a_ang = _transform(la, M_a)
        # Representative y at the seam side: take the endpoint closer to seam.
        d0 = abs(float((a_p0 - seam_point) @ seam_normal))
        d1 = abs(float((a_p1 - seam_point) @ seam_normal))
        a_at_seam = a_p0 if d0 < d1 else a_p1
        # Tangent along the seam (perpendicular to normal)
        seam_tangent = np.array([-seam_normal[1], seam_normal[0]])
        a_y = float((a_at_seam - seam_point) @ seam_tangent)

        for b_p0, b_p1, b_ang, _ in b_t:
            # Pick B endpoint closer to seam
            db0 = abs(float((b_p0 - seam_point) @ seam_normal))
            db1 = abs(float((b_p1 - seam_point) @ seam_normal))
            b_at_seam = b_p0 if db0 < db1 else b_p1
            if min(db0, db1) > search_radius_px:
                continue
            b_y = float((b_at_seam - seam_point) @ seam_tangent)
            if abs(b_y - a_y) > max_y_disc_px:
                continue
            # angle diff, allowing 180-deg ambiguity
            da = abs(a_ang - b_ang)
            da = min(da, abs(da - np.pi))
            if da > max_angle_rad:
                continue
            continued += 1
            break

    if expected == 0:
        return 0.0, 0
    return continued / expected, expected
