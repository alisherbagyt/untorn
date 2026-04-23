"""
untorn.contours
===============
Phase 2: Extract contours, support points, and edge descriptors.

Based on Richter thesis §3.3 (polygonal approximation),
§4.2-4.4 (support point features), §6.3.1 (compatibility scores).
"""

import json
import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import distance_transform_edt

from . import config as cfg
from .boundary import attach_subpixel_contours_all
from .text_lines import attach_text_lines_all
from .io_utils import save_image, save_mask


def extract_support_points(contour: np.ndarray) -> np.ndarray:
    """
    Douglas-Peucker polygonal approximation → support points (Thesis §3.3).
    Returns Nx2 array of (x, y) coordinates, ordered counterclockwise.
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = cfg.POLY_EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape(-1, 2)


def compute_edge_segments(support_points: np.ndarray) -> list[dict]:
    """
    Break the polygon into edge segments between consecutive support points.
    Each segment stores: start_idx, end_idx, start_pt, end_pt, length, angle,
    midpoint, normal direction.
    """
    n = len(support_points)
    segments = []
    for i in range(n):
        j = (i + 1) % n
        p0 = support_points[i].astype(float)
        p1 = support_points[j].astype(float)
        d = p1 - p0
        length = np.linalg.norm(d)
        angle = np.arctan2(d[1], d[0])
        mid = (p0 + p1) / 2
        # Inward-pointing normal (90° clockwise for CCW contour)
        if length > 0:
            normal = np.array([d[1], -d[0]]) / length
        else:
            normal = np.array([0.0, 0.0])

        segments.append({
            "start_idx": i,
            "end_idx": j,
            "start_pt": p0,
            "end_pt": p1,
            "length": float(length),
            "angle": float(angle),
            "midpoint": mid,
            "normal": normal,
        })
    return segments


def extract_boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixel coords (Nx2, x/y) from binary mask."""
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = cv2.bitwise_xor(mask, eroded)
    ys, xs = np.where(boundary > 127)
    return np.column_stack([xs, ys])


def compute_signed_distance_map(mask: np.ndarray) -> np.ndarray:
    """
    Signed distance transform (Thesis §8.4).
    Positive = inside (foreground), negative = outside (background), 0 = boundary.
    """
    fg = mask > 127
    bg = ~fg
    dist_fg = distance_transform_edt(fg)
    dist_bg = distance_transform_edt(bg)
    return dist_fg - dist_bg


def sample_contour_pixels(contour: np.ndarray, n_samples: int = 500) -> np.ndarray:
    """
    Densely sample points along a contour polyline.
    Returns Nx2 array of (x,y) coordinates evenly spaced along the contour.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    # Compute cumulative arc length
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]

    if total < 1:
        return pts

    # Sample evenly
    sample_dists = np.linspace(0, total, n_samples, endpoint=False)
    sampled = np.zeros((n_samples, 2))
    for k, sd in enumerate(sample_dists):
        idx = np.searchsorted(cum_lengths, sd, side='right') - 1
        idx = min(idx, len(pts) - 2)
        local_t = (sd - cum_lengths[idx]) / max(seg_lengths[idx], 1e-8)
        sampled[k] = pts[idx] + local_t * diffs[idx]

    return sampled


# ═══════════════════════════════════════════════════════════════════════════
#  Curvature feature strings (Wolfson turning function — Paper 1 §2.2)
# ═══════════════════════════════════════════════════════════════════════════

def resample_arc_length(pts: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Resample an open polyline at equal arc-length intervals.
    Returns Nx2 array of resampled (x,y) coordinates.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]

    if total < 1e-6:
        return pts[:1].copy()

    sample_dists = np.linspace(0, total, n_samples)
    sampled = np.zeros((n_samples, 2))

    for k, sd in enumerate(sample_dists):
        idx = np.searchsorted(cum_lengths, sd, side='right') - 1
        idx = min(idx, len(pts) - 2)
        local_t = (sd - cum_lengths[idx]) / max(seg_lengths[idx], 1e-8)
        local_t = min(local_t, 1.0)
        sampled[k] = pts[idx] + local_t * diffs[idx]

    return sampled


def compute_curvature_string(pts: np.ndarray,
                              n_samples: int = None,
                              window: int = None):
    """
    Wolfson turning function → curvature feature string (Paper 1 §2.2).

    Steps:
      1. Resample edge points at equal arc-length intervals
      2. Compute tangent angle at each sample (turning function)
      3. Compute Δθ (change in angle — curvature approximation)
      4. Smooth with running average of `window` consecutive values

    The resulting feature string is invariant to rotation and translation.

    Returns:
        (resampled_pts, curvature_string)
        - resampled_pts: Nx2 array of resampled points
        - curvature_string: 1D array of smoothed curvature values
          (length ≈ n_samples - window - 1)
    """
    if n_samples is None:
        n_samples = cfg.CURV_N_SAMPLES
    if window is None:
        window = cfg.CURV_SMOOTH_WINDOW

    resampled = resample_arc_length(pts, n_samples)

    if len(resampled) < window + 3:
        return resampled, np.array([], dtype=np.float64)

    # Turning function: angle of tangent at each point
    tangents = np.diff(resampled, axis=0)                      # (N-1) x 2
    angles = np.arctan2(tangents[:, 1], tangents[:, 0])        # (N-1,)

    # Unwrap to avoid ±π discontinuities
    angles = np.unwrap(angles)

    # Δθ: change in angle ≈ curvature
    dtheta = np.diff(angles)                                   # (N-2,)

    if len(dtheta) < window:
        return resampled, dtheta

    # Smooth with running average for noise robustness
    kernel = np.ones(window, dtype=np.float64) / window
    curvature = np.convolve(dtheta, kernel, mode='valid')      # (N-2-window+1,)

    return resampled, curvature


def extract_edge_color_profile(image_rgb: np.ndarray, mask: np.ndarray,
                                contour: np.ndarray, depth: int = 8) -> np.ndarray:
    """
    Extract color profile along the inside of the contour boundary (Thesis §4.4.1 LCE).
    For each boundary point, sample `depth` pixels inward and store their colors.
    Returns Nx(depth*3) array.
    """
    h, w = image_rgb.shape[:2]
    boundary_pts = sample_contour_pixels(contour, n_samples=300)

    # Compute contour normals (inward)
    n_pts = len(boundary_pts)
    profiles = []
    for i in range(n_pts):
        p = boundary_pts[i]
        p_prev = boundary_pts[(i - 1) % n_pts]
        p_next = boundary_pts[(i + 1) % n_pts]

        tangent = p_next - p_prev
        tlen = np.linalg.norm(tangent)
        if tlen < 1e-6:
            continue
        tangent /= tlen

        # Inward normal: for CCW contour, rotate tangent 90° clockwise
        normal = np.array([tangent[1], -tangent[0]])

        # Check if normal points inward (should point into the mask)
        test_pt = (p + normal * 3).astype(int)
        if 0 <= test_pt[0] < w and 0 <= test_pt[1] < h:
            if mask[test_pt[1], test_pt[0]] == 0:
                normal = -normal  # flip

        # Sample depth pixels inward
        colors = []
        for d in range(1, depth + 1):
            sp = (p + normal * d).astype(int)
            sx, sy = np.clip(sp[0], 0, w-1), np.clip(sp[1], 0, h-1)
            if mask[sy, sx] > 0:
                colors.extend(image_rgb[sy, sx].tolist())
            else:
                colors.extend([0, 0, 0])

        profiles.append(colors)

    return np.array(profiles, dtype=np.float32) if profiles else np.zeros((0, depth*3), dtype=np.float32)


def analyze_fragments(fragments: list[dict], image_rgb: np.ndarray,
                      debug_dir: Path) -> list[dict]:
    """
    Compute support points, edge segments, boundary info, and color profiles
    for each fragment. Saves debug output.
    """
    contour_debug = debug_dir / "contours"
    contour_debug.mkdir(parents=True, exist_ok=True)

    # Sub-pixel boundary refinement: snap each mask-boundary pixel to the
    # local gradient ridge along its inward normal. Downstream curvature
    # extraction and edge matching read `frag["contour_subpixel"]` and fall
    # back to the integer `frag["contour"]` if disabled.
    if cfg.BOUNDARY_REFINE_ENABLED:
        print("  Refining fragment boundaries to sub-pixel ...")
    attach_subpixel_contours_all(fragments, image_rgb)

    # Per-fragment text-line detection (projection profile + baseline fit).
    # Stored on frag["text_lines"]; used by the matching text-continuity gate.
    print("  Detecting text baselines ...")
    attach_text_lines_all(fragments, image_rgb)

    print("  Extracting support points and edge descriptors ...")

    vis = image_rgb.copy()
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
              (0,255,255),(128,0,0),(0,128,0),(0,0,128),(128,128,0)]
    all_meta = []

    for frag in fragments:
        fid = frag["id"]
        c = colors[fid % len(colors)]

        # Support points
        sp = extract_support_points(frag["contour"])
        frag["support_points"] = sp

        # Edge segments
        segs = compute_edge_segments(sp)
        frag["edge_segments"] = segs

        # Boundary pixels
        bp = extract_boundary_pixels(frag["mask"])
        frag["boundary_pixels"] = bp

        # Signed distance map
        sdf = compute_signed_distance_map(frag["mask"])
        frag["sdf"] = sdf

        # Edge color profile
        color_prof = extract_edge_color_profile(image_rgb, frag["mask"], frag["contour"])
        frag["color_profile"] = color_prof

        # Visualization
        cv2.drawContours(vis, [frag["contour"]], -1, c, 2)
        for tl in frag.get("text_lines", []):
            p0 = tuple(np.round(tl["p0"]).astype(int))
            p1 = tuple(np.round(tl["p1"]).astype(int))
            cv2.line(vis, p0, p1, c, 1, cv2.LINE_AA)
        for k, pt in enumerate(sp):
            cv2.circle(vis, tuple(pt), 5, c, -1)
            cv2.putText(vis, str(k), (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

        # Save individual support point image
        frag_vis = image_rgb.copy()
        cv2.drawContours(frag_vis, [frag["contour"]], -1, c, 2)
        for pt in sp:
            cv2.circle(frag_vis, tuple(pt), 5, c, -1)
        x, y, bw, bh = frag["bbox"]
        pad = 30
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(image_rgb.shape[1], x+bw+pad), min(image_rgb.shape[0], y+bh+pad)
        save_image(frag_vis[y0:y1, x0:x1], str(contour_debug / f"support_pts_{fid:02d}.png"))

        # Save SDF visualization — clamp to [-50, +50] for visible detail
        clamp = 50.0
        sdf_clamped = np.clip(sdf, -clamp, clamp)
        # Normalize to 0-255: -50→0, 0→128, +50→255
        sdf_u8 = ((sdf_clamped + clamp) / (2 * clamp) * 255).astype(np.uint8)
        sdf_color = cv2.applyColorMap(sdf_u8, cv2.COLORMAP_JET)
        sdf_color = cv2.cvtColor(sdf_color, cv2.COLOR_BGR2RGB)
        # Overlay contour in white for reference
        contour_mask = np.zeros(sdf.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [frag["contour"]], -1, 255, 2)
        sdf_color[contour_mask > 0] = [255, 255, 255]
        save_image(sdf_color[y0:y1, x0:x1], str(contour_debug / f"sdf_{fid:02d}.png"))

        n_text_lines = len(frag.get("text_lines", []))
        meta = {
            "id": fid,
            "n_support_points": len(sp),
            "n_edge_segments": len(segs),
            "n_boundary_pixels": len(bp),
            "n_text_lines": n_text_lines,
            "total_perimeter": round(sum(s["length"] for s in segs), 1),
            "edge_lengths": [round(s["length"], 1) for s in segs],
        }
        all_meta.append(meta)
        print(f"    Fragment {fid}: {len(sp)} support pts, "
              f"{len(segs)} edges, {n_text_lines} text lines, "
              f"perimeter={meta['total_perimeter']:.0f}px")

    save_image(vis, str(contour_debug / "all_support_points.png"))
    with open(contour_debug / "contours_meta.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    return fragments