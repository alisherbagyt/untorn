"""
untorn.contours
===============
Phase 2: Extract contours, support points, and curvature features.

Per-fragment fields stored on the dict:
    contour_subpixel, support_points, boundary_pixels, text_lines

Each torn edge gets a curvature string downstream in matching.prepare_edges_
and_sdt; the per-edge data is built by `fragment_io` (Step 2 onwards).

Note (post Step 1 cleanup): we no longer store `edge_segments`, `sdf`, or
`color_profile` on the fragment dict — those were never consumed downstream
and the SDT is regenerated as `_sdt_interior` inside `matching.prepare_edges_
and_sdt` from the current sub-pixel-aware mask. The SDF visualisation PNG is
still emitted because the frontend reads it.
"""

import json
import numpy as np
import cv2
from pathlib import Path

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


def _signed_distance_for_viz(mask: np.ndarray) -> np.ndarray:
    """SDT used only for the per-fragment debug PNG. Float32 H×W;
    positive = inside, negative = outside. Computed with cv2.distanceTransform
    (faster than scipy and within ±0.5 px of the true Euclidean distance).
    """
    fg = (mask > 127).astype(np.uint8) * 255
    bg = (mask <= 127).astype(np.uint8) * 255
    dist_fg = cv2.distanceTransform(fg, cv2.DIST_L2, 3).astype(np.float32)
    dist_bg = cv2.distanceTransform(bg, cv2.DIST_L2, 3).astype(np.float32)
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


def analyze_fragments(fragments: list[dict], image_rgb: np.ndarray,
                      debug_dir: Path) -> list[dict]:
    """
    Phase 2 entry point. Delegates per-fragment analysis to
    `fragment_io.build_all` (single ingest path) and renders the debug
    overlays + meta JSON.

    Post-Step-2 the per-fragment fields (contour_subpixel, support_points,
    boundary_pixels, edges with curvature strings, _sdt_interior, text_lines,
    text_angle_canonical, paper_lab, ink_density) are all populated by
    fragment_io. This wrapper only emits the debug artefacts the frontend
    consumes.
    """
    from . import fragment_io

    contour_debug = debug_dir / "contours"
    contour_debug.mkdir(parents=True, exist_ok=True)

    print("  Per-fragment canonical analysis (boundary, edges, text, paper) ...")
    fragment_io.build_all(fragments, image_rgb)

    vis = image_rgb.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)]
    all_meta: list[dict] = []

    for frag in fragments:
        fid = frag["id"]
        c = colors[fid % len(colors)]
        sp = frag["support_points"]
        bp = frag["boundary_pixels"]
        edges = frag.get("edges", [])

        # Polygon-segment lengths derived from support points — used only for
        # the meta JSON (frontend chart reads `edge_lengths`).
        segs = compute_edge_segments(sp)

        # Whole-fragment SDT for visualization only.
        sdf = _signed_distance_for_viz(frag["mask"])

        # All-fragments overlay: contour, baselines, support points.
        cv2.drawContours(vis, [frag["contour"]], -1, c, 2)
        for tl in frag.get("text_lines", []):
            p0 = tuple(np.round(tl["p0"]).astype(int))
            p1 = tuple(np.round(tl["p1"]).astype(int))
            cv2.line(vis, p0, p1, c, 1, cv2.LINE_AA)
        for k, pt in enumerate(sp):
            cv2.circle(vis, tuple(pt), 5, c, -1)
            cv2.putText(vis, str(k), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

        # Per-fragment support-points crop.
        frag_vis = image_rgb.copy()
        cv2.drawContours(frag_vis, [frag["contour"]], -1, c, 2)
        for pt in sp:
            cv2.circle(frag_vis, tuple(pt), 5, c, -1)
        x, y, bw, bh = frag["bbox"]
        pad = 30
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(image_rgb.shape[1], x + bw + pad), min(image_rgb.shape[0], y + bh + pad)
        save_image(frag_vis[y0:y1, x0:x1],
                   str(contour_debug / f"support_pts_{fid:02d}.png"))

        # SDF debug PNG, clamped to [-50, +50] for visible detail.
        clamp = 50.0
        sdf_clamped = np.clip(sdf, -clamp, clamp)
        sdf_u8 = ((sdf_clamped + clamp) / (2 * clamp) * 255).astype(np.uint8)
        sdf_color = cv2.applyColorMap(sdf_u8, cv2.COLORMAP_JET)
        sdf_color = cv2.cvtColor(sdf_color, cv2.COLOR_BGR2RGB)
        contour_mask = np.zeros(sdf.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [frag["contour"]], -1, 255, 2)
        sdf_color[contour_mask > 0] = [255, 255, 255]
        save_image(sdf_color[y0:y1, x0:x1],
                   str(contour_debug / f"sdf_{fid:02d}.png"))

        n_text_lines = len(frag.get("text_lines", []))
        n_torn = sum(1 for e in edges if e.get("is_torn"))
        n_factory = len(edges) - n_torn
        text_angle = frag.get("text_angle_canonical")
        meta = {
            "id":                  fid,
            "n_support_points":    len(sp),
            "n_edge_segments":     len(segs),
            "n_boundary_pixels":   len(bp),
            "n_text_lines":        n_text_lines,
            "n_torn_edges":        n_torn,
            "n_factory_edges":     n_factory,
            "total_perimeter":     round(sum(s["length"] for s in segs), 1),
            "edge_lengths":        [round(s["length"], 1) for s in segs],
            "text_angle_deg":      (round(float(np.degrees(text_angle)), 2)
                                     if text_angle is not None else None),
            "ink_density":         round(float(frag.get("ink_density", 0.0)), 4),
        }
        all_meta.append(meta)
        print(f"    Fragment {fid}: {len(sp)} sp, "
              f"{n_torn} torn / {n_factory} factory edges, "
              f"{n_text_lines} text lines, "
              f"perimeter={meta['total_perimeter']:.0f}px"
              + (f", text_angle={meta['text_angle_deg']:+.1f}°"
                 if text_angle is not None else ""))

    save_image(vis, str(contour_debug / "all_support_points.png"))
    with open(contour_debug / "contours_meta.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    return fragments