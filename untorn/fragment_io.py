"""
untorn.fragment_io
==================
Phase 2 — per-fragment canonical analysis.

The single ingest path that turns a raw segmentation fragment
``{id, mask, bbox, area, contour, centroid}`` into a fully-populated
descriptor with everything the downstream matcher and assembler need:

  1. Sub-pixel contour       (boundary.py — gradient-ridge snap).
  2. Support points          (Douglas-Peucker on the integer contour).
  3. Boundary pixel set      (cheap, used by the boundary-proximity gate).
  4. Torn / factory edges    (RANSAC line-fit + straightness classifier).
  5. Per-torn-edge curvature strings (Wolfson turning function).
  6. Interior SDT            (cv2.distanceTransform on the mask).
  7. Text baselines          (text_lines.detect_text_lines).
  8. Canonical text rotation (signed angle that would lay text horizontal).
  9. Paper-LAB fingerprint   (median LAB of non-ink interior pixels).
 10. Whole-fragment ink density.

Why this module exists
----------------------
Pre-rework, the per-fragment build was scattered:
  contours.analyze_fragments → support_points + dead fields
  boundary.attach_subpixel_contours_all → contour_subpixel
  text_lines.attach_text_lines_all → text_lines
  matching.prepare_edges_and_sdt → edges + curvature + SDT + paper_lab

Each one wrote to a different subset of the fragment dict and the matcher
re-extracted edges anyway. This module replaces the scattered sequence with
``build_all(fragments, image_rgb)``.

Output: every fragment dict carries the keys

    contour_subpixel, support_points, boundary_pixels,
    edges (each with pts, length, is_torn, outward_normal, midpoint,
           start_sp, end_sp, _resampled, _curvature for torn edges),
    _sdt_interior,
    text_lines, text_angle_canonical,
    paper_lab,
    ink_density.

Downstream code reads from the dict; this is intentional — the existing
modules that already consume `frag["edges"]`, `frag["mask"]`, etc. don't have
to change.
"""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

from . import config as cfg
from .boundary import attach_subpixel_contours_all
from .contours import (
    extract_support_points,
    extract_boundary_pixels,
    compute_curvature_string,
)
from .text_lines import attach_text_lines_all


# ══════════════════════════════════════════════════════════════════════════
#  Edge classification helpers (moved from matching.py)
#
#  An edge is "factory" if it lies on a straight line within a tight
#  tolerance — those are document corners and original page edges, never
#  the seam of a tear. Everything else is "torn" and gets a curvature
#  string for matching.
# ══════════════════════════════════════════════════════════════════════════

def _ransac_line_inlier_ratio(pts: np.ndarray,
                              threshold_px: float = 1.5,
                              n_iter: int = 50) -> float:
    """Fraction of points within `threshold_px` of the best-fit line."""
    if len(pts) < 3:
        return 1.0
    n = len(pts)
    best = 0.0
    rng = np.random.default_rng(seed=42)
    for _ in range(n_iter):
        idx = rng.choice(n, 2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        d = p2 - p1
        ll = float(np.linalg.norm(d))
        if ll < 1e-6:
            continue
        normal = np.array([-d[1], d[0]]) / ll
        dists = np.abs((pts - p1) @ normal)
        ratio = float(np.sum(dists < threshold_px) / n)
        if ratio > best:
            best = ratio
    return best


def _straightness(pts: np.ndarray) -> float:
    """End-to-end / arc-length ratio. 1.0 = perfectly straight."""
    if len(pts) < 2:
        return 1.0
    return float(np.linalg.norm(pts[-1] - pts[0]) / max(
        np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)), 1e-6))


def _classify_edge(pts: np.ndarray) -> tuple[bool, float, float]:
    """Returns (is_torn, ransac_inlier_ratio, straightness)."""
    s = _straightness(pts)
    r = _ransac_line_inlier_ratio(pts, threshold_px=1.5)
    is_factory = r > 0.92 and s > 0.98
    return (not is_factory), r, s


def _compute_outward_normal(edge_pts: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
    """
    Unit normal at the edge midpoint pointing AWAY from the fragment mask.

    Robust against tiny fragments by trying a series of probe distances; if
    none disambiguate (mask is too small for any probe), fall back to the
    candidate that points from the centroid of `edge_pts` to the centroid
    of the mask negation, which is geometrically correct even when no
    integer probe lands inside the mask.
    """
    if len(edge_pts) < 2:
        return np.array([0.0, 0.0])
    direction = edge_pts[-1] - edge_pts[0]
    dl = float(np.linalg.norm(direction))
    if dl < 1e-6:
        return np.array([0.0, 0.0])
    direction = direction / dl
    n1 = np.array([-direction[1], direction[0]])
    n2 = -n1
    mid = edge_pts[len(edge_pts) // 2]
    h, w = mask.shape

    # Try a generous range of probe distances. For very small fragments
    # (~30-50 px) the longer probes always escape the mask; the short
    # ones disambiguate.
    for step in (2, 3, 5, 8, 12, 20, 35):
        t1 = (mid + n1 * step).astype(int)
        t2 = (mid + n2 * step).astype(int)
        in1 = (0 <= t1[0] < w and 0 <= t1[1] < h and mask[t1[1], t1[0]] > 127)
        in2 = (0 <= t2[0] < w and 0 <= t2[1] < h and mask[t2[1], t2[0]] > 127)
        if in1 and not in2:
            return n2
        if in2 and not in1:
            return n1

    # Probes inconclusive — sample the mask centroid relative to the
    # midpoint and pick the normal pointing AWAY from it. This is what we
    # mean by "outward" geometrically.
    ys, xs = np.where(mask > 127)
    if len(xs):
        mc = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        away = mid - mc
        if float(away @ n1) > 0:
            return n1
        return n2
    return n1


# ══════════════════════════════════════════════════════════════════════════
#  Edge extraction from contour + support points
# ══════════════════════════════════════════════════════════════════════════

def _extract_edges_from_contour(contour: np.ndarray,
                                support_points: np.ndarray,
                                mask: np.ndarray,
                                min_edge_length: float = 15.0
                                ) -> list[dict]:
    """Break the contour polyline at support points into edges.

    Returns a list of edge dicts:
        pts          (N, 2) float64 — polyline pixels in CCW order
        length       float arc length
        straightness float
        ransac_inlier_ratio float
        is_torn      bool
        midpoint     (2,) float64
        direction    (2,) unit vector along edge endpoints
        outward_normal (2,) unit vector pointing away from mask
        start_sp     int — support-point index at edge start
        end_sp       int — support-point index at edge end
    """
    contour_pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    n_sp = len(support_points)
    if n_sp == 0 or len(contour_pts) < 3:
        return []

    # Snap each support point to its nearest contour pixel index.
    sp_indices = []
    for sp in support_points:
        dists = np.linalg.norm(contour_pts - sp.astype(np.float64), axis=1)
        sp_indices.append(int(np.argmin(dists)))
    order = np.argsort(sp_indices)
    sp_sorted = [sp_indices[o] for o in order]
    sp_ids = [int(order[k]) for k in range(n_sp)]

    edges: list[dict] = []
    for k in range(n_sp):
        sci = sp_sorted[k]
        eci = sp_sorted[(k + 1) % n_sp]
        if eci > sci:
            epts = contour_pts[sci:eci + 1]
        else:
            epts = np.vstack([contour_pts[sci:], contour_pts[:eci + 1]])
        if len(epts) < 2:
            continue
        al = float(np.sum(np.linalg.norm(np.diff(epts, axis=0), axis=1)))
        if al < min_edge_length:
            continue

        is_torn, rr, st = _classify_edge(epts)
        mid = epts[len(epts) // 2].copy()
        d = epts[-1] - epts[0]
        dl = float(np.linalg.norm(d))
        if dl > 0:
            d = d / dl
        on = _compute_outward_normal(epts, mask)
        edges.append({
            "pts":                  epts,
            "start_sp":             sp_ids[k],
            "end_sp":               sp_ids[(k + 1) % n_sp],
            "length":               al,
            "straightness":         st,
            "ransac_inlier_ratio":  rr,
            "is_torn":              is_torn,
            "midpoint":             mid,
            "direction":            d,
            "outward_normal":       on,
        })
    return edges


# ══════════════════════════════════════════════════════════════════════════
#  Paper-LAB fingerprint (per fragment, median of non-ink interior)
# ══════════════════════════════════════════════════════════════════════════

def _paper_lab_fingerprint(image_rgb: np.ndarray,
                           mask: np.ndarray,
                           erode_px: int = 5,
                           ink_grayscale_max: int = 140
                           ) -> np.ndarray | None:
    """Median LAB of paper-only pixels — interior, away from boundary, not ink."""
    if mask is None:
        return None
    k = max(1, int(erode_px))
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    interior = cv2.erode((mask > 127).astype(np.uint8), kernel)
    if interior.sum() < 16:
        interior = (mask > 127).astype(np.uint8)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    paper_mask = (interior > 0) & (gray >= ink_grayscale_max)
    if paper_mask.sum() < 16:
        paper_mask = interior > 0
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pts = lab[paper_mask]
    if pts.size == 0:
        return None
    return np.median(pts, axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  Canonical text-rotation angle
# ══════════════════════════════════════════════════════════════════════════

def _canonical_text_angle(text_lines: list[dict]) -> float | None:
    """
    Median local baseline angle in [-π/2, π/2). Returns the angle that
    would need to be ROTATED OUT to lay text horizontal — i.e. the angle
    you would PASS to a rotation matrix as -θ.

    Returns None when there is not enough text signal (≤ 1 baseline).
    """
    if not text_lines or len(text_lines) < 1:
        return None
    angles: list[float] = []
    for ln in text_lines:
        a = float(ln.get("angle", 0.0))
        if a >= math.pi / 2:
            a -= math.pi
        elif a < -math.pi / 2:
            a += math.pi
        angles.append(a)
    if not angles:
        return None
    return float(np.median(angles))


# ══════════════════════════════════════════════════════════════════════════
#  Public entry points
# ══════════════════════════════════════════════════════════════════════════

def build_descriptor(frag: dict, image_rgb: np.ndarray,
                     gray_full: np.ndarray | None = None) -> dict:
    """
    Run the full per-fragment analysis on `frag` (in place).

    Idempotent: if descriptor fields already exist they are overwritten.
    """
    mask = frag["mask"]

    # 1. Sub-pixel contour. Stored on frag["contour_subpixel"].
    if cfg.BOUNDARY_REFINE_ENABLED:
        from .boundary import refine_boundary_subpixel
        if gray_full is None:
            gray_full = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        frag["contour_subpixel"] = refine_boundary_subpixel(
            mask, gray_full)
    else:
        frag["contour_subpixel"] = (
            np.asarray(frag["contour"], dtype=np.float64).reshape(-1, 2))

    # 2. Support points (Douglas-Peucker on the integer contour — we want
    #    integer support indices for the edge-segment break logic).
    sp = extract_support_points(frag["contour"])
    frag["support_points"] = sp

    # 3. Boundary pixels.
    frag["boundary_pixels"] = extract_boundary_pixels(mask)

    # 4. Torn/factory edges. We feed the sub-pixel contour when available
    #    so curvature strings reflect sub-pixel boundary geometry; the
    #    integer support points snap onto the closest sub-pixel point.
    contour_for_edges = frag["contour_subpixel"]
    if contour_for_edges is None or len(contour_for_edges) < 3:
        contour_for_edges = frag["contour"]
    edges = _extract_edges_from_contour(
        contour_for_edges, sp, mask,
        min_edge_length=15.0)

    # 5. Per-torn-edge curvature strings.
    for e in edges:
        if e.get("is_torn"):
            resamp, curv = compute_curvature_string(e["pts"])
            e["_resampled"] = resamp
            e["_curvature"] = curv
    frag["edges"] = edges

    # 6. Interior SDT — cv2 is fine for this, ~1 ms per fragment.
    fg = (mask > 127).astype(np.uint8) * 255
    frag["_sdt_interior"] = cv2.distanceTransform(
        fg, cv2.DIST_L2, 3).astype(np.float32)

    # 7. Text baselines — populates frag["text_lines"].
    from .text_lines import detect_text_lines
    frag["text_lines"] = detect_text_lines(frag, image_rgb)

    # 8. Canonical text rotation (None if no usable text signal).
    frag["text_angle_canonical"] = _canonical_text_angle(frag["text_lines"])

    # 9. Paper-LAB fingerprint.
    frag["paper_lab"] = _paper_lab_fingerprint(image_rgb, mask)

    # 10. Whole-fragment ink density.
    if gray_full is None:
        gray_full = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    fg_bool = mask > 127
    denom = int(fg_bool.sum())
    if denom > 0:
        ink = fg_bool & (gray_full < int(getattr(cfg, "TEXT_LINE_INK_GRAYSCALE_MAX", 140)))
        frag["ink_density"] = float(ink.sum() / denom)
    else:
        frag["ink_density"] = 0.0

    return frag


def _anchor_strength(length_px: float,
                     curv_std: float,
                     curv_range: float,
                     ink_density: float,
                     reference_length_px: float) -> float:
    """Per-torn-edge "how matchable is this edge" score in [0, 1].

    Long, curvy, ink-rich edges score near 1.0; short, near-straight,
    ink-poor edges score near 0.0. Used by the assembler to order
    candidate enumeration so the strongest seams are matched first.
    """
    len_score = min(1.0, length_px / max(reference_length_px, 1.0))
    curv_score = (min(1.0, curv_std / 0.35) * 0.6
                  + min(1.0, curv_range / 1.6) * 0.4)
    ink_score = min(1.0, ink_density / 0.2)
    return float(0.5 * len_score + 0.35 * curv_score + 0.15 * ink_score)


def _assign_anchor_strengths(fragments: list[dict]) -> None:
    """Two-pass anchor-strength assignment: gather reference length across the
    whole scene, then score each torn edge against it. Mutates frag["edges"]
    in place by adding an ``anchor_strength`` field per torn edge.
    """
    all_lens: list[float] = []
    for f in fragments:
        for e in f.get("edges", []) or []:
            if e.get("is_torn"):
                all_lens.append(float(e.get("length", 0.0)))
    if all_lens:
        ref = float(np.quantile(np.asarray(all_lens), 0.75))
    else:
        ref = 100.0
    if ref <= 0:
        ref = 100.0

    for f in fragments:
        ink = float(f.get("ink_density", 0.0))
        per_edge_strength: list[float] = []
        for e in f.get("edges", []) or []:
            if not e.get("is_torn"):
                e["anchor_strength"] = 0.0
                per_edge_strength.append(0.0)
                continue
            curv = e.get("_curvature")
            if curv is not None and len(curv) >= 4:
                arr = np.asarray(curv, dtype=np.float64)
                c_std = float(np.std(arr))
                c_rng = float(np.ptp(arr))
            else:
                c_std = c_rng = 0.0
            score = _anchor_strength(
                length_px=float(e.get("length", 0.0)),
                curv_std=c_std, curv_range=c_rng,
                ink_density=ink,
                reference_length_px=ref)
            e["anchor_strength"] = score
            per_edge_strength.append(score)
        # Cache the fragment's max-anchor torn edge score — used by the
        # candidate enumerator for cheap pair ordering.
        if per_edge_strength:
            f["max_anchor_strength"] = float(max(per_edge_strength))
        else:
            f["max_anchor_strength"] = 0.0


def build_all(fragments: list[dict], image_rgb: np.ndarray) -> list[dict]:
    """Build the canonical descriptor for every fragment, then assign
    per-torn-edge anchor strengths across the whole scene."""
    if not fragments:
        return fragments
    gray_full = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    for frag in fragments:
        build_descriptor(frag, image_rgb, gray_full=gray_full)
    _assign_anchor_strengths(fragments)
    return fragments


__all__ = ["build_descriptor", "build_all"]
