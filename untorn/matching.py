"""
untorn.matching
===============
Pair-level edge matching and rigid alignment.

Given two candidate-neighbor fragments, find the best (torn_edge_a,
torn_edge_b) pairing and return a SE(2) transform (R, t) that maps
fragment B's local coords onto fragment A's local coords, plus a
confidence score in [0, 1] (higher = better).

Pipeline for a single edge pair
-------------------------------
1.  Smith-Waterman local alignment on curvature feature strings.
    Gives a set of corresponding sample indices between the two edges.
    (Wolfson turning function; Stieber et al. 2010.)
2.  Procrustes rigid fit on those correspondences → (R_0, t_0, rms_0).
3.  ICP jitter-correction (Levenberg-Marquardt-style iterative closest
    point refinement) using full boundary-pixel sets → (R, t, rms).
    This is the "micro-adjust" step the spec asks for and is what
    eliminates sub-pixel text-line offsets that pure Procrustes leaves
    behind.
4.  Signed-distance-transform physical gate (Richter §8.5):
      - fragment B's body must not penetrate fragment A's foreground,
        and vice versa;
      - matched seam points must coincide after (R, t).
5.  Multi-component score → normalized confidence in [0, 1].
      score = geometric_fit * visual_continuity_weight

The entry points are:
    prepare_edges_and_sdt(fragments)
        -- populates frag["edges"], frag["_curv"], frag["_sdt_interior"]
    match_pair(frag_a, frag_b, image_rgb, direction_hint=None)
        -- returns best match dict or None
    icp_refine(src_pts, dst_tree, R0, t0, max_iter=15)
        -- exposed for reconstruction's post-merge refinement
"""

from __future__ import annotations

import os
import numpy as np
import cv2
from scipy.spatial import cKDTree

from . import config as cfg
from .contours import compute_curvature_string
from .appearance import seam_patch_cosine
from .text_lines import text_line_continuity


# Toggle verbose per-edge-pair rejection tracing with
#   set UNTORN_MATCH_TRACE=1   (Windows)
_MATCH_TRACE = bool(os.environ.get("UNTORN_MATCH_TRACE"))


# ══════════════════════════════════════════════════════════════════════════
#  Paper-color fingerprint (for the gate-D paper-LAB prefilter/score)
# ══════════════════════════════════════════════════════════════════════════

def _mean_paper_lab(image_rgb: np.ndarray,
                    frag: dict,
                    erode_px: int = 5,
                    ink_grayscale_max: int = 140) -> np.ndarray | None:
    """
    Median LAB of paper-only pixels inside a fragment:
      inside the mask, away from the boundary, and *not* ink.
    Returns (3,) float32 or None if no paper pixels could be isolated.
    """
    mask = frag.get("mask")
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


def attach_paper_lab_all(fragments: list[dict], image_rgb: np.ndarray) -> None:
    """Cache a per-fragment paper-color LAB fingerprint on frag['paper_lab']."""
    for frag in fragments:
        frag["paper_lab"] = _mean_paper_lab(image_rgb, frag)


def _paper_lab_delta(frag_a: dict, frag_b: dict) -> float:
    """LAB ΔE76 between two fragments' paper fingerprints. inf if missing."""
    la = frag_a.get("paper_lab")
    lb = frag_b.get("paper_lab")
    if la is None or lb is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(la) - np.asarray(lb)))


def _paper_color_score(frag_a: dict, frag_b: dict) -> float:
    """Map the LAB ΔE into a [0, 1] compatibility score (1 = identical)."""
    d = _paper_lab_delta(frag_a, frag_b)
    if not np.isfinite(d):
        return 0.5   # neutral when we can't measure
    return float(max(0.0, 1.0 - d / max(cfg.MATCH_PAPER_COLOR_DELTA_MAX, 1e-6)))


# ══════════════════════════════════════════════════════════════════════════
#  Edge extraction (torn vs factory classification via RANSAC line inlier)
# ══════════════════════════════════════════════════════════════════════════

def _ransac_line_inlier_ratio(pts: np.ndarray,
                              threshold_px: float = 1.5,
                              n_iter: int = 50) -> float:
    """Fraction of points within threshold_px of the best-fit line."""
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
    """Return the 2D unit normal at the edge midpoint pointing away from mask."""
    direction = edge_pts[-1] - edge_pts[0]
    dl = float(np.linalg.norm(direction))
    if dl < 1e-6:
        return np.array([0.0, 0.0])
    direction = direction / dl
    n1 = np.array([-direction[1], direction[0]])
    n2 = -n1
    mid = edge_pts[len(edge_pts) // 2]
    h, w = mask.shape
    for step in [5, 10, 20, 40]:
        t1 = (mid + n1 * step).astype(int)
        t2 = (mid + n2 * step).astype(int)
        in1 = (0 <= t1[0] < w and 0 <= t1[1] < h and mask[t1[1], t1[0]] > 127)
        in2 = (0 <= t2[0] < w and 0 <= t2[1] < h and mask[t2[1], t2[0]] > 127)
        if in1 and not in2:
            return n2
        if in2 and not in1:
            return n1
    return n1


def extract_edges_from_contour(contour: np.ndarray,
                                support_points: np.ndarray,
                                mask: np.ndarray,
                                min_edge_length: float = 20.0) -> list[dict]:
    """
    Break the contour polyline at support points into directed edges,
    classify each as torn/factory, attach geometric metadata.
    """
    contour_pts = contour.reshape(-1, 2).astype(np.float64)
    n_sp = len(support_points)
    sp_indices = []
    for sp in support_points:
        dists = np.linalg.norm(contour_pts - sp.astype(np.float64), axis=1)
        sp_indices.append(int(np.argmin(dists)))
    order = np.argsort(sp_indices)
    sp_sorted = [sp_indices[o] for o in order]
    sp_ids = [int(order[k]) for k in range(n_sp)]

    edges = []
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
            "pts": epts,
            "start_sp": sp_ids[k], "end_sp": sp_ids[(k + 1) % n_sp],
            "length": al, "straightness": st,
            "ransac_inlier_ratio": rr, "is_torn": is_torn,
            "midpoint": mid, "direction": d, "outward_normal": on,
        })
    return edges


# ══════════════════════════════════════════════════════════════════════════
#  Smith-Waterman for real-valued curvature strings
# ══════════════════════════════════════════════════════════════════════════

def smith_waterman_real(f1: np.ndarray, f2: np.ndarray,
                        eps1: float | None = None,
                        eps2: float | None = None,
                        w_match: float | None = None,
                        w_close: float | None = None,
                        w_far: float | None = None,
                        w_gap: float | None = None
                        ) -> tuple[float, list[int], list[int]]:
    """
    Local alignment between two curvature strings. Returns
    (max_score, aligned_indices_f1, aligned_indices_f2).
    """
    if eps1 is None: eps1 = cfg.SW_EPSILON_1
    if eps2 is None: eps2 = cfg.SW_EPSILON_2
    if w_match is None: w_match = cfg.SW_MATCH_SCORE
    if w_close is None: w_close = cfg.SW_CLOSE_PENALTY
    if w_far is None: w_far = cfg.SW_FAR_PENALTY
    if w_gap is None: w_gap = cfg.SW_GAP_PENALTY

    m, n = len(f1), len(f2)
    if m == 0 or n == 0:
        return 0.0, [], []

    f1a = np.asarray(f1, dtype=np.float64)
    f2a = np.asarray(f2, dtype=np.float64)
    diffs = np.abs(f1a[:, None] - f2a[None, :])
    W = np.where(diffs <= eps1, w_match,
         np.where(diffs <= eps2, w_close, w_far))

    M = np.zeros((m + 1, n + 1), dtype=np.float64)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = M[i - 1, j - 1] + W[i - 1, j - 1]
            up   = M[i - 1, j]     + w_gap
            left = M[i, j - 1]     + w_gap
            best_val, best_tr = 0.0, 0
            if diag > best_val: best_val, best_tr = diag, 1
            if up   > best_val: best_val, best_tr = up,   2
            if left > best_val: best_val, best_tr = left, 3
            M[i, j] = best_val
            trace[i, j] = best_tr

    max_score = float(M.max())
    if max_score <= 0:
        return 0.0, [], []

    i, j = np.unravel_index(M.argmax(), M.shape)
    align1, align2 = [], []
    while i > 0 and j > 0 and M[i, j] > 0:
        t = trace[i, j]
        if t == 1:
            align1.append(i - 1); align2.append(j - 1)
            i -= 1; j -= 1
        elif t == 2:
            i -= 1
        elif t == 3:
            j -= 1
        else:
            break
    align1.reverse()
    align2.reverse()
    return max_score, align1, align2


# ══════════════════════════════════════════════════════════════════════════
#  Procrustes rigid alignment (SVD)
# ══════════════════════════════════════════════════════════════════════════

def procrustes_rigid(src: np.ndarray, dst: np.ndarray
                     ) -> tuple[float, np.ndarray, float, np.ndarray]:
    """
    Find (R, t) minimising sum ||R src_i + t - dst_i||^2.
    Returns (angle_rad, translation[2], rms, R 2x2).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert len(src) == len(dst) >= 2

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    H = (src - src_mean).T @ (dst - dst_mean)
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, np.sign(d)])
    R = Vt.T @ D @ U.T
    t = dst_mean - R @ src_mean
    angle = float(np.arctan2(R[1, 0], R[0, 0]))
    warped = (R @ src.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((warped - dst) ** 2, axis=1))))
    return angle, t.astype(np.float64), rms, R


# ══════════════════════════════════════════════════════════════════════════
#  SE(2) affine helpers
# ══════════════════════════════════════════════════════════════════════════

def affine_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 3x3 homogeneous matrix from a 2D R and t."""
    M = np.eye(3, dtype=np.float64)
    M[:2, :2] = R
    M[:2,  2] = t
    return M


def affine_apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine to Nx2 points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ homog.T).T[:, :2]


def affine_angle_translation(M: np.ndarray) -> tuple[float, float, float]:
    """(angle_rad, tx, ty) from 3x3."""
    return (float(np.arctan2(M[1, 0], M[0, 0])),
            float(M[0, 2]), float(M[1, 2]))


# ══════════════════════════════════════════════════════════════════════════
#  ICP — iterative closest point jitter correction (spec §5b)
# ══════════════════════════════════════════════════════════════════════════

def icp_refine(src_pts: np.ndarray,
                dst_pts: np.ndarray,
                R0: np.ndarray,
                t0: np.ndarray,
                max_iter: int = None,
                tol: float = 1e-4,
                max_correspondence_dist: float = None
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Iterative closest point refinement starting from (R0, t0).

    At each iteration:
      1. Warp src_pts by the current (R, t).
      2. Find nearest neighbour in dst_pts for each warped point
         (KDTree, O(|src| log |dst|)).
      3. Drop correspondences whose distance exceeds
         max_correspondence_dist (outlier rejection — avoids pulling
         the fit toward faraway non-matching parts of the contour).
      4. Re-run Procrustes on the surviving correspondences.

    Returns the refined (R, t, rms). This is exactly the "Levenberg-
    Marquardt style" rigid-only ICP variant recommended in the spec.
    """
    if max_iter is None:
        max_iter = cfg.ICP_MAX_ITER
    if max_correspondence_dist is None:
        max_correspondence_dist = cfg.ICP_MAX_CORR_DIST_PX

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if len(src) < 3 or len(dst) < 3:
        # Not enough data to refine — return the initial estimate.
        warped = (R0 @ src.T).T + t0
        if len(src) and len(dst):
            rms = float(np.sqrt(np.mean(np.sum(
                (warped - dst[:len(warped)]) ** 2, axis=1))))
        else:
            rms = 0.0
        return R0, t0, rms

    R, t = R0.copy(), t0.copy()
    dst_tree = cKDTree(dst)
    prev_rms = None

    for _ in range(max_iter):
        warped = (R @ src.T).T + t
        d, idx = dst_tree.query(warped, k=1)
        keep = d < max_correspondence_dist
        if keep.sum() < 3:
            break
        _, t_new, rms, R_new = procrustes_rigid(src[keep], dst[idx[keep]])
        R, t = R_new, t_new
        if prev_rms is not None and abs(prev_rms - rms) < tol:
            prev_rms = rms
            break
        prev_rms = rms

    # Final RMS on all survivors (last computed)
    return R, t, float(prev_rms if prev_rms is not None else 0.0)


# ══════════════════════════════════════════════════════════════════════════
#  Appearance continuity (strip NCC) — spec: visual continuity criterion
# ══════════════════════════════════════════════════════════════════════════

def _score_edge_appearance(image_rgb: np.ndarray,
                            edge_a: dict, edge_b: dict,
                            matched_a: np.ndarray, matched_b: np.ndarray,
                            strip_width: int = 8) -> float:
    """
    Compare interior-facing pixel strips along two matched edges.
    Returns sappearance in [0, 1] (0 = identical, 1 = opposite).
    Falls back to 0.5 on uninformative strips.
    """
    h, w = image_rgb.shape[:2]

    def extract_strip(pts: np.ndarray, outward_normal: np.ndarray) -> np.ndarray:
        interior = -outward_normal
        n = float(np.linalg.norm(interior))
        if n < 1e-9:
            return np.zeros((0, strip_width, 3), dtype=np.float32)
        interior = interior / n
        rows = []
        for pt in pts:
            row = []
            for step in range(1, strip_width + 1):
                sp = pt + step * interior
                xi = int(max(0, min(w - 1, round(float(sp[0])))))
                yi = int(max(0, min(h - 1, round(float(sp[1])))))
                row.append(image_rgb[yi, xi].astype(np.float32))
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    strip_a = extract_strip(matched_a, edge_a["outward_normal"])
    strip_b = extract_strip(matched_b, edge_b["outward_normal"])
    if strip_a.size == 0 or strip_b.size == 0:
        return 0.5
    n = min(len(strip_a), len(strip_b))
    flat_a = strip_a[:n].flatten()
    flat_b = strip_b[:n].flatten()
    std_a = float(np.std(flat_a)); std_b = float(np.std(flat_b))
    if std_a < 1.0 or std_b < 1.0:
        diff = abs(float(np.mean(flat_a)) - float(np.mean(flat_b))) / 255.0
        return min(diff, 1.0)
    corr = float(np.clip(np.corrcoef(flat_a, flat_b)[0, 1], -1.0, 1.0))
    return (1.0 - corr) / 2.0


# ══════════════════════════════════════════════════════════════════════════
#  Signed-distance-transform physical gate (spec: global boundary rule)
# ══════════════════════════════════════════════════════════════════════════

def _pair_alignment_gate(frag_a: dict, frag_b: dict,
                          R: np.ndarray, t: np.ndarray,
                          matched_a: np.ndarray, matched_b: np.ndarray
                          ) -> tuple[bool, dict]:
    """
    After Procrustes/ICP, verify the alignment is physically plausible:
      1. Fragment B's contour must not penetrate fragment A's interior.
      2. Fragment A's contour must not penetrate fragment B's interior.
      3. Matched seam points must actually coincide after (R, t).
    """
    sdt_A = frag_a.get("_sdt_interior")
    sdt_B = frag_b.get("_sdt_interior")
    if sdt_A is None or sdt_B is None:
        return True, {"skipped": True}

    contour_B = frag_b["contour"].astype(np.float64).reshape(-1, 2)
    contour_A = frag_a["contour"].astype(np.float64).reshape(-1, 2)

    B_in_A = contour_B @ R.T + t.reshape(1, 2)
    hA, wA = sdt_A.shape
    xs = np.clip(B_in_A[:, 0].round().astype(int), 0, wA - 1)
    ys = np.clip(B_in_A[:, 1].round().astype(int), 0, hA - 1)
    depths_B_into_A = sdt_A[ys, xs]
    penet_B = depths_B_into_A[depths_B_into_A > 0]
    frac_B  = float(penet_B.size) / max(contour_B.shape[0], 1)
    mean_depth_B = float(penet_B.mean()) if penet_B.size else 0.0

    R_inv = R.T
    t_inv = -R_inv @ t.reshape(2)
    A_in_B = contour_A @ R_inv.T + t_inv.reshape(1, 2)
    hB, wB = sdt_B.shape
    xs = np.clip(A_in_B[:, 0].round().astype(int), 0, wB - 1)
    ys = np.clip(A_in_B[:, 1].round().astype(int), 0, hB - 1)
    depths_A_into_B = sdt_B[ys, xs]
    penet_A = depths_A_into_B[depths_A_into_B > 0]
    frac_A  = float(penet_A.size) / max(contour_A.shape[0], 1)
    mean_depth_A = float(penet_A.mean()) if penet_A.size else 0.0

    mb_warped = matched_b @ R.T + t.reshape(1, 2)
    seam_residuals = np.linalg.norm(mb_warped - matched_a, axis=1)
    median_gap = float(np.median(seam_residuals))
    p90_gap    = float(np.quantile(seam_residuals, 0.90))

    diag = {
        "overlap_frac_B_into_A": round(frac_B, 4),
        "overlap_mean_depth_B":  round(mean_depth_B, 2),
        "overlap_frac_A_into_B": round(frac_A, 4),
        "overlap_mean_depth_A":  round(mean_depth_A, 2),
        "seam_median_gap_px":    round(median_gap, 2),
        "seam_p90_gap_px":       round(p90_gap, 2),
    }

    if frac_B > cfg.SDT_OVERLAP_FRAC_THRESH \
            and mean_depth_B > cfg.SDT_OVERLAP_DEPTH_THRESH:
        diag["rejected_reason"] = "B overlaps A foreground"
        return False, diag
    if frac_A > cfg.SDT_OVERLAP_FRAC_THRESH \
            and mean_depth_A > cfg.SDT_OVERLAP_DEPTH_THRESH:
        diag["rejected_reason"] = "A overlaps B foreground"
        return False, diag
    if median_gap > cfg.SDT_SEAM_GAP_THRESH_PX:
        diag["rejected_reason"] = f"seam gap {median_gap:.1f}"
        return False, diag
    return True, diag


# ══════════════════════════════════════════════════════════════════════════
#  Full-edge physical-fit evaluator
#
#  The SW + Procrustes + ICP stack optimises a sub-arc of the tear. It
#  does NOT directly penalise uncovered stretches at the ends of the
#  tear or mild overlap away from the matched sub-arc. `evaluate_edge_fit`
#  looks at the WHOLE two polylines under (R, t) and reports:
#     - overlap_px   : sum of penetration depth (SDT) of all points on
#                      edge B that land inside A's foreground, plus the
#                      symmetric A-into-B measurement
#     - gap_px       : mean nearest-neighbour distance from sampled edge-A
#                      points to warped edge B (and vice versa), i.e. how
#                      far the two edge polylines sit apart on average
#     - coverage     : fraction of each edge's arc-length whose nearest
#                      opposite-polyline point is within COVERAGE_TOLERANCE_PX
#     - fit_cost     : scalar that goes into the match ranking. Lower is
#                      better; see cfg.FIT_W_* for the combining weights.
#  This is what makes the matcher honour the "touch the whole line of the
#  edge, no overlap, no gaps" requirement.
# ══════════════════════════════════════════════════════════════════════════

def evaluate_edge_fit(frag_a: dict, frag_b: dict,
                      edge_a: dict, edge_b: dict,
                      R: np.ndarray, t: np.ndarray) -> dict:
    """
    Full-length physical fit of edge_b (warped by R, t) against edge_a.

    Returns a dict with keys:
      fit_overlap_px, fit_overlap_frac,
      fit_gap_px,
      fit_coverage_a, fit_coverage_b, fit_coverage,
      fit_cost
    """
    pts_a = np.asarray(edge_a["pts"], dtype=np.float64)
    pts_b = np.asarray(edge_b["pts"], dtype=np.float64)
    if len(pts_a) < 2 or len(pts_b) < 2:
        return {
            "fit_overlap_px":    0.0,
            "fit_overlap_frac":  0.0,
            "fit_gap_px":        1e6,
            "fit_coverage_a":    0.0,
            "fit_coverage_b":    0.0,
            "fit_coverage":      0.0,
            "fit_cost":          1e6,
        }

    pts_b_in_a = pts_b @ R.T + t.reshape(1, 2)

    # ── Overlap: sum of SDT depth at each warped edge-B point that lands
    #    inside A's foreground. Symmetric for A-into-B.
    overlap_px = 0.0
    overlap_count = 0
    total_count = 0
    sdt_A = frag_a.get("_sdt_interior")
    if sdt_A is not None:
        hA, wA = sdt_A.shape
        xs = np.clip(pts_b_in_a[:, 0].round().astype(int), 0, wA - 1)
        ys = np.clip(pts_b_in_a[:, 1].round().astype(int), 0, hA - 1)
        depths = sdt_A[ys, xs]
        # Only penetrations more than ~1 px count as real overlap (rounding
        # noise near the seam gives a thin false positive strip otherwise).
        positive = depths[depths > 1.0]
        overlap_px += float(positive.sum())
        overlap_count += int(positive.size)
        total_count += int(depths.size)

    sdt_B = frag_b.get("_sdt_interior")
    if sdt_B is not None:
        R_inv = R.T
        t_inv = -R_inv @ t.reshape(2)
        pts_a_in_b = pts_a @ R_inv.T + t_inv.reshape(1, 2)
        hB, wB = sdt_B.shape
        xs = np.clip(pts_a_in_b[:, 0].round().astype(int), 0, wB - 1)
        ys = np.clip(pts_a_in_b[:, 1].round().astype(int), 0, hB - 1)
        depths = sdt_B[ys, xs]
        positive = depths[depths > 1.0]
        overlap_px += float(positive.sum())
        overlap_count += int(positive.size)
        total_count += int(depths.size)

    overlap_frac = (overlap_count / total_count) if total_count else 0.0

    # ── Gap & coverage: nearest-neighbour distance from each point on one
    #    polyline to the other. This is the "how tightly do the two edges
    #    touch along their whole length" metric the user asked for.
    tree_a = cKDTree(pts_a)
    d_b_to_a, _ = tree_a.query(pts_b_in_a, k=1)
    tree_b = cKDTree(pts_b_in_a)
    d_a_to_b, _ = tree_b.query(pts_a, k=1)

    gap_px = float((d_b_to_a.mean() + d_a_to_b.mean()) / 2.0)

    # Coverage: weight each sample by its local arc-length (so short
    # resampled segments don't dominate on either polyline).
    def _covered_fraction(pts: np.ndarray, dists: np.ndarray,
                           tol: float) -> float:
        if len(pts) < 2:
            return 0.0
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        # Per-sample weight: average of adjacent segment lengths (endpoints
        # get half weight). Gives total weight = arc length.
        w = np.zeros(len(pts), dtype=np.float64)
        w[:-1] += seg * 0.5
        w[1:]  += seg * 0.5
        total = float(w.sum())
        if total <= 0.0:
            return 0.0
        covered_mask = dists <= tol
        return float(w[covered_mask].sum() / total)

    tol = cfg.COVERAGE_TOLERANCE_PX
    cov_a = _covered_fraction(pts_a,        d_a_to_b, tol)
    cov_b = _covered_fraction(pts_b_in_a,   d_b_to_a, tol)
    coverage = 0.5 * (cov_a + cov_b)

    # ── Single-scalar cost ----------------------------------------------
    # Normalise overlap by edge length so a long edge doesn't get punished
    # simply for having more samples that could penetrate.
    edge_len = max(edge_a["length"], edge_b["length"], 1.0)
    w_ov   = cfg.FIT_W_OVERLAP
    w_gap  = cfg.FIT_W_GAP
    w_unc  = cfg.FIT_W_UNCOVERED
    overlap_norm = overlap_px / edge_len
    fit_cost = (w_ov  * overlap_norm
                + w_gap * gap_px
                + w_unc * (1.0 - coverage))

    return {
        "fit_overlap_px":    float(overlap_px),
        "fit_overlap_frac":  float(overlap_frac),
        "fit_gap_px":        float(gap_px),
        "fit_coverage_a":    float(cov_a),
        "fit_coverage_b":    float(cov_b),
        "fit_coverage":      float(coverage),
        "fit_cost":          float(fit_cost),
    }


# ══════════════════════════════════════════════════════════════════════════
#  Per-fragment pre-computation (edges, curvature, SDT)
# ══════════════════════════════════════════════════════════════════════════

def prepare_edges_and_sdt(fragments: list[dict],
                          image_rgb: np.ndarray | None = None) -> None:
    """
    Populate frag["edges"] with torn/factory classification, attach a
    per-edge curvature string, and cache the interior signed-distance
    transform for each fragment (used by the physical gate).

    If `image_rgb` is provided we also cache a per-fragment paper-LAB
    fingerprint (see `_mean_paper_lab`) for the matching paper-color gate.
    """
    if image_rgb is not None:
        attach_paper_lab_all(fragments, image_rgb)
    for frag in fragments:
        # Prefer the sub-pixel contour (from boundary.refine_boundary_subpixel)
        # when available; the integer contour is kept only as a fallback and
        # for legacy cv2 drawing paths.
        contour_for_edges = frag.get("contour_subpixel")
        if contour_for_edges is None or len(contour_for_edges) < 3:
            contour_for_edges = frag["contour"]
        frag["edges"] = extract_edges_from_contour(
            contour_for_edges, frag["support_points"],
            frag["mask"], min_edge_length=15.0)
        for e in frag["edges"]:
            if e["is_torn"]:
                resamp, curv = compute_curvature_string(e["pts"])
                e["_resampled"] = resamp
                e["_curvature"] = curv

        # Interior SDT: O(H·W) but done once.
        fg = (frag["mask"] > 127).astype(np.uint8) * 255
        frag["_sdt_interior"] = cv2.distanceTransform(
            fg, cv2.DIST_L2, 3).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  Directional gate — skip edge pairs whose outward normals don't face
# ══════════════════════════════════════════════════════════════════════════

def _edges_face_each_other(edge_a: dict, edge_b: dict,
                            centroid_offset: np.ndarray) -> bool:
    """
    An edge faces a neighbor only if its outward normal has positive dot
    product with the offset vector from A's centroid to B's centroid.
    Mirror check for B. This kills the O(k^2) edge pairings where both
    edges are on the wrong sides of the two fragments.
    """
    off = centroid_offset.astype(np.float64)
    n = float(np.linalg.norm(off))
    if n < 1e-6:
        return True
    off = off / n
    dot_a = float(edge_a["outward_normal"] @ off)
    dot_b = float(edge_b["outward_normal"] @ (-off))
    return dot_a > cfg.FACING_COSINE_MIN and dot_b > cfg.FACING_COSINE_MIN


# ══════════════════════════════════════════════════════════════════════════
#  Single edge-pair matcher (SW + Procrustes + ICP + SDT gate)
# ══════════════════════════════════════════════════════════════════════════

def _match_edge_pair(edge_a: dict, edge_b: dict,
                      image_rgb: np.ndarray | None,
                      tag: str = "",
                      frag_a: dict | None = None,
                      frag_b: dict | None = None) -> dict | None:
    """Return a partial match dict (missing frag_i/j/edge_i/j) or None.

    If `frag_a` and `frag_b` are supplied the function additionally attaches
    a full-edge physical-fit report (see `evaluate_edge_fit`) to the
    returned dict; this enables ranking candidates by fit_cost rather than
    by raw SW-derived stotal.
    """

    def _trace(reason, **extra):
        if _MATCH_TRACE:
            xs = " ".join(f"{k}={v}" for k, v in extra.items())
            print(f"    [trace] {tag}: reject {reason}  {xs}")

    if edge_a["length"] < cfg.MIN_TORN_EDGE_PX or \
       edge_b["length"] < cfg.MIN_TORN_EDGE_PX:
        _trace("short_edge")
        return None

    resamp_a = edge_a.get("_resampled")
    curv_a   = edge_a.get("_curvature")
    resamp_b = edge_b.get("_resampled")
    curv_b   = edge_b.get("_curvature")
    if resamp_a is None or curv_a is None or resamp_b is None or curv_b is None:
        resamp_a, curv_a = compute_curvature_string(edge_a["pts"])
        resamp_b, curv_b = compute_curvature_string(edge_b["pts"])

    if len(curv_a) < cfg.SW_MIN_ALIGNED or len(curv_b) < cfg.SW_MIN_ALIGNED:
        _trace("curv_too_short"); return None
    if np.std(curv_a) < cfg.CURV_MIN_STD or np.std(curv_b) < cfg.CURV_MIN_STD:
        _trace("low_curv_variance"); return None

    # Complementary orientation (primary case for torn edges)
    curv_b_comp = -curv_b[::-1]
    score_comp, ia_c, ib_c = smith_waterman_real(curv_a, curv_b_comp)
    score_dir,  ia_d, ib_d = smith_waterman_real(curv_a, curv_b)

    if score_comp >= score_dir:
        sw_score = score_comp; idx_a = ia_c; idx_b_sw = ib_c
        orientation = "complementary"
        n_b = len(curv_b)
        idx_b_orig = [n_b - 1 - k for k in idx_b_sw]
    else:
        if cfg.MATCH_REJECT_DIRECT:
            _trace("direct_forbidden"); return None
        sw_score = score_dir; idx_a = ia_d
        idx_b_orig = idx_b_sw = ib_d
        orientation = "direct"

    if sw_score < cfg.SW_MIN_SCORE or len(idx_a) < cfg.SW_MIN_ALIGNED:
        _trace("sw_weak"); return None

    # Map curvature indices → resampled point indices
    offset = cfg.CURV_SMOOTH_WINDOW // 2 + 1
    pt_idx_a = np.clip(np.array(idx_a)      + offset, 0, len(resamp_a) - 1)
    pt_idx_b = np.clip(np.array(idx_b_orig) + offset, 0, len(resamp_b) - 1)
    matched_a = resamp_a[pt_idx_a]
    matched_b = resamp_b[pt_idx_b]
    if len(matched_a) < 3:
        _trace("few_matched_pts"); return None

    # Multi-seed Procrustes: SW may have picked a wrong sub-arc, so we fit
    # from several overlapping windows into the matched correspondences and
    # keep the seed with the lowest RMS. This defeats the "locked to wrong
    # rotation because SW latched onto the wrong end" local minimum that
    # single-start Procrustes falls into under edge noise.
    N = len(matched_a)
    n_seeds = max(1, int(cfg.MATCH_PROCRUSTES_SEEDS))
    if N < 4 or n_seeds <= 1:
        seed_windows = [(0, N)]
    else:
        # Overlapping sliding windows of ~2/3 N. Equivalent to the full arc
        # when n_seeds == 1 and to {head, centre, tail} for n_seeds == 3.
        win = max(3, (2 * N + n_seeds) // (n_seeds + 1))
        starts = np.linspace(0, max(0, N - win), n_seeds).astype(int)
        seed_windows = [(int(s), int(min(N, s + win))) for s in starts]
        seed_windows.append((0, N))   # always include the full arc as backup

    best_seed = None   # (rms, angle, t, R, window)
    for (ws, we) in seed_windows:
        if we - ws < 3:
            continue
        ang_s, t_s, rms_s, R_s = procrustes_rigid(
            matched_b[ws:we], matched_a[ws:we])
        if not np.isfinite(rms_s):
            continue
        if abs(np.degrees(ang_s)) > cfg.RECON_MAX_ROTATION_DEG:
            continue
        if best_seed is None or rms_s < best_seed[0]:
            best_seed = (rms_s, ang_s, t_s, R_s, (ws, we))

    if best_seed is None:
        _trace("procrustes_no_seed"); return None
    rms0, angle0, t0, R0, seed_win = best_seed
    if rms0 > cfg.MATCH_MAX_RMS:
        _trace("rms_high", rms=f"{rms0:.2f}"); return None

    # ── Two-phase ICP jitter correction on full edge point sets ─────
    # We refine against the full edge-point polyline of A (not just the SW
    # correspondences) so that text-line edges along the tear get snapped
    # to their nearest counterpart even if SW didn't sample them. The
    # COARSE pass uses a wide correspondence tolerance to PULL apart-
    # sitting polylines together, then the FINE pass tightens the fit.
    # This is what makes a "drifted-but-real" pair actually touch after
    # alignment.
    R_coarse, t_coarse, rms_coarse = icp_refine(
        src_pts=edge_b["pts"],
        dst_pts=edge_a["pts"],
        R0=R0, t0=t0,
        max_iter=cfg.ICP_MAX_ITER,
        max_correspondence_dist=cfg.ICP_COARSE_DIST_PX,
    )
    R, t, rms = icp_refine(
        src_pts=edge_b["pts"],
        dst_pts=edge_a["pts"],
        R0=R_coarse, t0=t_coarse,
        max_iter=cfg.ICP_MAX_ITER,
        max_correspondence_dist=cfg.ICP_MAX_CORR_DIST_PX,
    )

    # ICP may drift into a bad basin if src/dst are too dissimilar;
    # prefer the initial Procrustes estimate if ICP made things worse
    # (higher rms) OR if it rotated far away from the initial estimate.
    angle_icp = float(np.arctan2(R[1, 0], R[0, 0]))
    drift_deg = abs(np.degrees(angle_icp - angle0))
    # Relaxed drift cap: with multi-seed Procrustes the initial pose is much
    # less likely to be wildly off, so we can afford ICP to move further
    # looking for the real basin. The SDT physical gate downstream catches
    # the rare case where ICP rotates into an interpenetrating minimum.
    drift_cap = max(cfg.ICP_MAX_DRIFT_DEG, cfg.MATCH_ICP_DRIFT_DEG)
    if rms > rms0 + 0.5 or drift_deg > drift_cap:
        R, t, rms = R0, t0, rms0
    angle = float(np.arctan2(R[1, 0], R[0, 0]))

    # Post-ICP rotation bound — catches cases where ICP found a better
    # local minimum at an implausible orientation.
    if abs(np.degrees(angle)) > cfg.RECON_MAX_ROTATION_DEG:
        _trace("post_icp_rotation_large", ang=f"{np.degrees(angle):.1f}")
        return None

    # Multi-component score (lower = better)
    arc_a = float(np.sum(np.linalg.norm(np.diff(matched_a, axis=0), axis=1)))
    arc_b = float(np.sum(np.linalg.norm(np.diff(matched_b, axis=0), axis=1)))
    avg_arc = max((arc_a + arc_b) / 2.0, 1.0)
    sarea = min(rms / avg_arc * 5.0, 1.0)
    slen  = 1.0 - min(len(idx_a) / max(min(len(curv_a), len(curv_b)), 1), 1.0)

    sub_a = curv_a[np.array(idx_a)]
    if orientation == "complementary":
        sub_b = curv_b_comp[np.array(idx_b_sw)]
    else:
        sub_b = curv_b[np.array(idx_b_orig)]
    if len(sub_a) > 2 and np.std(sub_a) > 1e-8 and np.std(sub_b) > 1e-8:
        scorr = (1.0 - float(np.corrcoef(sub_a, sub_b)[0, 1])) / 2.0
    else:
        scorr = 0.5

    if image_rgb is not None:
        sappearance = _score_edge_appearance(
            image_rgb, edge_a, edge_b, matched_a, matched_b)
    else:
        sappearance = 0.5

    w_app = cfg.MATCH_APPEARANCE_WEIGHT
    stotal = sarea + slen + scorr + w_app * sappearance  # in [0, 3+w_app]

    # Legacy geometry-only confidence (kept for trace/diagnostics).
    geom_conf = max(0.0, min(1.0, 1.0 - stotal / cfg.CONFIDENCE_STOTAL_SPAN))

    # ── Gate C: DINOv2 seam appearance ────────────────────────────────────
    # Sample a few patches on each side of the proposed seam in canvas space
    # and cosine-compare the feature vectors. The caller only populates
    # frag["dinov2"] when the DINOv2 extractor has been run earlier in the
    # pipeline; otherwise we neutralise the gate.
    dinov2_score = 0.5
    dinov2_n     = 0
    if (frag_a is not None and frag_b is not None and
            frag_a.get("dinov2") is not None and frag_b.get("dinov2") is not None):
        M_ba = affine_from_Rt(R, t)           # B -> A's frame
        I    = np.eye(3, dtype=np.float64)    # A stays in its own frame
        seam_mid = 0.5 * (matched_a[len(matched_a) // 2] +
                          affine_apply(M_ba, matched_b[len(matched_b) // 2:
                                                       len(matched_b) // 2 + 1])[0])
        # Normal along the seam: perpendicular to local arc tangent.
        if len(matched_a) >= 2:
            tgt = matched_a[-1] - matched_a[0]
            tgt_n = np.linalg.norm(tgt)
            if tgt_n > 1e-6:
                tgt = tgt / tgt_n
                seam_normal = np.array([-tgt[1], tgt[0]], dtype=np.float64)
            else:
                seam_normal = np.array([1.0, 0.0])
        else:
            seam_normal = np.array([1.0, 0.0])
        # Point normal from A's side to B's side: the seam separates A's
        # interior from B's; after pose, B's centroid - A's centroid in A's
        # frame tells us which way is "toward B".
        try:
            b_cen_in_a = affine_apply(M_ba, np.asarray(frag_b["centroid"])
                                               .reshape(1, 2))[0]
            a_cen      = np.asarray(frag_a["centroid"], dtype=np.float64)
            toward_b   = b_cen_in_a - a_cen
            if np.dot(seam_normal, toward_b) < 0:
                seam_normal = -seam_normal
        except Exception:
            pass
        dinov2_score, dinov2_n = seam_patch_cosine(
            frag_a, frag_b, I, M_ba, seam_mid, seam_normal)
        if dinov2_n >= 2 and dinov2_score < cfg.MATCH_APPEARANCE_COS_MIN:
            _trace("dinov2_low", cos=f"{dinov2_score:.2f}")
            return None

    # ── Gate D: text-line continuity ──────────────────────────────────────
    text_score = 0.0
    text_expected = 0
    if (frag_a is not None and frag_b is not None and
            frag_a.get("text_lines") is not None and
            frag_b.get("text_lines") is not None):
        M_ba = affine_from_Rt(R, t)
        I    = np.eye(3, dtype=np.float64)
        # Seam centre in A's frame (reuse matched midpoint).
        seam_mid = matched_a[len(matched_a) // 2]
        tgt = matched_a[-1] - matched_a[0]
        tgt_n = np.linalg.norm(tgt)
        if tgt_n > 1e-6:
            tgt = tgt / tgt_n
            seam_normal = np.array([-tgt[1], tgt[0]], dtype=np.float64)
        else:
            seam_normal = np.array([1.0, 0.0])
        text_score, text_expected = text_line_continuity(
            frag_a["text_lines"], frag_b["text_lines"], I, M_ba,
            seam_mid, seam_normal,
            max_y_disc_px=cfg.TEXT_LINE_MAX_Y_DISC_PX,
            max_angle_disc_deg=cfg.TEXT_LINE_MAX_ANGLE_DISC_DEG,
            search_radius_px=cfg.TEXT_LINE_SEAM_RADIUS_PX)
        if (text_expected >= cfg.MATCH_TEXT_LINE_MIN_EXPECT and
                text_score < cfg.MATCH_TEXT_LINE_MIN_CONT):
            _trace("text_discontinuity",
                   score=f"{text_score:.2f}", expected=text_expected)
            return None

    # ── Paper-color similarity (soft score, no hard gate here) ────────────
    paper_score = _paper_color_score(frag_a, frag_b) \
        if (frag_a is not None and frag_b is not None) else 0.5

    # ── Combined confidence — weighted, bounded [0, 1] ────────────────────
    # Each sub-score is already in [0, 1]:
    #   geom_conf     : lower stotal -> higher value
    #   dinov2_score  : 0.5 * (cos + 1) in [0, 1]
    #   strip_ncc     : 1 - sappearance (sappearance is a cost in [0, 1])
    #   text_score    : n_continued / n_expected in [0, 1]
    #   paper_score   : 1 - ΔE/ΔE_max
    strip_ncc_score = float(max(0.0, 1.0 - sappearance))
    # Text-line weight shifts onto geometry when the pair has no text signal
    # (otherwise fragments with no writing near the seam would always be
    # penalised). Expected == 0 means the gate didn't apply.
    w_geom = cfg.CONF_W_GEOMETRY
    w_app_ = cfg.CONF_W_APPEARANCE
    w_str  = cfg.CONF_W_STRIP_NCC
    w_txt  = cfg.CONF_W_TEXT_LINE
    w_pap  = cfg.CONF_W_PAPER_COLOR
    if text_expected < cfg.MATCH_TEXT_LINE_MIN_EXPECT:
        w_geom += w_txt
        w_txt = 0.0
    conf = (w_geom * geom_conf +
            w_app_ * dinov2_score +
            w_str  * strip_ncc_score +
            w_txt  * text_score +
            w_pap  * paper_score)
    conf = float(max(0.0, min(1.0, conf)))

    result = {
        "orientation": orientation,
        "sw_score":    float(sw_score),
        "n_aligned":   int(len(idx_a)),
        "angle":       float(angle),
        "translation": t.astype(np.float64),
        "R":           R.astype(np.float64),
        "rms":         float(rms),
        "rms_procrustes": float(rms0),
        "sarea":       float(sarea),
        "slen":        float(slen),
        "scorr":       float(scorr),
        "sappearance": float(sappearance),
        "stotal":      float(stotal),
        "geom_conf":   float(geom_conf),
        "dinov2_score": float(dinov2_score),
        "dinov2_n":    int(dinov2_n),
        "strip_ncc_score": float(strip_ncc_score),
        "text_score":  float(text_score),
        "text_expected": int(text_expected),
        "paper_score": float(paper_score),
        "confidence":  float(conf),
        "matched_a":   np.asarray(matched_a, dtype=np.float64),
        "matched_b":   np.asarray(matched_b, dtype=np.float64),
    }

    # ── Full-edge physical-fit (overlap/gap/coverage) under final (R, t).
    # This is the "touch the whole line, no overlap, no blanks" metric
    # used to rank candidates in match_pair below.
    if frag_a is not None and frag_b is not None:
        fit = evaluate_edge_fit(frag_a, frag_b, edge_a, edge_b, R, t)
        result.update(fit)

    return result


# ══════════════════════════════════════════════════════════════════════════
#  Top-level entry: best match between two candidate-neighbor fragments
# ══════════════════════════════════════════════════════════════════════════

def match_pair(frag_a: dict, frag_b: dict,
                image_rgb: np.ndarray | None = None,
                direction_hint: str | None = None
                ) -> dict | None:
    """
    Find the best (torn_edge_a, torn_edge_b) pairing between the two
    fragments.  Prunes candidates by:
      * both edges must be torn (factory edges never meet another);
      * outward normals must face each other along the centroid offset
        (spec: "immediate horizontal/vertical neighbor" implies the
        meeting edges face each other in scan coordinates);
      * SDT-gate rejects right-curvature wrong-edge false positives.

    Returns the best match (lowest stotal) or None.
    """
    torn_a = [(k, e) for k, e in enumerate(frag_a["edges"]) if e["is_torn"]]
    torn_b = [(k, e) for k, e in enumerate(frag_b["edges"]) if e["is_torn"]]
    if not torn_a or not torn_b:
        return None

    centroid_off = (np.asarray(frag_b["centroid"], dtype=np.float64)
                    - np.asarray(frag_a["centroid"], dtype=np.float64))
    best: dict | None = None
    best_cost = float("inf")

    # Tag used for verbose matching traces
    id_a = frag_a["id"]; id_b = frag_b["id"]

    for ki, ea in torn_a:
        for kj, eb in torn_b:
            # Directional facing gate (cheap, removes ~75% of pairings)
            if not _edges_face_each_other(ea, eb, centroid_off):
                continue

            tag = f"frag{id_a}.e{ki} <> frag{id_b}.e{kj}"
            result = _match_edge_pair(
                ea, eb, image_rgb, tag=tag,
                frag_a=frag_a, frag_b=frag_b,
            )
            if result is None:
                continue

            # Physical gate (SDT overlap + seam gap at matched points)
            ok, gate_diag = _pair_alignment_gate(
                frag_a, frag_b, result["R"], result["translation"],
                result["matched_a"], result["matched_b"])
            if not ok:
                continue
            result["alignment_gate"] = gate_diag

            # Structural upper bound: if overlap/gap/coverage put this
            # candidate above MAX_ATTACH_COST there's no point keeping it
            # — the SW score can be high yet the two polylines still sit
            # apart from each other (or through each other).
            fit_cost = result.get("fit_cost", float("inf"))
            if fit_cost > cfg.MAX_ATTACH_COST:
                if _MATCH_TRACE:
                    print(f"    [trace] {tag}: reject fit_cost_high  "
                          f"cost={fit_cost:.1f}")
                continue

            # Rank by physical fit — this is what enforces the user's
            # "touch the whole edge, no overlap, no blank gaps" priority.
            # stotal is still kept on the dict as a secondary diagnostic.
            if fit_cost < best_cost:
                best_cost = fit_cost
                best = result
                best["edge_i"] = ki
                best["edge_j"] = kj

    if best is None:
        return None

    best["frag_i"] = frag_a["id"]
    best["frag_j"] = frag_b["id"]
    best["direction_hint"] = direction_hint
    return best


__all__ = [
    "extract_edges_from_contour",
    "smith_waterman_real",
    "procrustes_rigid",
    "icp_refine",
    "affine_from_Rt",
    "affine_apply",
    "affine_angle_translation",
    "prepare_edges_and_sdt",
    "evaluate_edge_fit",
    "match_pair",
]
