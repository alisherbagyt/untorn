"""untorn.matching — pair-level edge matching and rigid alignment.

Phase 3 inner core. Given two fragment dicts (each in its own canvas),
return the best (R, t, confidence, fit_cost) that aligns them — or
``None`` when no torn-edge pair survives the gate cascade.

Design rules (engine-rebuild April 2026)
----------------------------------------
* **No rotation cap.** Curvature matching + Procrustes + ICP recover the
  full SE(2). Whatever rotation comes out, comes out. The SDT physical
  gate plus the full-edge ``fit_cost`` ceiling reject genuinely bad pose
  estimates regardless of angle.
* **Both orientations always tried.** Smith-Waterman runs on both the
  complementary curvature (``-curv_b[::-1]``) and the direct curvature.
  The higher score wins. Within a 5% margin, complementary wins (it is
  the physical prior for torn edges).
* **No silent gates.** Every gate logs its decision through the
  ``untorn.engine.matching`` logger. ``match_pair`` returns either an
  accepted dict (with full diagnostics in ``gates``) or ``None`` when
  every candidate edge pair was rejected.
* **Single confidence formula.** ``confidence = 0.7 * geom + 0.3 *
  appearance``. ``geom`` is derived from the *physical* fit cost
  (overlap + gap + uncovered), not from a sum of partial penalties.
  ``appearance`` is the mean of whichever appearance signals are
  available (paper-LAB, DINOv2 cosine, strip-NCC).
* **Lazy SDT.** If a fragment is missing ``_sdt_interior``, this module
  computes it on the fly rather than silently passing a "skipped" gate.

Public API
----------
    prepare_edges_and_sdt(fragments, image_rgb=None) -> None
    match_pair(frag_a, frag_b, image_rgb=None, *, hint=None) -> dict | None
    evaluate_edge_fit(frag_a, frag_b, edge_a, edge_b, R, t) -> dict
    procrustes_rigid(src, dst) -> (angle, t, rms, R)
    icp_refine(src_pts, dst_pts, R0, t0, max_iter=None, tol=1e-4,
                max_correspondence_dist=None) -> (R, t, rms)
    affine_from_Rt(R, t) -> 3x3
    affine_apply(M, pts) -> Nx2
    affine_angle_translation(M) -> (angle, tx, ty)
    smith_waterman_real(f1, f2, ...) -> (score, align1, align2)
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import config as cfg
from .contours import compute_curvature_string

logger = logging.getLogger("untorn.engine.matching")


# ══════════════════════════════════════════════════════════════════════════
#  Smith-Waterman local alignment on real-valued curvature strings
#  (kept verbatim from the prior implementation — proven correct)
# ══════════════════════════════════════════════════════════════════════════

def smith_waterman_real(f1: np.ndarray, f2: np.ndarray,
                        eps1: float | None = None,
                        eps2: float | None = None,
                        w_match: float | None = None,
                        w_close: float | None = None,
                        w_far: float | None = None,
                        w_gap: float | None = None
                        ) -> tuple[float, list[int], list[int]]:
    """Local alignment between two curvature strings."""
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
    """Find (R, t) minimising ``sum ||R src_i + t - dst_i||^2``.

    Returns ``(angle_rad, translation[2], rms, R[2x2])``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError(
            f"procrustes_rigid needs >=2 paired points; got "
            f"src={len(src)}, dst={len(dst)}")

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
    """``(angle_rad, tx, ty)`` from a 3x3 affine."""
    return (float(np.arctan2(M[1, 0], M[0, 0])),
            float(M[0, 2]), float(M[1, 2]))


# ══════════════════════════════════════════════════════════════════════════
#  ICP — iterative closest point with no rotation cap
# ══════════════════════════════════════════════════════════════════════════

def icp_refine(src_pts: np.ndarray,
                dst_pts: np.ndarray,
                R0: np.ndarray,
                t0: np.ndarray,
                max_iter: int | None = None,
                tol: float = 1e-4,
                max_correspondence_dist: float | None = None
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """Iterative closest point starting from (R0, t0).

    No drift cap — the caller compares the final RMS to the seed RMS and
    reverts if ICP made things worse.
    """
    if max_iter is None:
        max_iter = cfg.ICP_MAX_ITER
    if max_correspondence_dist is None:
        max_correspondence_dist = cfg.ICP_MAX_CORR_DIST_PX

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if len(src) < 3 or len(dst) < 3:
        warped = (R0 @ src.T).T + t0 if len(src) else src
        if len(src) and len(dst):
            n = min(len(warped), len(dst))
            rms = float(np.sqrt(np.mean(np.sum(
                (warped[:n] - dst[:n]) ** 2, axis=1))))
        else:
            rms = 0.0
        return R0.copy(), np.asarray(t0, dtype=np.float64).copy(), rms

    R = R0.copy()
    t = np.asarray(t0, dtype=np.float64).copy()
    dst_tree = cKDTree(dst)
    prev_rms = None

    for _ in range(int(max_iter)):
        warped = (R @ src.T).T + t
        d, idx = dst_tree.query(warped, k=1)
        keep = d < max_correspondence_dist
        if int(keep.sum()) < 3:
            break
        _, t_new, rms, R_new = procrustes_rigid(src[keep], dst[idx[keep]])
        R, t = R_new, t_new
        if prev_rms is not None and abs(prev_rms - rms) < tol:
            prev_rms = rms
            break
        prev_rms = rms

    return R, t, float(prev_rms if prev_rms is not None else 0.0)


# ══════════════════════════════════════════════════════════════════════════
#  Outward-normal facing gate
# ══════════════════════════════════════════════════════════════════════════

def _edges_face_each_other(edge_a: dict, edge_b: dict,
                            centroid_offset: np.ndarray) -> bool:
    """Both outward normals must point along the centroid offset."""
    off = np.asarray(centroid_offset, dtype=np.float64)
    n = float(np.linalg.norm(off))
    if n < 1e-6:
        return True
    off = off / n
    dot_a = float(np.asarray(edge_a["outward_normal"]) @ off)
    dot_b = float(np.asarray(edge_b["outward_normal"]) @ (-off))
    return dot_a > cfg.FACING_COSINE_MIN and dot_b > cfg.FACING_COSINE_MIN


# ══════════════════════════════════════════════════════════════════════════
#  Strip-NCC dissimilarity (interior-facing pixel strips on either side)
#  Cost in [0, 1]: 0 means strips are perfectly correlated, 1 = anti-corr.
# ══════════════════════════════════════════════════════════════════════════

def _strip_ncc_dissimilarity(image_rgb: np.ndarray,
                              edge_a: dict, edge_b: dict,
                              matched_a: np.ndarray,
                              matched_b: np.ndarray,
                              strip_width: int = 8) -> float:
    if (matched_a is None or matched_b is None or
            len(matched_a) == 0 or len(matched_b) == 0):
        return 0.5

    def _strip(pts: np.ndarray, outward_normal: np.ndarray) -> np.ndarray:
        interior = -np.asarray(outward_normal, dtype=np.float32)
        n = float(np.linalg.norm(interior))
        if n < 1e-9:
            return np.zeros((0, strip_width, 3), dtype=np.float32)
        interior = interior / n
        steps = np.arange(1, strip_width + 1, dtype=np.float32)
        offsets = steps[None, :, None] * interior[None, None, :]
        coords = pts.astype(np.float32)[:, None, :] + offsets
        map_x = np.ascontiguousarray(coords[..., 0], dtype=np.float32)
        map_y = np.ascontiguousarray(coords[..., 1], dtype=np.float32)
        sampled = cv2.remap(image_rgb, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        return sampled.astype(np.float32)

    strip_a = _strip(np.asarray(matched_a), edge_a["outward_normal"])
    strip_b = _strip(np.asarray(matched_b), edge_b["outward_normal"])
    if strip_a.size == 0 or strip_b.size == 0:
        return 0.5
    n = min(len(strip_a), len(strip_b))
    flat_a = strip_a[:n].reshape(-1)
    flat_b = strip_b[:n].reshape(-1)
    std_a = float(np.std(flat_a))
    std_b = float(np.std(flat_b))
    if std_a < 1.0 or std_b < 1.0:
        diff = abs(float(np.mean(flat_a)) - float(np.mean(flat_b))) / 255.0
        return float(min(diff, 1.0))
    corr = float(np.clip(np.corrcoef(flat_a, flat_b)[0, 1], -1.0, 1.0))
    return float((1.0 - corr) / 2.0)


# ══════════════════════════════════════════════════════════════════════════
#  Paper-LAB compatibility (a soft score in [0, 1])
# ══════════════════════════════════════════════════════════════════════════

_PAPER_DELTA_MAX = 35.0   # one constant for the whole engine


def _paper_lab_score(frag_a: dict, frag_b: dict) -> float:
    la = frag_a.get("paper_lab")
    lb = frag_b.get("paper_lab")
    if la is None or lb is None:
        return 0.5
    d = float(np.linalg.norm(np.asarray(la) - np.asarray(lb)))
    return float(max(0.0, 1.0 - d / _PAPER_DELTA_MAX))


# ══════════════════════════════════════════════════════════════════════════
#  SDT physical gate (lazy: computes SDT if missing rather than skipping)
# ══════════════════════════════════════════════════════════════════════════

def _ensure_sdt(frag: dict) -> np.ndarray:
    sdt = frag.get("_sdt_interior")
    if sdt is not None:
        return sdt
    fg = (frag["mask"] > 127).astype(np.uint8) * 255
    sdt = cv2.distanceTransform(fg, cv2.DIST_L2, 3).astype(np.float32)
    frag["_sdt_interior"] = sdt
    logger.info("computed missing SDT for fragment %s on the fly",
                frag.get("id"))
    return sdt


def _sdt_physical_gate(frag_a: dict, frag_b: dict,
                        R: np.ndarray, t: np.ndarray,
                        matched_a: np.ndarray, matched_b: np.ndarray
                        ) -> tuple[bool, dict]:
    """Gate per Richter §8.5: B's body must not penetrate A's interior, and
    matched seam points must coincide after (R, t).

    Returns ``(ok, diag_dict)`` where diag includes the rejection reason on
    failure (key ``rejected_reason``).
    """
    sdt_A = _ensure_sdt(frag_a)
    sdt_B = _ensure_sdt(frag_b)

    contour_B = np.asarray(frag_b["contour"], dtype=np.float64).reshape(-1, 2)
    contour_A = np.asarray(frag_a["contour"], dtype=np.float64).reshape(-1, 2)

    B_in_A = contour_B @ R.T + t.reshape(1, 2)
    hA, wA = sdt_A.shape
    xs = np.clip(B_in_A[:, 0].round().astype(int), 0, wA - 1)
    ys = np.clip(B_in_A[:, 1].round().astype(int), 0, hA - 1)
    depths_B_into_A = sdt_A[ys, xs]
    penet_B = depths_B_into_A[depths_B_into_A > 0]
    frac_B = float(penet_B.size) / max(contour_B.shape[0], 1)
    mean_depth_B = float(penet_B.mean()) if penet_B.size else 0.0

    R_inv = R.T
    t_inv = -R_inv @ t.reshape(2)
    A_in_B = contour_A @ R_inv.T + t_inv.reshape(1, 2)
    hB, wB = sdt_B.shape
    xs = np.clip(A_in_B[:, 0].round().astype(int), 0, wB - 1)
    ys = np.clip(A_in_B[:, 1].round().astype(int), 0, hB - 1)
    depths_A_into_B = sdt_B[ys, xs]
    penet_A = depths_A_into_B[depths_A_into_B > 0]
    frac_A = float(penet_A.size) / max(contour_A.shape[0], 1)
    mean_depth_A = float(penet_A.mean()) if penet_A.size else 0.0

    mb_warped = matched_b @ R.T + t.reshape(1, 2)
    seam_residuals = np.linalg.norm(mb_warped - matched_a, axis=1)
    median_gap = float(np.median(seam_residuals))
    p90_gap = float(np.quantile(seam_residuals, 0.90))

    diag = {
        "overlap_frac_B_into_A": round(frac_B, 4),
        "overlap_mean_depth_B":  round(mean_depth_B, 2),
        "overlap_frac_A_into_B": round(frac_A, 4),
        "overlap_mean_depth_A":  round(mean_depth_A, 2),
        "seam_median_gap_px":    round(median_gap, 2),
        "seam_p90_gap_px":       round(p90_gap, 2),
    }

    if (frac_B > cfg.SDT_OVERLAP_FRAC_THRESH and
            mean_depth_B > cfg.SDT_OVERLAP_DEPTH_THRESH):
        diag["rejected_reason"] = "B_overlaps_A"
        return False, diag
    if (frac_A > cfg.SDT_OVERLAP_FRAC_THRESH and
            mean_depth_A > cfg.SDT_OVERLAP_DEPTH_THRESH):
        diag["rejected_reason"] = "A_overlaps_B"
        return False, diag
    if median_gap > cfg.SDT_SEAM_GAP_THRESH_PX:
        diag["rejected_reason"] = f"seam_gap_high({median_gap:.1f})"
        return False, diag
    return True, diag


# ══════════════════════════════════════════════════════════════════════════
#  Full-edge physical fit evaluator (kept verbatim — the truth metric)
# ══════════════════════════════════════════════════════════════════════════

def evaluate_edge_fit(frag_a: dict, frag_b: dict,
                      edge_a: dict, edge_b: dict,
                      R: np.ndarray, t: np.ndarray) -> dict:
    """Full-length physical fit of edge_b (warped by R, t) against edge_a.

    Returns a dict with ``fit_overlap_px, fit_overlap_frac, fit_gap_px,
    fit_coverage_a, fit_coverage_b, fit_coverage, fit_cost``.
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

    overlap_px = 0.0
    overlap_count = 0
    total_count = 0
    sdt_A = frag_a.get("_sdt_interior")
    if sdt_A is not None:
        hA, wA = sdt_A.shape
        xs = np.clip(pts_b_in_a[:, 0].round().astype(int), 0, wA - 1)
        ys = np.clip(pts_b_in_a[:, 1].round().astype(int), 0, hA - 1)
        depths = sdt_A[ys, xs]
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

    tree_a = cKDTree(pts_a)
    d_b_to_a, _ = tree_a.query(pts_b_in_a, k=1)
    tree_b = cKDTree(pts_b_in_a)
    d_a_to_b, _ = tree_b.query(pts_a, k=1)
    gap_px = float((d_b_to_a.mean() + d_a_to_b.mean()) / 2.0)

    def _covered_fraction(pts: np.ndarray, dists: np.ndarray,
                           tol: float) -> float:
        if len(pts) < 2:
            return 0.0
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        w = np.zeros(len(pts), dtype=np.float64)
        w[:-1] += seg * 0.5
        w[1:]  += seg * 0.5
        total = float(w.sum())
        if total <= 0.0:
            return 0.0
        return float(w[dists <= tol].sum() / total)

    tol = cfg.COVERAGE_TOLERANCE_PX
    cov_a = _covered_fraction(pts_a, d_a_to_b, tol)
    cov_b = _covered_fraction(pts_b_in_a, d_b_to_a, tol)
    coverage = 0.5 * (cov_a + cov_b)

    edge_len = max(edge_a["length"], edge_b["length"], 1.0)
    overlap_norm = overlap_px / edge_len
    fit_cost = (cfg.FIT_W_OVERLAP * overlap_norm
                + cfg.FIT_W_GAP * gap_px
                + cfg.FIT_W_UNCOVERED * (1.0 - coverage))

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
#  Per-fragment pre-computation (top-up: only fills what's missing)
# ══════════════════════════════════════════════════════════════════════════

def prepare_edges_and_sdt(fragments: list[dict],
                          image_rgb: np.ndarray | None = None) -> None:
    """Top-up pass for fragments that didn't go through fragment_io.build_all.

    fragment_io is the canonical ingest path. This function only fills in
    fields that are missing, so it stays safe to call from legacy callers
    (unit tests, partial benchmarks).
    """
    for frag in fragments:
        if not frag.get("edges"):
            from . import fragment_io as fio
            contour_for_edges = frag.get("contour_subpixel")
            if contour_for_edges is None or len(contour_for_edges) < 3:
                contour_for_edges = frag["contour"]
            sp = frag.get("support_points")
            if sp is None:
                from .contours import extract_support_points
                sp = extract_support_points(frag["contour"])
                frag["support_points"] = sp
            frag["edges"] = fio._extract_edges_from_contour(
                contour_for_edges, sp, frag["mask"], min_edge_length=15.0)
        for e in frag["edges"]:
            if e.get("is_torn") and e.get("_curvature") is None:
                resamp, curv = compute_curvature_string(e["pts"])
                e["_resampled"] = resamp
                e["_curvature"] = curv
        _ensure_sdt(frag)
        if frag.get("paper_lab") is None and image_rgb is not None:
            from .fragment_io import _paper_lab_fingerprint
            frag["paper_lab"] = _paper_lab_fingerprint(image_rgb, frag["mask"])


# ══════════════════════════════════════════════════════════════════════════
#  Internal: per-edge-pair matcher (full 360 deg, both orientations)
# ══════════════════════════════════════════════════════════════════════════

# How much closer must the direct-orientation SW score be to the
# complementary-orientation score before we accept "direct" instead?
# Strict equality is too brittle; we want a small bias toward complementary
# (the physical prior for torn seams).
_COMPLEMENT_TIE_MARGIN = 0.05


def _match_edge_pair(edge_a: dict, edge_b: dict,
                     frag_a: dict, frag_b: dict,
                     image_rgb: np.ndarray | None,
                     ) -> dict | None:
    """Match a single (edge_a, edge_b) pair end-to-end.

    Returns the match dict on success, or ``None`` after logging the gate
    that failed.
    """
    tag = f"frag{frag_a['id']}.e{edge_a.get('_idx', '?')} <> " \
          f"frag{frag_b['id']}.e{edge_b.get('_idx', '?')}"

    # Gate 0: sanity on edge geometry and length
    if len(edge_a.get("pts", ())) < 3 or len(edge_b.get("pts", ())) < 3:
        logger.debug("%s reject edge_degenerate", tag)
        return None
    if (edge_a["length"] < cfg.MIN_TORN_EDGE_PX or
            edge_b["length"] < cfg.MIN_TORN_EDGE_PX):
        logger.debug("%s reject short_edge len_a=%.1f len_b=%.1f",
                     tag, edge_a["length"], edge_b["length"])
        return None

    # Gate 1: curvature strings present and non-degenerate
    resamp_a = edge_a.get("_resampled")
    curv_a = edge_a.get("_curvature")
    resamp_b = edge_b.get("_resampled")
    curv_b = edge_b.get("_curvature")
    if resamp_a is None or curv_a is None or resamp_b is None or curv_b is None:
        resamp_a, curv_a = compute_curvature_string(np.asarray(edge_a["pts"]))
        resamp_b, curv_b = compute_curvature_string(np.asarray(edge_b["pts"]))
        edge_a["_resampled"], edge_a["_curvature"] = resamp_a, curv_a
        edge_b["_resampled"], edge_b["_curvature"] = resamp_b, curv_b
    if len(curv_a) < cfg.SW_MIN_ALIGNED or len(curv_b) < cfg.SW_MIN_ALIGNED:
        logger.debug("%s reject curv_too_short", tag)
        return None
    if (np.std(curv_a) < cfg.CURV_MIN_STD or
            np.std(curv_b) < cfg.CURV_MIN_STD):
        logger.debug("%s reject low_curv_variance", tag)
        return None

    # Gate 2: Smith-Waterman, BOTH orientations, pick the higher (with a
    # small margin in favour of "complementary").
    curv_b_comp = -np.asarray(curv_b)[::-1]
    score_comp, ia_c, ib_c = smith_waterman_real(curv_a, curv_b_comp)
    score_dir,  ia_d, ib_d = smith_waterman_real(curv_a, curv_b)
    n_b = len(curv_b)

    # Tie-break: prefer complementary if within 5% of direct.
    if score_comp >= score_dir * (1.0 - _COMPLEMENT_TIE_MARGIN):
        sw_score = score_comp
        idx_a = ia_c
        idx_b_orig = [n_b - 1 - k for k in ib_c]
        sub_curv_b = curv_b_comp
        idx_b_for_corr = ib_c
        orientation = "complementary"
    else:
        sw_score = score_dir
        idx_a = ia_d
        idx_b_orig = ib_d
        sub_curv_b = np.asarray(curv_b)
        idx_b_for_corr = ib_d
        orientation = "direct"

    if sw_score < cfg.SW_MIN_SCORE or len(idx_a) < cfg.SW_MIN_ALIGNED:
        logger.debug("%s reject sw_weak score=%.2f n=%d",
                     tag, sw_score, len(idx_a))
        return None

    # Map curvature indices to resampled-point indices (offset to align
    # with the smoothing window).
    offset = cfg.CURV_SMOOTH_WINDOW // 2 + 1
    pt_idx_a = np.clip(np.array(idx_a) + offset, 0, len(resamp_a) - 1)
    pt_idx_b = np.clip(np.array(idx_b_orig) + offset, 0, len(resamp_b) - 1)
    matched_a = resamp_a[pt_idx_a]
    matched_b = resamp_b[pt_idx_b]
    if len(matched_a) < 3:
        logger.debug("%s reject few_matched_pts", tag)
        return None

    # Gate 3: Procrustes initial pose.
    # Always run multi-seed sub-arc Procrustes; when both fragments have
    # a canonical text angle, FILTER seeds to those near the text-prior
    # rotation (delta or delta + pi). Text orientation gives the global
    # rotation constraint; sub-arc Procrustes gives the precise local
    # translation. Combining them prevents the "matched a wrong-edge
    # sub-arc with low RMS but at a wrong rotation" failure mode.
    N = len(matched_a)
    text_prior_used = False
    text_a = frag_a.get("text_angle_canonical")
    text_b = frag_b.get("text_angle_canonical")
    # Text-prior is reliable only when each fragment has multiple text
    # lines to average over. The text detector searches a +-30 deg window;
    # fragments rotated outside that range produce noise with 0-1
    # spurious "lines" — those would corrupt the rotation prior.
    n_lines_a = len(frag_a.get("text_lines") or [])
    n_lines_b = len(frag_b.get("text_lines") or [])
    have_text_prior = (text_a is not None and text_b is not None and
                        n_lines_a >= 3 and n_lines_b >= 3)

    text_priors: list[float] = []   # allowed rotations (radians)
    if have_text_prior:
        delta = float(text_a) - float(text_b)
        text_priors = [delta, delta + math.pi]

    # Multi-seed sub-arc Procrustes.
    n_seeds = max(1, int(cfg.MATCH_PROCRUSTES_SEEDS))
    if N < 4 or n_seeds <= 1:
        seed_windows = [(0, N)]
    else:
        win = max(3, (2 * N + n_seeds) // (n_seeds + 1))
        starts = np.linspace(0, max(0, N - win), n_seeds).astype(int)
        seed_windows = [(int(s), int(min(N, s + win))) for s in starts]
        seed_windows.append((0, N))

    seed_candidates: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    rotation_tol = math.radians(15.0)
    for (ws, we) in seed_windows:
        if we - ws < 3:
            continue
        try:
            _, t_s, rms_s, R_s = procrustes_rigid(
                matched_b[ws:we], matched_a[ws:we])
        except ValueError:
            continue
        if not np.isfinite(rms_s):
            continue
        if have_text_prior:
            # Reject seeds whose rotation disagrees with the text prior.
            ang_s = float(np.arctan2(R_s[1, 0], R_s[0, 0]))
            ok = False
            for d in text_priors:
                drift = abs((ang_s - d + math.pi) % (2 * math.pi) - math.pi)
                if drift <= rotation_tol:
                    ok = True
                    break
            if not ok:
                continue
        seed_candidates.append((rms_s, R_s, t_s, "subarc"))

    # Always also include the centroid-aligning text-prior seed itself —
    # acts as a fallback when every sub-arc Procrustes seed gets filtered
    # out by the rotation tolerance.
    if have_text_prior:
        for d in text_priors:
            ct, st = math.cos(d), math.sin(d)
            R_prior = np.array([[ct, -st], [st, ct]], dtype=np.float64)
            cb = matched_b.mean(axis=0)
            ca = matched_a.mean(axis=0)
            t_prior = ca - R_prior @ cb
            warped = (R_prior @ matched_b.T).T + t_prior
            rms_prior = float(np.sqrt(np.mean(np.sum(
                (warped - matched_a) ** 2, axis=1))))
            if np.isfinite(rms_prior):
                seed_candidates.append((rms_prior, R_prior, t_prior, "text"))
        text_prior_used = True

    if not seed_candidates:
        logger.debug("%s reject procrustes_no_seed", tag)
        return None

    # Pick the lowest-RMS seed.
    seed_candidates.sort(key=lambda s: s[0])
    rms0, R0, t0, seed_kind = seed_candidates[0]
    # Sub-arc seeds are tightly constrained to their window; high seed
    # RMS means the sub-arc isn't real signal. Text-fallback seeds used
    # whole-fragment centroids and may have higher RMS that ICP fixes.
    if seed_kind == "subarc" and rms0 > cfg.MATCH_MAX_RMS:
        logger.debug("%s reject rms_high(seed) rms=%.2f cap=%.2f kind=%s",
                     tag, rms0, cfg.MATCH_MAX_RMS, seed_kind)
        return None

    # Gate 4: ICP — coarse then fine; revert if RMS got worse than seed.
    R_coarse, t_coarse, rms_coarse = icp_refine(
        src_pts=np.asarray(edge_b["pts"]),
        dst_pts=np.asarray(edge_a["pts"]),
        R0=R0, t0=t0,
        max_iter=cfg.ICP_MAX_ITER,
        max_correspondence_dist=cfg.ICP_COARSE_DIST_PX,
    )
    R, t, rms = icp_refine(
        src_pts=np.asarray(edge_b["pts"]),
        dst_pts=np.asarray(edge_a["pts"]),
        R0=R_coarse, t0=t_coarse,
        max_iter=cfg.ICP_MAX_ITER,
        max_correspondence_dist=cfg.ICP_MAX_CORR_DIST_PX,
    )
    icp_reverted = False
    angle_seed = float(np.arctan2(R0[1, 0], R0[0, 0]))
    angle_icp = float(np.arctan2(R[1, 0], R[0, 0]))

    if seed_kind == "subarc":
        # Sub-arc path: revert if ICP made the fit worse.
        if rms > rms0 + 0.5 or not np.isfinite(rms):
            R, t, rms = R0, t0, rms0
            icp_reverted = True
    else:
        # Text-prior path: bind rotation to the text orientation. ICP
        # may translate freely, but if it rotates more than 15 deg away
        # from the seed (which already fixed text orientation +/- 180),
        # that's ICP latching onto a wrong-edge sub-arc — revert to seed.
        drift = abs((angle_icp - angle_seed + math.pi) % (2 * math.pi)
                     - math.pi)
        if drift > math.radians(15.0) or not np.isfinite(rms):
            R, t, rms = R0, t0, rms0
            icp_reverted = True

    angle = float(np.arctan2(R[1, 0], R[0, 0]))

    # Final RMS gate (covers both seed paths).
    if rms > cfg.MATCH_MAX_RMS:
        logger.debug("%s reject rms_high(final) rms=%.2f cap=%.2f kind=%s",
                     tag, rms, cfg.MATCH_MAX_RMS, seed_kind)
        return None

    # Gate 5: SDT physical gate — fail loud if SDT missing.
    ok_sdt, sdt_diag = _sdt_physical_gate(
        frag_a, frag_b, R, t, matched_a, matched_b)
    if not ok_sdt:
        logger.debug("%s reject sdt %s", tag, sdt_diag.get("rejected_reason"))
        return None

    # Gate 6: full-edge physical fit cost.
    fit = evaluate_edge_fit(frag_a, frag_b, edge_a, edge_b, R, t)
    fit_cost = float(fit["fit_cost"])
    if fit_cost > cfg.MAX_ATTACH_COST:
        logger.debug("%s reject fit_cost_high cost=%.1f cap=%.1f",
                     tag, fit_cost, cfg.MAX_ATTACH_COST)
        return None

    # Gate 7: optional Siamese edge-matcher CNN.
    em_prob: float | None = None
    em_cos: float | None = None
    if image_rgb is not None and getattr(cfg, "EDGE_MATCHER_ENABLED", False):
        from . import edge_matcher as em
        if em.is_loaded():
            try:
                em_out = em.score_edge_pair(image_rgb, edge_a, edge_b,
                                              orientation)
            except Exception as exc:
                logger.warning("%s edge_matcher raised %s — gate skipped",
                               tag, exc)
                em_out = None
            if em_out is not None:
                em_prob = float(em_out["match_prob"])
                em_cos = float(em_out["cosine"])
                if em_prob < cfg.EDGE_MATCHER_MIN_SCORE:
                    logger.debug("%s reject edge_matcher_low prob=%.3f",
                                 tag, em_prob)
                    return None

    # Appearance signals (informative, not gating)
    paper_score = _paper_lab_score(frag_a, frag_b)
    strip_dis = (_strip_ncc_dissimilarity(image_rgb, edge_a, edge_b,
                                            matched_a, matched_b)
                 if image_rgb is not None else 0.5)
    strip_score = float(max(0.0, 1.0 - strip_dis))

    dinov2_score = None
    if (frag_a.get("dinov2") is not None and
            frag_b.get("dinov2") is not None):
        try:
            from .appearance import seam_patch_cosine
            M_ba = affine_from_Rt(R, t)
            I = np.eye(3, dtype=np.float64)
            seam_mid = matched_a[len(matched_a) // 2]
            tgt = matched_a[-1] - matched_a[0]
            tn = float(np.linalg.norm(tgt))
            if tn > 1e-6:
                tgt = tgt / tn
                seam_normal = np.array([-tgt[1], tgt[0]], dtype=np.float64)
            else:
                seam_normal = np.array([1.0, 0.0])
            cos, n_patches = seam_patch_cosine(
                frag_a, frag_b, I, M_ba, seam_mid, seam_normal)
            if n_patches >= 2:
                dinov2_score = float(0.5 * (cos + 1.0))
        except Exception as exc:
            logger.info("seam_patch_cosine unavailable: %s", exc)

    # Aggregate appearance: mean of available signals.
    appearance_signals = [paper_score, strip_score]
    if dinov2_score is not None:
        appearance_signals.append(dinov2_score)
    appearance = float(np.mean(appearance_signals))

    # Confidence: 0.7 * geometry + 0.3 * appearance.
    geom = float(max(0.0, 1.0 - fit_cost / max(cfg.MAX_ATTACH_COST, 1e-6)))
    confidence = float(np.clip(
        cfg.CONF_W_GEOMETRY * geom + cfg.CONF_W_APPEARANCE * appearance,
        0.0, 1.0))

    return {
        "frag_i": frag_a["id"],
        "frag_j": frag_b["id"],
        "edge_i": int(edge_a.get("_idx", -1)),
        "edge_j": int(edge_b.get("_idx", -1)),
        "R":            R.astype(np.float64),
        "t":            np.asarray(t, dtype=np.float64),
        "translation":  np.asarray(t, dtype=np.float64),  # alias used by some helpers
        "angle":        float(angle),
        "rms":          float(rms),
        "rms_seed":     float(rms0),
        "icp_reverted": bool(icp_reverted),
        "orientation":  orientation,
        "text_prior_used": bool(text_prior_used),
        "sw_score":     float(sw_score),
        "n_aligned":    int(len(idx_a)),
        "matched_a":    np.asarray(matched_a, dtype=np.float64),
        "matched_b":    np.asarray(matched_b, dtype=np.float64),
        "confidence":   confidence,
        "fit_cost":     fit_cost,
        "fit_overlap_px":  float(fit["fit_overlap_px"]),
        "fit_gap_px":      float(fit["fit_gap_px"]),
        "fit_coverage":    float(fit["fit_coverage"]),
        "appearance":   appearance,
        "geom_score":   geom,
        "paper_score":  paper_score,
        "strip_score":  strip_score,
        "dinov2_score": dinov2_score,
        "edge_matcher_prob": em_prob,
        "edge_matcher_cos":  em_cos,
        "sdt_gate":     sdt_diag,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Public entry: best match between two fragments
# ══════════════════════════════════════════════════════════════════════════

# Boundary-proximity gate: two fragments whose nearest boundary pixels are
# more than this far apart cannot physically share a seam at the working
# resolution. SAM segments to the OUTSIDE of dark page boundaries so true
# adjacent fragments end up 30-120 px apart at boundary points; we pad to
# 250 to handle scan layouts where fragments were placed with extra space.
# Beyond 250 px the curvature-only matcher starts coincidentally matching
# unrelated tear shapes (false positives); below that, the SDT physical
# gate and full-edge fit_cost catch wrong-pair attempts.
_BOUNDARY_PROX_PX = 250.0

# Edges whose lengths differ by more than this ratio cannot be a seam pair.
_EDGE_LENGTH_RATIO_MAX = 2.5


def _torn_edges(frag: dict) -> list[tuple[int, dict]]:
    out = []
    for k, e in enumerate(frag.get("edges", []) or []):
        if e.get("is_torn"):
            e["_idx"] = int(k)   # cache the index for log tags
            out.append((k, e))
    return out


def match_pair(frag_a: dict, frag_b: dict,
               image_rgb: np.ndarray | None = None,
               *, hint: str | None = None,
               direction_aware: bool = True) -> dict | None:
    """Find the best torn-edge pairing between two fragments.

    Returns the match dict on success (lowest fit_cost across all torn
    edge pairs), or ``None`` when every candidate was rejected.

    Parameters
    ----------
    direction_aware:
        When True (default and the production case), apply the
        outward-normal facing gate using the centroid offset between the
        two fragments. Both fragments must live in the SAME canvas frame
        (which is what Phase 1 produces). When False, skip this gate —
        fragments are assumed to be in independent local frames (tests,
        boards, partial batches) and a centroid-offset direction would
        be meaningless.
    """
    torn_a = _torn_edges(frag_a)
    torn_b = _torn_edges(frag_b)
    if not torn_a or not torn_b:
        logger.debug("frag%s <> frag%s reject no_torn_edges",
                     frag_a.get("id"), frag_b.get("id"))
        return None

    # Boundary-proximity pre-gate (cheap rejection of unrelated fragments).
    # Only meaningful when fragments share a canvas (direction_aware==True);
    # in isolated-canvas mode their boundary pixels are in different
    # frames and the proximity number means nothing.
    if direction_aware:
        bnd_a = frag_a.get("boundary_pixels")
        bnd_b = frag_b.get("boundary_pixels")
        if (bnd_a is not None and bnd_b is not None and
                len(bnd_a) and len(bnd_b)):
            try:
                min_d = float(cKDTree(np.asarray(bnd_a, dtype=np.float64))
                              .query(np.asarray(bnd_b, dtype=np.float64), k=1)[0]
                              .min())
                if min_d > _BOUNDARY_PROX_PX:
                    logger.debug("frag%s <> frag%s reject boundary_far d=%.1f",
                                 frag_a["id"], frag_b["id"], min_d)
                    return None
            except Exception:
                pass  # boundary check is best-effort; never fatal

    centroid_off = None
    if direction_aware:
        centroid_off = (np.asarray(frag_b["centroid"], dtype=np.float64) -
                        np.asarray(frag_a["centroid"], dtype=np.float64))

    best: dict | None = None
    best_fit = float("inf")

    for ki, ea in torn_a:
        for kj, eb in torn_b:
            # Edge-length ratio gate (cheap pre-screen).
            la, lb = float(ea["length"]), float(eb["length"])
            if la <= 0 or lb <= 0:
                continue
            ratio = max(la, lb) / min(la, lb)
            if ratio > _EDGE_LENGTH_RATIO_MAX:
                continue

            # Directional facing gate — only when both fragments share a
            # canvas frame. Cuts ~75% of pairs in production AND prevents
            # cross-cluster false positives (tear-on-left-of-A matched to
            # tear-on-left-of-C when B sits between them).
            if (direction_aware and centroid_off is not None and
                    not _edges_face_each_other(ea, eb, centroid_off)):
                continue

            ea["_idx"] = ki
            eb["_idx"] = kj
            result = _match_edge_pair(ea, eb, frag_a, frag_b, image_rgb)
            if result is None:
                continue
            if result["fit_cost"] < best_fit:
                best_fit = result["fit_cost"]
                best = result

    if best is None:
        logger.debug("frag%s <> frag%s reject all_pairs_failed",
                     frag_a["id"], frag_b["id"])
        return None
    best["hint"] = hint
    return best


__all__ = [
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
