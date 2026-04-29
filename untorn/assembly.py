"""
untorn.assembly
===============
Layout-agnostic global assembly by MST-growth, with edge-contact
refinement and contact-constrained bundle adjustment (Steps 4 + 5).

Strategy
--------
1. Enumerate every torn-edge × torn-edge pair, prefiltered by edge-length
   ratio, paper-color compatibility, and (when enabled) the LBP grid
   filter. Pairs are ordered by a combined ``paper_lab + (1 −
   min(anchor_strength))`` cost so seams owned by long, curvy,
   distinctive edges are matched first (Step 3).
2. Score each surviving pair via the four-/five-gate matcher; the matcher
   uses each fragment's canonical text-rotation angle as a Procrustes
   seed (Step 3) so the relative pose comes out level by construction.
3. Seed the MST. We pick the strongest mutual-rank pair (each side ranks
   the other near the top); fall back to highest absolute confidence when
   no mutual seed attaches.
4. Grow the MST by attaching the highest-confidence pair whose one side
   is already placed. Conflicts (multiple free fragments wanting the
   same anchor) are resolved by networkx's max-weight matching.
5. **Seam-contact refinement (Step 4)**. For every recorded attach, run
   ``seam_solver.refine_pair`` to shrink (Δθ, Δdx, Δdy) under the
   ``evaluate_edge_fit`` cost (gap + overlap + uncovered) plus an
   absolute SDT-penetration penalty. Updates the cache so downstream
   steps see the refined relative pose.
6. **Contact-constrained bundle adjustment (Step 5)**. LM optimises the
   pose graph with two residual sources per pair: the legacy SW
   correspondences AND dense edge-to-edge correspondences resampled at
   ``BA_DENSE_EDGE_SAMPLES`` per torn edge. The dense term physically
   welds polylines together — the seam goes from "two pinned points"
   to "two welded curves".
7. Orphan rescue (relaxed) and aggressive rescue (force-match) pick up
   any fragment that didn't attach during MST growth. The seam refiner
   is run again afterwards so rescued attaches are tightened.
8. Cluster reconciliation merges separated clusters via the
   lowest-cost cross-cluster bridge match, then BA polishes the now-
   unified pose graph.

NOTE — removed in Step 3: cosmetic post-placement text rotation passes
(`_apply_per_fragment_text_rotation`, `_maybe_global_text_rotation`).
Per-fragment text orientation is now a Procrustes seed at match time;
rotating fragments after placement would invalidate the seam alignment
ICP and the seam solver just achieved.

Public contract (matches the old module so pipeline.py is unchanged):
    reconstruct(fragments, image_rgb, debug_dir) -> dict[int, 3x3 affine]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import networkx as nx

from . import config as cfg
from . import matching as M
from . import fragment_profile as fp
from . import edge_rank as er
from . import seam_solver
from .io_utils import save_image


# ══════════════════════════════════════════════════════════════════════════
#  SE(2) / canvas helpers (identical semantics to the legacy module)
# ══════════════════════════════════════════════════════════════════════════

def _fit_cost(match: dict | None) -> float:
    if match is None:
        return float("inf")
    return float(match.get("fit_cost", float("inf")))


def _transform_contour(frag: dict, M_global: np.ndarray) -> np.ndarray:
    pts = frag["contour"].astype(np.float64).reshape(-1, 2)
    return M.affine_apply(M_global, pts)


def _transformed_bbox(frag: dict, M_global: np.ndarray
                      ) -> tuple[float, float, float, float]:
    x, y, w, h = frag["bbox"]
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                       dtype=np.float64)
    warped = M.affine_apply(M_global, corners)
    return (float(warped[:, 0].min()), float(warped[:, 1].min()),
            float(warped[:, 0].max()), float(warped[:, 1].max()))


def _global_frame_bounds(fragments: list[dict],
                         transforms: dict[int, np.ndarray],
                         placed: set[int]
                         ) -> tuple[float, float, float, float]:
    if not placed:
        return 0.0, 0.0, 0.0, 0.0
    mn_x = mn_y = float("inf")
    mx_x = mx_y = float("-inf")
    for i in placed:
        x0, y0, x1, y1 = _transformed_bbox(fragments[i], transforms[i])
        mn_x = min(mn_x, x0); mn_y = min(mn_y, y0)
        mx_x = max(mx_x, x1); mx_y = max(mx_y, y1)
    return mn_x, mn_y, mx_x, mx_y


def _warp_mask_to_canvas(frag: dict, M_global: np.ndarray,
                         canvas_w: int, canvas_h: int,
                         ox: float, oy: float) -> np.ndarray:
    M_shifted = M_global.copy()
    M_shifted[0, 2] += ox
    M_shifted[1, 2] += oy
    bx, by, bw, bh = frag["bbox"]
    sub_mask = frag["mask"][by:by + bh, bx:bx + bw]
    origin = M_shifted @ np.array([bx, by, 1.0], dtype=np.float64)
    M_sub = M_shifted.copy()
    M_sub[0, 2] = origin[0]
    M_sub[1, 2] = origin[1]
    M_2x3 = M_sub[:2, :].astype(np.float32)
    warped = cv2.warpAffine(sub_mask, M_2x3, (canvas_w, canvas_h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped > 127


def _global_overlap_ok(fragments: list[dict],
                       transforms: dict[int, np.ndarray],
                       new_idx: int, placed: set[int]) -> bool:
    if not placed:
        return True
    mn_x, mn_y, mx_x, mx_y = _global_frame_bounds(
        fragments, transforms, placed | {new_idx})
    pad = 20
    canvas_w = int(mx_x - mn_x) + 2 * pad
    canvas_h = int(mx_y - mn_y) + 2 * pad
    if canvas_w <= 0 or canvas_h <= 0 \
            or canvas_w > cfg.OVERLAP_CANVAS_MAX \
            or canvas_h > cfg.OVERLAP_CANVAS_MAX:
        return True
    ox = -mn_x + pad
    oy = -mn_y + pad
    mask_new = _warp_mask_to_canvas(
        fragments[new_idx], transforms[new_idx], canvas_w, canvas_h, ox, oy)
    area_new = int(mask_new.sum())
    if area_new < 10:
        return True
    for p in placed:
        if p == new_idx:
            continue
        mask_p = _warp_mask_to_canvas(
            fragments[p], transforms[p], canvas_w, canvas_h, ox, oy)
        inter = int(np.logical_and(mask_new, mask_p).sum())
        smaller = min(area_new, int(mask_p.sum()))
        if smaller > 0 and inter / smaller > cfg.RECON_OVERLAP_THRESH:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════
#  Pair-match cache — memoize (i, j) -> best match dict
# ══════════════════════════════════════════════════════════════════════════

class _MatchCache:
    """Lazy memoized pair matcher. Mirrors the legacy interface."""

    def __init__(self, fragments: list[dict], image_rgb: np.ndarray):
        self.fragments = fragments
        self.image_rgb = image_rgb
        self.cache: dict[tuple[int, int], dict | None] = {}
        self.n_hits = 0
        self.n_calls = 0

    def match(self, i: int, j: int, direction: str | None = None
              ) -> dict | None:
        if i == j:
            return None
        lo, hi = (i, j) if i < j else (j, i)
        if (lo, hi) in self.cache:
            self.n_hits += 1
            cached = self.cache[(lo, hi)]
            if cached is None:
                return None
            out = dict(cached)
            out["direction_hint"] = direction
            if cached["frag_i"] != self.fragments[i]["id"]:
                R_inv = cached["R"].T
                t_inv = -R_inv @ cached["translation"].reshape(2)
                out["R"] = R_inv
                out["translation"] = t_inv
                out["frag_i"] = self.fragments[i]["id"]
                out["frag_j"] = self.fragments[j]["id"]
                out["edge_i"], out["edge_j"] = cached["edge_j"], cached["edge_i"]
                out["matched_a"], out["matched_b"] = \
                    cached["matched_b"], cached["matched_a"]
            return out
        self.n_calls += 1
        result = M.match_pair(self.fragments[lo], self.fragments[hi],
                              self.image_rgb, direction_hint=direction)
        self.cache[(lo, hi)] = result
        return self.match(i, j, direction)


# ══════════════════════════════════════════════════════════════════════════
#  Attach primitive — "snap j onto i under i's current global pose"
# ══════════════════════════════════════════════════════════════════════════

def _seam_line_ok(match: dict) -> tuple[bool, str]:
    """
    Final connection-line gate: the matched edges must physically touch
    along their shared length under (R, t). Reads fields populated by
    matching.evaluate_edge_fit (overlap, gap, coverage).

    Returns (accepted, reason). Accepted means the seam is clean; the
    reason string is logged for the operator when we reject.
    """
    if "fit_cost" not in match:
        return True, "no_fit_metrics"
    cost = float(match.get("fit_cost", 0.0))
    if cost > cfg.MAX_ATTACH_COST:
        return False, f"fit_cost={cost:.1f} > max"
    overlap_frac = float(match.get("fit_overlap_frac", 0.0))
    if overlap_frac > cfg.SEAM_MAX_OVERLAP_FRAC:
        return False, f"overlap_frac={overlap_frac:.2f}"
    gap_px = float(match.get("fit_gap_px", 0.0))
    if gap_px > cfg.SEAM_MAX_GAP_PX:
        return False, f"gap_px={gap_px:.1f}"
    coverage = float(match.get("fit_coverage", 1.0))
    if coverage < cfg.SEAM_MIN_COVERAGE:
        return False, f"coverage={coverage:.2f}"
    return True, "ok"


def _attach(fragments: list[dict],
            transforms: dict[int, np.ndarray],
            placed: set[int], locked: dict[int, float],
            match: dict, i: int, j: int,
            enforce_seam: bool = True) -> tuple[bool, str]:
    """Compose transforms so fragment j sits where `match` says it should.

    Returns (success, reason). Reason is "ok" on success, otherwise a
    short tag indicating which gate rejected the attach.
    """
    if enforce_seam:
        ok, reason = _seam_line_ok(match)
        if not ok:
            return False, f"seam:{reason}"
    M_rel = M.affine_from_Rt(match["R"], match["translation"])
    M_j_global = transforms[i] @ M_rel
    old = transforms.get(j, np.eye(3)).copy()
    transforms[j] = M_j_global
    if not _global_overlap_ok(fragments, transforms, j, placed):
        transforms[j] = old
        return False, "overlap"
    placed.add(j)
    locked[j] = max(locked.get(j, 0.0), match["confidence"])
    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════
#  Candidate pair enumeration — the layout-agnostic replacement for the
#  corner-seeded neighbor graph. We enumerate every distinct-fragment
#  pair, prefilter by paper-color compatibility and edge-length ratio,
#  then score each surviving pair through the four-gate matcher.
# ══════════════════════════════════════════════════════════════════════════

def _max_torn_edge_length(frag: dict) -> float:
    longest = 0.0
    for e in frag.get("edges", []):
        if e.get("is_torn", False):
            longest = max(longest, float(e["length"]))
    return longest


def _approx_centroid_distance(frag_i: dict, frag_j: dict) -> float:
    """Fast local-distance proxy used by candidate prefiltering."""
    c_i = np.asarray(frag_i.get("centroid", (0.0, 0.0)), dtype=np.float64)
    c_j = np.asarray(frag_j.get("centroid", (0.0, 0.0)), dtype=np.float64)
    return float(np.linalg.norm(c_i - c_j))


def _enumerate_pair_candidates(fragments: list[dict],
                                image_rgb: np.ndarray | None = None
                                ) -> list[tuple[int, int]]:
    """All i<j pairs with at least one torn edge each, paper-color compatible.

    Pair score (lower is better, used to rank within
    ``ASSEMBLY_MAX_CANDIDATE_PAIRS``):

        score = paper_lab_ΔE  +  λ·(1 − min(anchor_a, anchor_b))

    where ``anchor_a/b`` is the max torn-edge anchor strength on each
    fragment (assigned by ``fragment_io._assign_anchor_strengths``). This
    prioritises pairs where BOTH fragments own at least one strong torn
    edge — long, curvy, ink-rich — over pairs where one side has only
    short or near-straight torn segments.

    When ``cfg.GRID_FILTER_ENABLED`` is set and ``image_rgb`` is provided, the
    survivors are additionally intersected with the grid/binary fast-filter
    top-K candidates per fragment (see ``untorn.grid_filter``).
    """
    from .matching import _paper_lab_delta
    n = len(fragments)
    max_edges = [_max_torn_edge_length(f) for f in fragments]
    ratio_max = float(getattr(
        cfg, "ASSEMBLY_MAX_EDGE_LENGTH_RATIO",
        getattr(cfg, "ASSEMBLY_EDGE_LENGTH_RATIO_MAX", 3.0)))
    edge_prox = float(getattr(cfg, "ASSEMBLY_EDGE_PROXIMITY_PX", 200.0))
    lab_max = float(cfg.MATCH_PAPER_COLOR_DELTA_MAX)
    # Weight on the anchor-strength penalty. Same units as ΔE so the two
    # terms add cleanly; tuned so a pair with anchor 0 costs about as much
    # as a paper-color delta of 1.0×ΔE_max.
    anchor_weight = lab_max
    pair_scores: list[tuple[float, int, int]] = []
    for i in range(n):
        if max_edges[i] <= 0.0:
            continue
        anchor_i = float(fragments[i].get("max_anchor_strength", 0.0))
        for j in range(i + 1, n):
            if max_edges[j] <= 0.0:
                continue
            # Edge-length ratio prefilter.
            a, b = max_edges[i], max_edges[j]
            if max(a, b) / max(min(a, b), 1e-6) > ratio_max:
                continue
            # Locality prefilter.
            if _approx_centroid_distance(fragments[i], fragments[j]) > (3.0 * edge_prox):
                continue
            d = _paper_lab_delta(fragments[i], fragments[j])
            if np.isfinite(d) and d > lab_max * 2.0:
                continue
            paper_term = d if np.isfinite(d) else lab_max
            anchor_j = float(fragments[j].get("max_anchor_strength", 0.0))
            anchor_term = anchor_weight * (1.0 - min(anchor_i, anchor_j))
            pair_scores.append((paper_term + anchor_term, i, j))
    # Score-ascending; cap.
    pair_scores.sort()
    paper_pairs = [(i, j) for (_s, i, j) in
                    pair_scores[: cfg.ASSEMBLY_MAX_CANDIDATE_PAIRS]]

    # Grid/binary fast-filter intersection.
    if not getattr(cfg, "GRID_FILTER_ENABLED", False) or image_rgb is None \
            or len(paper_pairs) == 0:
        return paper_pairs

    try:
        from . import grid_filter as gf
    except Exception as exc:
        print(f"  -- Grid filter unavailable ({exc}); skipping pre-screen.")
        return paper_pairs

    t0 = time.time()
    try:
        index = gf.build_index(fragments, image_rgb)
        per_frag = gf.screen_candidates(
            index, top_k=int(getattr(cfg, "GRID_FILTER_TOP_K", 8)))
    except Exception as exc:
        print(f"  -- Grid filter failed ({exc}); falling back to paper "
              f"prefilter only.")
        return paper_pairs

    keep: set[tuple[int, int]] = set()
    for i, partners in per_frag.items():
        for (j, _score) in partners:
            keep.add((min(i, j), max(i, j)))

    survivors = [(i, j) for (i, j) in paper_pairs
                  if (min(i, j), max(i, j)) in keep]
    n_kept = len(survivors)
    n_drop = len(paper_pairs) - n_kept
    avg_blocks = (sum(b.n_blocks for b in index.blocks) /
                   max(len(index.blocks), 1))
    print(f"  -- Grid filter: kept {n_kept}/{len(paper_pairs)} pairs "
          f"(dropped {n_drop}, avg {avg_blocks:.1f} blocks/frag) "
          f"({time.time() - t0:.2f}s)")
    return survivors


# ══════════════════════════════════════════════════════════════════════════
#  Text-line rotation (cosmetic) — REMOVED in Step 3.
#
#  The previous helpers `_global_text_rotation_angle`,
#  `_get_fragment_text_orientation`, `_apply_per_fragment_text_rotation`,
#  `_apply_global_rotation`, `_placed_centroid`, `_maybe_global_text_rotation`
#  rotated whole clusters and individual fragments AFTER placement to
#  cosmetically level baselines. They have been removed because:
#
#    1. The matcher now uses each fragment's canonical text-rotation as a
#       Procrustes seed (matching._match_edge_pair, "text_prior_used"), so
#       baselines come out levelled by construction.
#    2. Post-placement rotation invalidates the seam alignment ICP just
#       achieved — pixel-level seam tightness > cosmetic baseline tilt.
#    3. The two cosmetic passes ran with different overlap thresholds and
#       could undo each other's work.
#
#  If a future failure mode demands global rotation correction, prefer the
#  contact-constrained bundle adjustment (Step 5) — it can absorb a global
#  rotation residual jointly with seam constraints rather than as a separate
#  cosmetic pass.
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
#  Bundle adjustment — pin seed, vary every other placed fragment
# ══════════════════════════════════════════════════════════════════════════

def _resample_polyline(pts: np.ndarray, n_samples: int) -> np.ndarray:
    """Equal-arc-length resample. Returns (n_samples, 2) or empty if degenerate."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros((0, 2), dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        return pts[:1].astype(np.float64)
    targets = np.linspace(0.0, total, n_samples)
    out = np.empty((n_samples, 2), dtype=np.float64)
    j = 0
    for i, tval in enumerate(targets):
        while j < len(seg) - 1 and cum[j + 1] < tval:
            j += 1
        denom = float(seg[j]) if seg[j] > 1e-9 else 1.0
        alpha = (tval - cum[j]) / denom if seg[j] > 1e-9 else 0.0
        out[i] = pts[j] + alpha * (pts[j + 1] - pts[j])
    return out


def _bundle_adjust_poses(fragments: list[dict],
                         transforms: dict[int, np.ndarray],
                         placed: set[int], locked: dict[int, float],
                         pinned: set[int],
                         cache: _MatchCache,
                         merge_log: list[dict]) -> int:
    """
    LM refinement of every placed fragment's pose except `pinned`.

    Step 5 — contact-constrained BA. Residuals come from two sources per
    placed pair (lo, hi):

        1. Sparse SW seam-coincidence (legacy). For each correspondence
           between matched_a (in `lo`'s frame) and matched_b (in `hi`'s
           frame), residual = world(ma) − world(mb).

        2. Dense edge-to-edge contact. We resample each torn edge of the
           pair at ``BA_DENSE_EDGE_SAMPLES`` equal-arc points and ask
           every warped edge_b sample to coincide with its current
           nearest neighbour on warped edge_a (and the symmetric way).
           This is the residual that physically welds the two polylines
           along their full length — a generalisation of ICP from one
           pair to the whole pose graph.

    The dense term is rebuilt each LM evaluation (cKDTree per pair, per
    iteration); on the scenes we care about (≤ 40 fragments) that is
    sub-second total.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return 0
    from scipy.spatial import cKDTree

    use_dense = bool(getattr(cfg, "BA_DENSE_EDGE_ENABLED", True))
    n_dense = max(4, int(getattr(cfg, "BA_DENSE_EDGE_SAMPLES", 32)))
    w_dense = float(getattr(cfg, "BA_DENSE_EDGE_WEIGHT", 0.6))

    constraints: list[dict] = []
    for (lo, hi), m in cache.cache.items():
        if m is None or lo not in placed or hi not in placed:
            continue
        ma = np.asarray(m.get("matched_a"), dtype=np.float64)
        mb = np.asarray(m.get("matched_b"), dtype=np.float64)
        if ma.size == 0 or mb.size == 0 or len(ma) != len(mb) or len(ma) < 3:
            continue
        fc = float(m.get("fit_cost", 10.0))
        sw_w = float(np.sqrt(1.0 / max(fc, 1.0)))

        # Resolve the (edge_a, edge_b) for the dense term. Cached `m` is
        # stored in (lo, hi) orientation in the cache, but the
        # _MatchCache.match() layer normalises orientation to whichever
        # access order the caller used; here we read `cache.cache[(lo,hi)]`
        # directly so the stored orientation governs which fragment owns
        # which edge_idx.
        edge_lo_pts = None
        edge_hi_pts = None
        if use_dense:
            edge_i = int(m.get("edge_i", -1))
            edge_j = int(m.get("edge_j", -1))
            edges_lo = fragments[lo].get("edges") or []
            edges_hi = fragments[hi].get("edges") or []
            # cached orientation: m["frag_i"] should equal fragments[lo]["id"]
            # since the cache stores in (lo, hi) order. Defensively detect it.
            lo_is_i = (int(m.get("frag_i", -1)) == int(fragments[lo]["id"]))
            edge_lo_idx = edge_i if lo_is_i else edge_j
            edge_hi_idx = edge_j if lo_is_i else edge_i
            if 0 <= edge_lo_idx < len(edges_lo) and 0 <= edge_hi_idx < len(edges_hi):
                e_lo = edges_lo[edge_lo_idx]
                e_hi = edges_hi[edge_hi_idx]
                if e_lo.get("is_torn") and e_hi.get("is_torn"):
                    edge_lo_pts = _resample_polyline(
                        np.asarray(e_lo["pts"], dtype=np.float64), n_dense)
                    edge_hi_pts = _resample_polyline(
                        np.asarray(e_hi["pts"], dtype=np.float64), n_dense)
                    if len(edge_lo_pts) < 2 or len(edge_hi_pts) < 2:
                        edge_lo_pts = edge_hi_pts = None

        constraints.append({
            "lo": lo, "hi": hi,
            "ma": ma, "mb": mb, "sw_w": sw_w,
            "edge_lo_pts": edge_lo_pts, "edge_hi_pts": edge_hi_pts,
        })

    if not constraints:
        return 0

    varying = sorted(i for i in placed if i not in pinned)
    if not varying:
        return 0

    fixed_Rt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in placed:
        if i in pinned:
            T = transforms[i]
            fixed_Rt[i] = (T[:2, :2].copy(), T[:2, 2].copy())

    x0 = np.zeros(3 * len(varying), dtype=np.float64)
    for k, i in enumerate(varying):
        T = transforms[i]
        x0[3 * k]     = float(np.arctan2(T[1, 0], T[0, 0]))
        x0[3 * k + 1] = float(T[0, 2])
        x0[3 * k + 2] = float(T[1, 2])

    def _residuals(x: np.ndarray) -> np.ndarray:
        Rs: dict[int, np.ndarray] = {}
        ts: dict[int, np.ndarray] = {}
        for k, i in enumerate(varying):
            theta = x[3 * k]; tx = x[3 * k + 1]; ty = x[3 * k + 2]
            ct = np.cos(theta); st = np.sin(theta)
            Rs[i] = np.array([[ct, -st], [st, ct]], dtype=np.float64)
            ts[i] = np.array([tx, ty], dtype=np.float64)
        parts: list[np.ndarray] = []
        for c in constraints:
            lo = c["lo"]; hi = c["hi"]
            R_lo, t_lo = (fixed_Rt[lo] if lo in fixed_Rt else (Rs[lo], ts[lo]))
            R_hi, t_hi = (fixed_Rt[hi] if hi in fixed_Rt else (Rs[hi], ts[hi]))
            # Sparse SW correspondences.
            world_a_sw = c["ma"] @ R_lo.T + t_lo
            world_b_sw = c["mb"] @ R_hi.T + t_hi
            parts.append(c["sw_w"] * (world_a_sw - world_b_sw).ravel())
            # Dense edge correspondences (Step 5).
            edge_lo_pts = c["edge_lo_pts"]; edge_hi_pts = c["edge_hi_pts"]
            if use_dense and edge_lo_pts is not None and edge_hi_pts is not None:
                warped_lo = edge_lo_pts @ R_lo.T + t_lo
                warped_hi = edge_hi_pts @ R_hi.T + t_hi
                # nearest-neighbour from each side to the other; symmetric
                # so we don't bias the solution toward one polyline.
                tree_lo = cKDTree(warped_lo)
                tree_hi = cKDTree(warped_hi)
                _, idx_lo = tree_lo.query(warped_hi, k=1)
                _, idx_hi = tree_hi.query(warped_lo, k=1)
                parts.append(w_dense * (warped_hi - warped_lo[idx_lo]).ravel())
                parts.append(w_dense * (warped_lo - warped_hi[idx_hi]).ravel())
        return np.concatenate(parts)

    initial_res = _residuals(x0)
    initial_cost = float(0.5 * np.sum(initial_res ** 2))
    t_start = time.time()
    result = least_squares(_residuals, x0, method="lm",
                           max_nfev=cfg.BA_MAX_ITER,
                           ftol=cfg.BA_FUNC_TOL, xtol=cfg.BA_FUNC_TOL)
    x = result.x
    final_cost = float(0.5 * np.sum(result.fun ** 2))

    old_T = {i: transforms[i].copy() for i in varying}
    proposals: list[tuple[int, float, float, np.ndarray]] = []
    for k, i in enumerate(varying):
        theta = float(x[3 * k])
        tx = float(x[3 * k + 1]); ty = float(x[3 * k + 2])
        old = old_T[i]
        old_theta = float(np.arctan2(old[1, 0], old[0, 0]))
        old_tx = float(old[0, 2]); old_ty = float(old[1, 2])
        d_theta_deg = abs(np.degrees(theta - old_theta))
        d_shift_px = float(np.hypot(tx - old_tx, ty - old_ty))
        if d_theta_deg > cfg.BA_MAX_ROTATION_DEG \
                or d_shift_px > cfg.BA_MAX_TRANSLATION_PX:
            continue
        ct = np.cos(theta); st = np.sin(theta)
        T_new = np.array([[ct, -st, tx],
                          [st,  ct, ty],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        proposals.append((i, d_theta_deg, d_shift_px, T_new))

    for (i, _dt, _ds, T_new) in proposals:
        transforms[i] = T_new
    kept = 0
    for (i, _dt, _ds, _T_new) in proposals:
        if not _global_overlap_ok(fragments, transforms, i, placed):
            transforms[i] = old_T[i]
        else:
            kept += 1

    merge_log.append({
        "phase":          "bundle_adjust",
        "n_constraints":  len(constraints),
        "n_varying":      len(varying),
        "n_proposed":     len(proposals),
        "n_updates_kept": kept,
        "cost_initial":   round(initial_cost, 2),
        "cost_final":     round(final_cost, 2),
        "solver_time_s":  round(time.time() - t_start, 2),
        "nfev":           int(result.nfev),
    })
    return kept


# ══════════════════════════════════════════════════════════════════════════
#  Step 4 — post-attach edge-contact pose refinement
# ══════════════════════════════════════════════════════════════════════════

# Phases in the merge log that represent a real "attach edge" — i.e. a
# placement that was driven by a cached pair match we can refine.
_ATTACH_PHASES = frozenset({
    "seed", "mst_attach", "orphan_rescue",
    "aggressive_orphan_rescue",
})


def _refine_seams_from_log(fragments: list[dict],
                           transforms: dict[int, np.ndarray],
                           placed: set[int],
                           cache: _MatchCache,
                           merge_log: list[dict]) -> int:
    """Walk the merge log; for every recorded attach, refine the cached
    relative pose by minimising the edge-contact + overlap cost
    (``seam_solver.refine_pair``). Update the cache in place and recompose
    the attached fragment's global transform from the refined Δ.

    Returns the number of attaches whose refinement actually improved the
    seam.

    Notes:
      * A seam refinement that worsens overlap globally is reverted.
      * Refinements are applied in merge-log order so each Δ propagates
        forward through any chain rooted at the same anchor.
      * Cluster reconcile entries are skipped: those represent a rigid
        cluster shift, not a per-pair alignment.
    """
    if not getattr(cfg, "SEAM_SOLVER_ENABLED", True):
        return 0

    fid_to_idx = {int(fragments[i]["id"]): i for i in range(len(fragments))}
    n_improved = 0

    for entry in list(merge_log):  # snapshot — we append diagnostic entries
        phase = entry.get("phase")
        if phase not in _ATTACH_PHASES:
            continue
        i_id = entry.get("anchor")
        j_id = entry.get("attached")
        if i_id is None or j_id is None:
            continue
        i_idx = fid_to_idx.get(int(i_id))
        j_idx = fid_to_idx.get(int(j_id))
        if i_idx is None or j_idx is None:
            continue
        if i_idx not in placed or j_idx not in placed:
            continue

        m = cache.match(i_idx, j_idx)
        if m is None:
            continue
        edge_i = int(m.get("edge_i", -1))
        edge_j = int(m.get("edge_j", -1))
        if edge_i < 0 or edge_j < 0:
            continue

        R0 = np.asarray(m["R"], dtype=np.float64).copy()
        t0 = np.asarray(m["translation"], dtype=np.float64).reshape(2).copy()
        R_new, t_new, diag = seam_solver.refine_pair(
            fragments[i_idx], fragments[j_idx], edge_i, edge_j, R0, t0)

        if "delta" not in diag or all(v == 0.0 for v in diag["delta"]):
            continue  # nothing to apply

        # Compose attached's NEW global transform from anchor's CURRENT
        # global transform. We use anchor's current transforms[i_idx] —
        # the merge_log replay convention propagates earlier refinements
        # forward without us having to track parent chains explicitly.
        M_rel_new = M.affine_from_Rt(R_new, t_new)
        old_transform = transforms[j_idx].copy()
        transforms[j_idx] = transforms[i_idx] @ M_rel_new

        # Sanity: the refined placement still has to satisfy the global
        # overlap rule. If the refinement traded one improvement for
        # another fragment's overlap, revert.
        if not _global_overlap_ok(fragments, transforms, j_idx, placed - {j_idx}):
            transforms[j_idx] = old_transform
            merge_log.append({
                "phase":       "seam_refine_reverted",
                "anchor":      int(i_id),
                "attached":    int(j_id),
                "reason":      "global_overlap",
                "init_cost":   diag.get("init_cost"),
                "final_cost":  diag.get("final_cost"),
                "delta":       diag.get("delta"),
            })
            continue

        # Accepted — update the cache so any downstream consumer (orphan
        # rescue, BA, cluster reconcile) sees the refined relative pose.
        lo, hi = (i_idx, j_idx) if i_idx < j_idx else (j_idx, i_idx)
        cached = cache.cache.get((lo, hi))
        if cached is not None:
            if int(cached.get("frag_i", -1)) == int(fragments[i_idx]["id"]):
                cached["R"] = R_new.copy()
                cached["translation"] = t_new.copy()
            else:
                # cached is stored in (j_idx, i_idx) orientation; invert.
                R_inv = R_new.T
                t_inv = -R_inv @ t_new
                cached["R"] = R_inv
                cached["translation"] = t_inv

        n_improved += 1
        merge_log.append({
            "phase":       "seam_refine",
            "anchor":      int(i_id),
            "attached":    int(j_id),
            "init_cost":   diag.get("init_cost"),
            "final_cost":  diag.get("final_cost"),
            "delta":       diag.get("delta"),
            "iters":       diag.get("iters"),
        })

    return n_improved


# ══════════════════════════════════════════════════════════════════════════
#  Orphan rescue — one more shot at every placed anchor, relaxed threshold
# ══════════════════════════════════════════════════════════════════════════

def _orphan_rescue(fragments: list[dict],
                   transforms: dict[int, np.ndarray],
                   placed: set[int], locked: dict[int, float],
                   cache: _MatchCache,
                   merge_log: list[dict],
                   min_confidence: float) -> int:
    added = 0
    while True:
        best: tuple[float, float, int, int, dict] | None = None
        for j in range(len(fragments)):
            if j in placed:
                continue
            for i in sorted(placed):
                mdict = cache.match(i, j)
                if mdict is None:
                    continue
                if mdict["confidence"] < min_confidence:
                    continue
                cost = _fit_cost(mdict)
                if cost > cfg.ORPHAN_MAX_ATTACH_COST:
                    continue
                # Rank primarily by confidence, break ties by lower fit cost.
                key = (-float(mdict["confidence"]), cost)
                if best is None or key < (best[0], best[1]):
                    best = (key[0], key[1], i, j, mdict)
        if best is None:
            break
        _neg_conf, cost, i, j, mdict = best
        # Orphans relax the seam-line gate: a near-touch is preferred over
        # leaving the piece floating off-canvas. The global overlap check
        # still applies and prevents truly bad placements.
        ok, reason = _attach(fragments, transforms, placed, locked, mdict, i, j,
                              enforce_seam=False)
        if not ok:
            key = (i, j) if i < j else (j, i)
            cache.cache[key] = None
            continue
        added += 1
        merge_log.append({
            "phase":       "orphan_rescue",
            "anchor":      fragments[i]["id"],
            "attached":    fragments[j]["id"],
            "confidence":  round(float(mdict["confidence"]), 3),
            "fit_cost":    round(cost, 2),
            "angle_deg":   round(float(np.degrees(mdict["angle"])), 2),
            "rms":         round(float(mdict["rms"]), 2),
            "reason":      reason,
        })
    return added


# ══════════════════════════════════════════════════════════════════════════
#  Cluster identification - union-find over merge_log attaches
# ══════════════════════════════════════════════════════════════════════════

def _identify_clusters(fragments: list[dict],
                       merge_log: list[dict],
                       placed: set[int]) -> list[set[int]]:
    """Walk the merge_log and group placed fragments by connected attach
    history. Returns a list of disjoint sets, sorted by descending size
    (the largest cluster first - useful when picking a 'master' cluster).
    """
    if not placed:
        return []
    parent = {i: i for i in placed}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    fid_to_idx = {int(fragments[i]["id"]): i for i in range(len(fragments))}
    attach_phases = {"seed", "mst_attach", "orphan_rescue",
                      "aggressive_orphan_rescue"}
    for entry in merge_log:
        if entry.get("phase") not in attach_phases:
            continue
        anchor_id   = entry.get("anchor")
        attached_id = entry.get("attached")
        if anchor_id is None or attached_id is None:
            continue
        a_idx = fid_to_idx.get(int(anchor_id))
        b_idx = fid_to_idx.get(int(attached_id))
        if a_idx is None or b_idx is None:
            continue
        if a_idx in placed and b_idx in placed:
            union(a_idx, b_idx)

    clusters: dict[int, set[int]] = {}
    for i in placed:
        r = find(i)
        clusters.setdefault(r, set()).add(i)
    return sorted(clusters.values(), key=len, reverse=True)


# ══════════════════════════════════════════════════════════════════════════
#  Cluster reconciliation - merge separately-anchored clusters via
#  cross-cluster pair matches. Each cluster's anchor was placed at
#  identity (= scan position) by aggressive orphan rescue, so the
#  clusters are independent in the canvas. Here we find the best
#  bridging match between two clusters and globally shift one cluster
#  so the bridge is satisfied.
# ══════════════════════════════════════════════════════════════════════════

def _cluster_overlap_check(fragments: list[dict],
                            transforms: dict[int, np.ndarray],
                            cluster_y: set[int],
                            placed_outside_y: set[int],
                            relax_factor: float = 2.5) -> bool:
    """Like _global_overlap_ok but tolerates more overlap during cluster
    reconciliation. Two clusters being bridged share a torn-edge seam, so
    some mask overlap near the bridge is expected (sub-pixel pose drift,
    feathering, sub-pixel mask boundaries) - rejecting that drops every
    cluster bridge. We use a higher overlap-fraction threshold here.
    """
    if not placed_outside_y:
        return True
    pad = 20
    all_idx = list(cluster_y) + list(placed_outside_y)
    mn_x = mn_y = float("inf")
    mx_x = mx_y = float("-inf")
    for i in all_idx:
        x0, y0, x1, y1 = _transformed_bbox(fragments[i], transforms[i])
        mn_x = min(mn_x, x0); mn_y = min(mn_y, y0)
        mx_x = max(mx_x, x1); mx_y = max(mx_y, y1)
    canvas_w = int(mx_x - mn_x) + 2 * pad
    canvas_h = int(mx_y - mn_y) + 2 * pad
    if canvas_w <= 0 or canvas_h <= 0 \
            or canvas_w > cfg.OVERLAP_CANVAS_MAX \
            or canvas_h > cfg.OVERLAP_CANVAS_MAX:
        return True
    ox = -mn_x + pad; oy = -mn_y + pad
    relaxed_thresh = cfg.RECON_OVERLAP_THRESH * relax_factor
    for i in cluster_y:
        mask_i = _warp_mask_to_canvas(
            fragments[i], transforms[i], canvas_w, canvas_h, ox, oy)
        area_i = int(mask_i.sum())
        if area_i < 10:
            continue
        for p in placed_outside_y:
            mask_p = _warp_mask_to_canvas(
                fragments[p], transforms[p], canvas_w, canvas_h, ox, oy)
            inter = int(np.logical_and(mask_i, mask_p).sum())
            smaller = min(area_i, int(mask_p.sum()))
            if smaller > 0 and inter / smaller > relaxed_thresh:
                return False
    return True


def _reconcile_clusters(fragments: list[dict],
                        transforms: dict[int, np.ndarray],
                        placed: set[int], locked: dict[int, float],
                        cache: _MatchCache,
                        merge_log: list[dict]) -> int:
    """Merge separate clusters where a good cross-cluster match exists.

    For every (cluster_X, cluster_Y) pair, search every (fx in X, fy in Y)
    for the lowest-fit-cost cached match, then compute the rigid SE(2)
    transform that aligns cluster_Y so the bridge is satisfied. Reject
    the merge only if it causes a major (>~20%) overlap on any fragment
    pair; minor seam-zone overlap is expected and accepted.

    Returns the number of clusters absorbed.
    """
    n_merges = 0
    cost_ceiling = float(getattr(cfg, "CLUSTER_MERGE_MAX_COST",
                                   cfg.ORPHAN_RESCUE_MAX_COST * 1.5))
    overlap_relax = float(getattr(cfg, "CLUSTER_MERGE_OVERLAP_RELAX", 2.5))
    # Track bridges we've already rejected so we don't loop on them.
    rejected_bridges: set[tuple[int, int]] = set()

    while True:
        clusters = _identify_clusters(fragments, merge_log, placed)
        if len(clusters) <= 1:
            break

        cluster_id: dict[int, int] = {}
        for idx, cl in enumerate(clusters):
            for frag_idx in cl:
                cluster_id[frag_idx] = idx

        # Best cross-cluster bridge among already-cached matches only.
        best: tuple[float, int, int, dict] | None = None
        for (lo, hi), m in cache.cache.items():
            if m is None or lo not in placed or hi not in placed:
                continue
            if cluster_id.get(lo) == cluster_id.get(hi):
                continue
            key = (min(lo, hi), max(lo, hi))
            if key in rejected_bridges:
                continue
            cost = float(m.get("fit_cost", float("inf")))
            if cost > cost_ceiling:
                continue
            if best is None or cost < best[0]:
                best = (cost, lo, hi, m)

        if best is None:
            break

        cost, fx, fy, m = best
        cluster_y = clusters[cluster_id.get(fy, 0)]

        # Compute the global shift that puts fy at its correct global pose
        # relative to fx, then propagate the same shift to every fragment
        # currently in cluster_y (they all share the same anchor frame).
        M_rel = M.affine_from_Rt(m["R"], m["translation"])
        target_T_fy = transforms[fx] @ M_rel
        try:
            current_T_fy_inv = np.linalg.inv(transforms[fy])
        except np.linalg.LinAlgError:
            rejected_bridges.add((min(fx, fy), max(fx, fy)))
            continue
        shift = target_T_fy @ current_T_fy_inv

        shift_dx = float(shift[0, 2])
        shift_dy = float(shift[1, 2])
        max_shift = float(getattr(cfg, "CLUSTER_MERGE_MAX_SHIFT_PX", 5000.0))
        if abs(shift_dx) > max_shift or abs(shift_dy) > max_shift:
            rejected_bridges.add((min(fx, fy), max(fx, fy)))
            continue

        # Save and apply.
        old: dict[int, np.ndarray] = {i: transforms[i].copy()
                                       for i in cluster_y}
        for i in cluster_y:
            transforms[i] = shift @ transforms[i]

        # Validate no MAJOR overlap was introduced. The seam-zone has
        # some inevitable overlap; the relaxed threshold avoids dropping
        # otherwise-correct bridges.
        passed_overlap = _cluster_overlap_check(
                fragments, transforms, cluster_y,
                placed - cluster_y, relax_factor=overlap_relax)
        if not passed_overlap:
            print(f"     bridge {fragments[fx]['id']}->{fragments[fy]['id']} "
                  f"rejected by overlap (cost={cost:.1f}, "
                  f"shift=({shift_dx:.0f},{shift_dy:.0f}))")
            for i in cluster_y:
                transforms[i] = old[i]
            rejected_bridges.add((min(fx, fy), max(fx, fy)))
            continue

        n_merges += 1
        merge_log.append({
            "phase":           "cluster_reconcile",
            "bridge_anchor":   int(fragments[fx]["id"]),
            "bridge_attached": int(fragments[fy]["id"]),
            "absorbed_ids":    sorted(int(fragments[i]["id"])
                                       for i in cluster_y),
            "fit_cost":        round(float(cost), 2),
            "shift_dx":        round(shift_dx, 2),
            "shift_dy":        round(shift_dy, 2),
        })

    return n_merges


# ══════════════════════════════════════════════════════════════════════════
#  Aggressive orphan rescue - try EVERY pair (placed-orphan and
#  orphan-orphan) at relaxed thresholds, and accept the lowest-cost
#  attach that still passes the global overlap check. This is the
#  "leave no fragment behind" pass.
# ══════════════════════════════════════════════════════════════════════════

def _aggressive_orphan_rescue(fragments: list[dict],
                              transforms: dict[int, np.ndarray],
                              placed: set[int], locked: dict[int, float],
                              cache: _MatchCache,
                              merge_log: list[dict],
                              image_rgb: np.ndarray) -> int:
    """
    The legacy orphan rescue only considered (placed_anchor, orphan)
    pairs. That misses the common case where two unplaced fragments are
    each other's true partner but neither has a placed neighbour. Here
    we additionally:
      1. Force a match call between every (orphan, anchor) pair, even if
         the prefilter dropped them (the prefilter is too aggressive on
         hard cases).
      2. Force matches between (orphan, orphan) pairs and place the
         best one as a new anchor cluster, then continue rescuing
         around it.
    Returns the number of fragments newly placed.
    """
    n = len(fragments)
    added = 0
    relaxed_conf = float(getattr(cfg, "ORPHAN_RESCUE_MIN_CONFIDENCE", 0.30))
    relaxed_cost = float(getattr(cfg, "ORPHAN_RESCUE_MAX_COST", 1200.0))

    while True:
        if len(placed) >= n:
            break
        unplaced = [i for i in range(n) if i not in placed]
        if not unplaced:
            break

        # ── Pass 1: orphan → placed-anchor. Force matches even outside the
        #    prefilter survivors, because the prefilter is exactly what
        #    lost these orphans in the first place.
        best_attach: tuple[float, int, int, dict] | None = None
        for j in unplaced:
            for i in placed:
                m = cache.match(i, j)
                if m is None:
                    continue
                cost = _fit_cost(m)
                conf = float(m.get("confidence", 0.0))
                if cost > relaxed_cost or conf < relaxed_conf:
                    continue
                key = (-conf, cost)
                if best_attach is None or key < (best_attach[0], best_attach[1]):
                    best_attach = (key[0], key[1], i, j, m)
        if best_attach is not None:
            _nc, cost, i, j, m = best_attach
            ok, reason = _attach(fragments, transforms, placed, locked, m, i, j,
                                  enforce_seam=False)
            if ok:
                added += 1
                merge_log.append({
                    "phase":       "aggressive_orphan_rescue",
                    "subphase":    "orphan_to_placed",
                    "anchor":      fragments[i]["id"],
                    "attached":    fragments[j]["id"],
                    "confidence":  round(float(m["confidence"]), 3),
                    "fit_cost":    round(float(cost), 2),
                })
                continue

        # ── Pass 2: orphan ↔ orphan - if no orphan is rescuable from a
        #    placed anchor, try to seed a NEW cluster from the best
        #    orphan-orphan pair.
        if len(unplaced) < 2:
            break
        best_pair: tuple[float, int, int, dict] | None = None
        for ix, i in enumerate(unplaced):
            for j in unplaced[ix + 1:]:
                m = cache.match(i, j)
                if m is None:
                    continue
                cost = _fit_cost(m)
                conf = float(m.get("confidence", 0.0))
                if cost > relaxed_cost or conf < relaxed_conf:
                    continue
                key = (-conf, cost)
                if best_pair is None or key < (best_pair[0], best_pair[1]):
                    best_pair = (key[0], key[1], i, j, m)
        if best_pair is None:
            break
        _nc, cost, i, j, m = best_pair
        # Anchor i at identity, attach j on top.
        transforms[i] = np.eye(3, dtype=np.float64)
        placed.add(i)
        locked[i] = max(locked.get(i, 0.0), float(m["confidence"]) * 0.5)
        ok, reason = _attach(fragments, transforms, placed, locked, m, i, j,
                              enforce_seam=False)
        if not ok:
            placed.discard(i)
            transforms[i] = np.eye(3, dtype=np.float64)
            break
        added += 2  # both i and j newly placed
        merge_log.append({
            "phase":       "aggressive_orphan_rescue",
            "subphase":    "orphan_orphan_seed",
            "anchor":      fragments[i]["id"],
            "attached":    fragments[j]["id"],
            "confidence":  round(float(m["confidence"]), 3),
            "fit_cost":    round(float(cost), 2),
        })
    return added


# ══════════════════════════════════════════════════════════════════════════
#  Conflict resolution (max-weight bipartite matching, unchanged semantics)
# ══════════════════════════════════════════════════════════════════════════

def _resolve_conflicts(candidate_anchors: dict[int, list[tuple[int, dict]]]
                       ) -> dict[int, tuple[int, dict]]:
    G = nx.Graph()
    for j, opts in candidate_anchors.items():
        for i, mdict in opts:
            G.add_edge(("j", j), ("i", i),
                       weight=float(mdict["confidence"]))
    if G.number_of_edges() == 0:
        return {}
    matching = nx.max_weight_matching(G, maxcardinality=True)
    out: dict[int, tuple[int, dict]] = {}
    for a, b in matching:
        if a[0] == "j" and b[0] == "i":
            j, i = a[1], b[1]
        elif a[0] == "i" and b[0] == "j":
            i, j = a[1], b[1]
        else:
            continue
        for ii, mdict in candidate_anchors[j]:
            if ii == i:
                out[i] = (j, mdict)
                break
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Debug snapshot (same format the legacy module wrote — frontend reads it)
# ══════════════════════════════════════════════════════════════════════════

def _save_step(image_rgb: np.ndarray, fragments: list[dict],
               transforms: dict[int, np.ndarray], placed: set[int],
               step_idx: int, tag: str, debug_dir: Path) -> None:
    if not placed:
        return
    mn_x, mn_y, mx_x, mx_y = _global_frame_bounds(fragments, transforms, placed)
    pad = 30
    cw = min(int(mx_x - mn_x) + 2 * pad, 8000)
    ch = min(int(mx_y - mn_y) + 2 * pad, 8000)
    if cw < 1 or ch < 1:
        return
    ox = -mn_x + pad; oy = -mn_y + pad
    canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 200
    for i in sorted(placed):
        frag = fragments[i]
        Mg = transforms[i].copy()
        Mg[0, 2] += ox; Mg[1, 2] += oy
        bx, by, bw, bh = frag["bbox"]
        sub_mask = frag["mask"][by:by + bh, bx:bx + bw]
        sub_img = np.zeros((bh, bw, 3), dtype=np.uint8)
        mk = sub_mask > 127
        sub_img[mk] = image_rgb[by:by + bh, bx:bx + bw][mk]
        origin = Mg @ np.array([bx, by, 1.0], dtype=np.float64)
        M_sub = Mg.copy()
        M_sub[0, 2] = origin[0]; M_sub[1, 2] = origin[1]
        M_2x3 = M_sub[:2, :].astype(np.float32)
        warped_img = cv2.warpAffine(
            sub_img, M_2x3, (cw, ch),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        warped_mask = cv2.warpAffine(
            sub_mask, M_2x3, (cw, ch),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mw = warped_mask > 127
        canvas[mw] = warped_img[mw]
    save_image(canvas, str(debug_dir / f"step_{step_idx:02d}_{tag}.png"))


# ══════════════════════════════════════════════════════════════════════════
#  Main entry — layout-agnostic MST-growth
# ══════════════════════════════════════════════════════════════════════════

def reconstruct(fragments: list[dict],
                image_rgb: np.ndarray,
                debug_dir: Path) -> dict[int, np.ndarray]:
    """
    Assemble fragments into a global pose graph by confidence-ranked
    MST-growth. Returns `{fragment_index: 3x3 affine in image frame}`.
    Unplaced fragments keep the identity transform.
    """
    recon_debug = debug_dir / "reconstruction"
    recon_debug.mkdir(parents=True, exist_ok=True)

    n = len(fragments)
    transforms: dict[int, np.ndarray] = {
        i: np.eye(3, dtype=np.float64) for i in range(n)}
    placed: set[int] = set()
    locked: dict[int, float] = {}
    merge_log: list[dict] = []

    if n < 2:
        print("  -- Only one fragment; nothing to assemble.")
        _write_assembly_artifacts(
            fragments=fragments, transforms=transforms,
            placed=placed, merge_log=merge_log, cache=None,
            recon_debug=recon_debug, status="trivial_single_fragment",
        )
        return transforms

    # ── Feature preparation ──────────────────────────────────────────────
    print("  -- Preparing edges + SDTs + paper-color fingerprints --")
    t0 = time.time()
    M.prepare_edges_and_sdt(fragments, image_rgb=image_rgb)
    print(f"     done ({time.time()-t0:.2f}s)")

    if cfg.DINOV2_ENABLED:
        print("  -- Extracting DINOv2 dense features --")
        t0 = time.time()
        from .appearance import attach_dinov2_features_all
        attach_dinov2_features_all(fragments, image_rgb)
        print(f"     done ({time.time()-t0:.2f}s)")

    cache = _MatchCache(fragments, image_rgb)

    # ── Phase A0: per-fragment profiles (individual analysis) ───────────
    print("  -- Building per-fragment profiles --")
    t0 = time.time()
    profiles = fp.build_fragment_profiles(fragments, image_rgb)
    fp.save_profiles(profiles, recon_debug)
    fp.print_profile_summary(profiles)
    print(f"     done ({time.time()-t0:.2f}s)")

    # ── Phase A: enumerate candidate pairs and score each ───────────────
    print("  -- Enumerating candidate edge pairs --")
    t0 = time.time()
    candidates = _enumerate_pair_candidates(fragments, image_rgb)
    # Backup pass: if the prefilter killed too many pairs (every fragment
    # should have at least 2-3 candidates so the matcher can choose), fall
    # back to "all torn vs torn" so the MST has options. The matcher's own
    # five gates still drop bad pairs - the prefilter is just a speed
    # heuristic, not a correctness gate.
    min_cand_per_frag = float(getattr(
        cfg, "ASSEMBLY_MIN_CANDIDATES_PER_FRAG", 3.0))
    if len(candidates) < min_cand_per_frag * n:
        print(f"     prefilter kept only {len(candidates)} pairs for "
              f"{n} fragments; expanding to all-torn-pairs.")
        all_pairs: list[tuple[int, int]] = []
        for i in range(n):
            if not any(e.get("is_torn") for e in fragments[i].get("edges", [])):
                continue
            for j in range(i + 1, n):
                if not any(e.get("is_torn") for e in fragments[j].get("edges", [])):
                    continue
                all_pairs.append((i, j))
        # Keep prefilter survivors first (they're the cheapest) then append
        # the rest deduplicated.
        seen = set(candidates)
        for (i, j) in all_pairs:
            if (i, j) not in seen:
                candidates.append((i, j))
                seen.add((i, j))
        # Cap so a 100-fragment scene doesn't blow up.
        cap = int(cfg.ASSEMBLY_MAX_CANDIDATE_PAIRS)
        if len(candidates) > cap:
            candidates = candidates[:cap]
    print(f"     {len(candidates)} candidate pairs to score "
          f"({time.time()-t0:.2f}s)")

    print("  -- Scoring candidate pairs --")
    t0 = time.time()
    ranked: list[tuple[float, int, int]] = []
    for (i, j) in candidates:
        m = cache.match(i, j)
        if m is None:
            continue
        if float(m["confidence"]) < cfg.ASSEMBLY_MIN_CONFIDENCE:
            continue
        ranked.append((float(m["confidence"]), i, j))
    ranked.sort(reverse=True)
    print(f"     {len(ranked)} pairs >= min confidence "
          f"({time.time()-t0:.2f}s)")

    # ── Phase A2: per-edge partner rankings + mutual-rank seed list ─────
    print("  -- Ranking partners per edge (mutual-best seed selection) --")
    t0 = time.time()
    rankings = er.rank_edges_per_fragment(fragments, candidates, cache,
                                            top_k=8)
    mutual = er.build_pair_mutual_scores(rankings, cache, fragments)
    seeds = er.ranked_seed_candidates(mutual, profiles=profiles, top_n=12)
    er.save_rankings(rankings, mutual, seeds, recon_debug)
    er.print_seed_summary(seeds, n=5)
    print(f"     {len(rankings)} edges ranked, "
          f"{len(mutual)} mutual-rank pairs ({time.time()-t0:.2f}s)")

    if not ranked and not seeds:
        print("  -- No candidate pair above the confidence floor. "
              "Returning identity placement.")
        _write_assembly_artifacts(
            fragments=fragments, transforms=transforms,
            placed=placed, merge_log=merge_log, cache=cache,
            recon_debug=recon_debug,
            n_candidates=len(candidates),
            n_ranked=len(ranked), n_mutual=len(mutual),
            n_seeds=len(seeds),
            status="no_candidate_above_min_confidence",
        )
        _finalize_release_dinov2(fragments)
        return transforms

    # ── Phase B: seed by mutual-rank, fall back to absolute confidence ──
    seeded = False
    conf0 = 0.0
    i0 = j0 = -1
    m0 = None
    seed_id = -1
    seed_source = None

    # Try mutual-best seeds in order; first one that attaches wins.
    for s in seeds:
        i_cand, j_cand = int(s["frag_a"]), int(s["frag_b"])
        m_cand = cache.match(i_cand, j_cand)
        if m_cand is None:
            continue
        if float(m_cand.get("confidence", 0.0)) < cfg.ASSEMBLY_MIN_CONFIDENCE:
            continue
        placed.clear()
        transforms[i_cand] = np.eye(3, dtype=np.float64)
        placed.add(i_cand)
        locked[i_cand] = 1.0
        ok, reason = _attach(fragments, transforms, placed, locked,
                              m_cand, i_cand, j_cand, enforce_seam=True)
        if ok:
            seeded = True
            seed_id = i_cand
            i0, j0, m0 = i_cand, j_cand, m_cand
            conf0 = float(m_cand["confidence"])
            seed_source = "mutual_rank"
            break
        # roll back
        placed.discard(i_cand)
        transforms[i_cand] = np.eye(3, dtype=np.float64)

    # Fall back to the legacy "highest absolute confidence" seed if no
    # mutual-best was attachable.
    if not seeded and ranked:
        for (conf_alt, i_alt, j_alt) in ranked:
            placed.clear()
            transforms[i_alt] = np.eye(3)
            placed.add(i_alt)
            locked[i_alt] = 1.0
            m_alt = cache.match(i_alt, j_alt)
            if m_alt is None:
                continue
            ok, reason = _attach(fragments, transforms, placed, locked, m_alt,
                                  i_alt, j_alt, enforce_seam=True)
            if ok:
                seed_id = i_alt
                i0, j0, m0 = i_alt, j_alt, m_alt
                conf0 = conf_alt
                seeded = True
                seed_source = "absolute_confidence"
                break

    if not seeded:
        print("  -- Could not seed any pair (mutual-rank or absolute). "
              "Returning identity placement.")
        _write_assembly_artifacts(
            fragments=fragments, transforms=transforms,
            placed=placed, merge_log=merge_log, cache=cache,
            recon_debug=recon_debug,
            n_candidates=len(candidates),
            n_ranked=len(ranked), n_mutual=len(mutual),
            n_seeds=len(seeds),
            status="no_seed_passed_attach_gate",
        )
        _finalize_release_dinov2(fragments)
        return transforms

    print(f"  -- Seed [{seed_source}]: frag {fragments[i0]['id']} + "
          f"{fragments[j0]['id']} @ confidence {conf0:.2f}")
    merge_log.append({"phase":      "seed",
                      "source":     seed_source,
                      "anchor":     fragments[i0]["id"],
                      "attached":   fragments[j0]["id"],
                      "confidence": round(conf0, 3)})
    _save_step(image_rgb, fragments, transforms, placed,
               0, "seed", recon_debug)

    # ── Phase C: MST growth ─────────────────────────────────────────────
    # Mutual-rank lookup: prefer candidates where each side ranks the other
    # near the top. Candidates outside the top-K mutual list still get a
    # chance, but they're penalised so a "mutual #1" wins over a higher-
    # absolute-confidence "asymmetric" pair.
    mutual_lookup: dict[tuple[int, int], dict] = mutual
    mutual_top = max(1, int(getattr(cfg, "ASSEMBLY_MUTUAL_TOP_K", 3)))

    def _pair_score(i_p: int, j_f: int, m: dict) -> float:
        """Combined ranking score = absolute confidence + mutual bonus."""
        conf = float(m.get("confidence", 0.0))
        key = (min(i_p, j_f), max(i_p, j_f))
        mu = mutual_lookup.get(key)
        if mu is None:
            return conf
        # Reward both sides ranking the other near the top.
        ra = int(mu.get("rank_a_in_b", 99))
        rb = int(mu.get("rank_b_in_a", 99))
        bonus = 0.0
        if ra <= mutual_top and rb <= mutual_top:
            bonus = 0.20  # strong mutual support
        elif ra <= mutual_top or rb <= mutual_top:
            bonus = 0.05  # one-sided top match
        return conf + bonus

    print("  -- Growing MST by mutual-rank-weighted confidence --")
    step = 0
    consecutive_seam_failures = 0
    while step < cfg.ASSEMBLY_MAX_STEPS:
        # Collect attachments touching the placed set
        per_free: dict[int, list[tuple[float, int, dict]]] = {}
        for (conf, i, j) in ranked:
            i_in, j_in = (i in placed), (j in placed)
            if i_in == j_in:
                continue     # both placed or both free
            i_p, j_f = (i, j) if i_in else (j, i)
            m = cache.match(i_p, j_f)
            if m is None:
                continue
            if float(m["confidence"]) < cfg.ASSEMBLY_MIN_CONFIDENCE:
                continue
            score = _pair_score(i_p, j_f, m)
            per_free.setdefault(j_f, []).append((score, i_p, m))

        if not per_free:
            break

        # For conflict resolution we need a (j -> [(anchor, match)]) shape
        # ordered by score; anchors win on score.
        per_free_sorted: dict[int, list[tuple[int, dict]]] = {}
        for j_f, opts in per_free.items():
            opts.sort(key=lambda x: -x[0])
            per_free_sorted[j_f] = [(i_p, m) for (_s, i_p, m) in opts]

        # Resolve conflicts — a single anchor can attract multiple free
        # fragments; max-weight matching picks the globally best assignment.
        resolved = _resolve_conflicts(per_free_sorted)

        if not resolved:
            # No matching fired (e.g., single edge) — greedy pick the best.
            best: tuple[float, int, int, dict] | None = None
            for j_f, opts in per_free.items():
                for (score, i_p, m) in opts:
                    if best is None or score > best[0]:
                        best = (score, i_p, j_f, m)
            if best is None:
                break
            _c, i_p, j_f, m = best
            resolved = {i_p: (j_f, m)}

        attached_this_round = 0
        for (i_p, (j_f, m)) in resolved.items():
            if j_f in placed:
                continue
            ok, reason = _attach(fragments, transforms, placed, locked,
                                  m, i_p, j_f, enforce_seam=True)
            if not ok:
                # Track but don't fail the round; the same pair won't be
                # retried because nothing about the placed set changed for
                # them - they fall out of the next iteration's per_free.
                merge_log.append({
                    "phase":     "mst_attach_rejected",
                    "anchor":    fragments[i_p]["id"],
                    "candidate": fragments[j_f]["id"],
                    "reason":    reason,
                    "fit_cost":  round(float(m.get("fit_cost", -1.0)), 2),
                })
                consecutive_seam_failures += 1
                continue
            consecutive_seam_failures = 0
            attached_this_round += 1
            step += 1
            mu_info = mutual_lookup.get(
                (min(i_p, j_f), max(i_p, j_f)))
            merge_log.append({
                "phase":       "mst_attach",
                "step":        step,
                "anchor":      fragments[i_p]["id"],
                "attached":    fragments[j_f]["id"],
                "confidence":  round(float(m["confidence"]), 3),
                "angle_deg":   round(float(np.degrees(m["angle"])), 2),
                "rms":         round(float(m["rms"]), 2),
                "mutual_rank_a": (int(mu_info["rank_a_in_b"])
                                   if mu_info else None),
                "mutual_rank_b": (int(mu_info["rank_b_in_a"])
                                   if mu_info else None),
            })
            # Note: prior versions of this loop ran a periodic
            # `_maybe_global_text_rotation` here to undo accumulated drift.
            # Step 3 makes per-fragment text orientation a Procrustes seed
            # at match time, so the matcher already places each fragment
            # consistent with its local text — there's no global drift to
            # cancel and rotating the cluster post-hoc would only break
            # seams that ICP just tightened.

        if attached_this_round == 0:
            break

    _save_step(image_rgb, fragments, transforms, placed,
               max(1, step), "mst_done", recon_debug)

    print(f"     MST grew to {len(placed)}/{n} placed in {step} steps")

    # ── Phase C2: seam-contact refinement ────────────────────────────────
    # Tighten every adjacent placed pair under the (gap + overlap +
    # coverage) cost. This converts the "MST-good-enough" pose into a
    # pixel-tight pose before BA jointly polishes the graph.
    print("  -- Seam-contact refinement (Step 4) --")
    n_refined = _refine_seams_from_log(
        fragments, transforms, placed, cache, merge_log)
    print(f"     refined {n_refined} attaches")
    if n_refined:
        _save_step(image_rgb, fragments, transforms, placed,
                   max(1, step), "seam_refined", recon_debug)

    # ── Phase D: bundle adjustment (pin only the seed) ──────────────────
    print("  -- Bundle adjustment (seed-pinned) --")
    _bundle_adjust_poses(fragments, transforms, placed, locked,
                         pinned={seed_id},
                         cache=cache, merge_log=merge_log)
    _save_step(image_rgb, fragments, transforms, placed,
               max(1, step) + 1, "bundle_adjust", recon_debug)

    # ── Phase E1: standard orphan rescue at relaxed confidence ──────────
    if len(placed) < n:
        print("  -- Orphan rescue at relaxed confidence --")
        n_rescued = _orphan_rescue(
            fragments, transforms, placed, locked, cache, merge_log,
            min_confidence=cfg.ASSEMBLY_ORPHAN_MIN_CONFIDENCE)
        print(f"     rescued {n_rescued}")
        _save_step(image_rgb, fragments, transforms, placed,
                   max(1, step) + 2, "orphan_rescue", recon_debug)

    # ── Phase E2: aggressive rescue (force-match orphans, allow new clusters) ─
    if (len(placed) < n and
            getattr(cfg, "ASSEMBLY_AGGRESSIVE_ORPHAN_RESCUE", True)):
        print("  -- Aggressive orphan rescue (force-match remaining) --")
        n_aggressive = _aggressive_orphan_rescue(
            fragments, transforms, placed, locked, cache, merge_log, image_rgb)
        print(f"     aggressively placed {n_aggressive}")
        _save_step(image_rgb, fragments, transforms, placed,
                   max(1, step) + 3, "aggressive_rescue", recon_debug)

    # Refine seams again — orphan + aggressive rescue may have added
    # attaches that didn't exist when the post-MST refinement ran.
    if len(placed) > 1:
        n_refined2 = _refine_seams_from_log(
            fragments, transforms, placed, cache, merge_log)
        if n_refined2:
            print(f"     post-rescue seam refinement: {n_refined2} attaches")

    # ── Phase F: cluster reconciliation - merge separate clusters by
    #    finding the lowest-cost cross-cluster pair match and globally
    #    shifting one cluster so the bridge is satisfied.
    if getattr(cfg, "ASSEMBLY_CLUSTER_RECONCILE", True):
        clusters_before = _identify_clusters(fragments, merge_log, placed)
        if len(clusters_before) > 1:
            print(f"  -- Cluster reconciliation ({len(clusters_before)} "
                  f"clusters detected) --")
            n_reconciled = _reconcile_clusters(
                fragments, transforms, placed, locked, cache, merge_log)
            clusters_after = _identify_clusters(fragments, merge_log, placed)
            print(f"     reconciled {n_reconciled} bridges, "
                  f"{len(clusters_after)} cluster(s) remaining")
            _save_step(image_rgb, fragments, transforms, placed,
                       max(1, step) + 4, "cluster_reconcile", recon_debug)
            # Re-run bundle adjustment now that the cluster graph is unified -
            # the cross-cluster bridges are new pose-graph constraints we
            # can refine over.
            print("  -- Bundle adjustment (post-reconcile) --")
            _bundle_adjust_poses(fragments, transforms, placed, locked,
                                 pinned={seed_id},
                                 cache=cache, merge_log=merge_log)

    # The cosmetic post-placement text-rotation passes
    # (_apply_per_fragment_text_rotation, _maybe_global_text_rotation) were
    # removed in Step 3. Per-fragment text orientation is now a Procrustes
    # seed at match time (matching._match_edge_pair); rotating fragments
    # post-placement would invalidate the seam refinement.
    _save_step(image_rgb, fragments, transforms, placed,
               max(1, step) + 4, "final", recon_debug)

    # ── Logs and VRAM cleanup ───────────────────────────────────────────
    _write_assembly_artifacts(
        fragments=fragments,
        transforms=transforms,
        placed=placed,
        merge_log=merge_log,
        cache=cache,
        recon_debug=recon_debug,
        n_candidates=len(candidates),
        n_ranked=len(ranked),
        n_mutual=len(mutual),
        n_seeds=len(seeds),
        seed_id=seed_id,
        seed_source=seed_source,
        seed_conf=conf0,
    )

    _finalize_release_dinov2(fragments)
    print(f"\n  Assembly complete: {len(placed)}/{n} placed, "
          f"{cache.n_calls} matcher calls / {cache.n_hits} cache hits")
    return transforms


def _write_assembly_artifacts(*,
                               fragments: list[dict],
                               transforms: dict[int, np.ndarray],
                               placed: set[int],
                               merge_log: list[dict],
                               cache: "_MatchCache | None",
                               recon_debug: Path,
                               n_candidates: int = 0,
                               n_ranked: int = 0,
                               n_mutual: int = 0,
                               n_seeds: int = 0,
                               seed_id: int = -1,
                               seed_source: str | None = None,
                               seed_conf: float = 0.0,
                               status: str | None = None) -> None:
    """Persist the canonical assembly outputs.

    Always writes:
      * merge_log.json
      * assembly_summary.json
      * final_translations.json — {dx, dy, angle_deg, placed} per fragment
      * final_transforms.json   — full 3x3 SE(2) affine per fragment

    Called from every reconstruct() return path so downstream consumers
    (composition, benchmark, backend) see a consistent artifact set even
    when the MST couldn't seed.
    """
    n = len(fragments)
    seed_fragment = (int(fragments[seed_id]["id"])
                      if 0 <= seed_id < n else None)

    with open(recon_debug / "merge_log.json", "w", encoding="utf-8") as fh:
        json.dump(merge_log, fh, indent=2)
    summary = {
        "n_fragments":          n,
        "n_placed":             len(placed),
        "n_candidate_pairs":    int(n_candidates),
        "n_ranked_pairs":       int(n_ranked),
        "n_mutual_pairs":       int(n_mutual),
        "n_seed_candidates":    int(n_seeds),
        "seed_fragment":        seed_fragment,
        "seed_source":          seed_source,
        "seed_confidence":      round(float(seed_conf), 4),
        "match_cache_calls":    int(cache.n_calls) if cache is not None else 0,
        "match_cache_hits":     int(cache.n_hits)  if cache is not None else 0,
        "fragments_unplaced":   sorted(int(fragments[i]["id"])
                                        for i in range(n) if i not in placed),
    }
    if status is not None:
        summary["status"] = status
    with open(recon_debug / "assembly_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    final_translations: dict[str, dict] = {}
    final_transforms: dict[str, list[list[float]]] = {}
    for i in range(n):
        fid = int(fragments[i].get("id", i))
        T = transforms[i]
        angle_rad = float(np.arctan2(T[1, 0], T[0, 0]))
        final_translations[str(fid)] = {
            "dx":        round(float(T[0, 2]), 2),
            "dy":        round(float(T[1, 2]), 2),
            "angle_deg": round(float(np.degrees(angle_rad)), 3),
            "placed":    bool(i in placed),
        }
        final_transforms[str(fid)] = [[round(float(T[r, c]), 6)
                                        for c in range(3)]
                                       for r in range(3)]
    with open(recon_debug / "final_translations.json", "w", encoding="utf-8") as fh:
        json.dump(final_translations, fh, indent=2)
    with open(recon_debug / "final_transforms.json", "w", encoding="utf-8") as fh:
        json.dump(final_transforms, fh, indent=2)


def _finalize_release_dinov2(fragments: list[dict]) -> None:
    """Release DINOv2 VRAM back to the pipeline so LaMa can claim it."""
    try:
        from .appearance import DINOv2Extractor
        DINOv2Extractor.release()
    except Exception:
        pass
    for f in fragments:
        f.pop("dinov2", None)


__all__ = ["reconstruct"]
