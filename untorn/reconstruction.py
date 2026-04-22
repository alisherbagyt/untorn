"""
untorn.reconstruction
=====================
Hierarchical proximity-based document reconstruction.

Strategy
--------
Pieces are provided in a roughly correct initial scan layout, so we
anchor the reconstruction at the four document corners and grow each
cluster inward. This keeps every merge local (small cumulative error)
and avoids the global exhaustive search a traditional jigsaw solver
would need.

Phases implemented here
-----------------------
  Phase II  : Corner seed batching (horizontal pivots) — 4 high-
              confidence 2-piece clusters at TL / TR / BL / BR.
  Phase III : Vertical expansion — one new piece below (TL/TR) or
              above (BL/BR) each seed, yielding 4 L-shaped 3-piece
              clusters.
  Phase IV  : Perimeter frame infilling — walk each edge of the
              document between adjacent corner clusters, attaching
              any perimeter pieces on the way.
  Phase V   : Interior infilling — confidence-first priority queue
              over all remaining candidate matches. Pieces with
              confidence >= HIGH_CONFIDENCE_LOCK are locked immediately
              and their own neighbors enter the queue. Low-confidence
              pieces are handled last via force-fit, constrained by
              the global document boundary (A4/Letter ratio sanity
              check) so blank/textureless pieces drop into the only
              remaining slots.

Conflict resolution
-------------------
When two un-placed fragments compete for the same neighbor slot we
resolve via weighted bipartite matching on the competing candidates
(max_weight_matching from networkx). The chosen assignment is the one
that maximizes total confidence while preserving the global layout.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import networkx as nx

from . import config as cfg
from .io_utils import save_image
from . import matching as M
from .neighbors import discover_neighbors, identify_corners


def _fit_cost(match: dict | None) -> float:
    """Extract the physical fit cost from a match dict; infinity on miss."""
    if match is None:
        return float("inf")
    return float(match.get("fit_cost", float("inf")))


# ══════════════════════════════════════════════════════════════════════════
#  Helpers: SE(2) composition, bbox/overlap in the global canvas frame
# ══════════════════════════════════════════════════════════════════════════

def _transform_contour(frag: dict, M_global: np.ndarray) -> np.ndarray:
    """Apply the global transform to frag's contour points (Nx2)."""
    pts = frag["contour"].astype(np.float64).reshape(-1, 2)
    return M.affine_apply(M_global, pts)


def _transformed_bbox(frag: dict, M_global: np.ndarray
                      ) -> tuple[float, float, float, float]:
    """Return the AABB (xmin, ymin, xmax, ymax) of frag under M_global."""
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
    """AABB of all currently-placed fragments in the global frame."""
    if not placed:
        return 0.0, 0.0, 0.0, 0.0
    mn_x = mn_y = float("inf")
    mx_x = mx_y = float("-inf")
    for i in placed:
        x0, y0, x1, y1 = _transformed_bbox(fragments[i], transforms[i])
        mn_x = min(mn_x, x0); mn_y = min(mn_y, y0)
        mx_x = max(mx_x, x1); mx_y = max(mx_y, y1)
    return mn_x, mn_y, mx_x, mx_y


# ── Global overlap check ─────────────────────────────────────────────────
# We warp each candidate's mask into the global canvas frame (anchored
# on the current AABB of placed pieces) and compare pixel intersection
# against every placed fragment's warped mask. Expensive but only runs
# once per accepted candidate — amortized over the whole pipeline this
# is cheap compared to matching itself.

def _warp_mask_to_canvas(frag: dict, M_global: np.ndarray,
                          canvas_w: int, canvas_h: int,
                          ox: float, oy: float) -> np.ndarray:
    """
    Warp fragment's mask into a canvas of size (canvas_h, canvas_w).
    The canvas is positioned so that image-frame origin (0, 0) maps to
    (ox, oy) on the canvas.
    """
    M_shifted = M_global.copy()
    M_shifted[0, 2] += ox
    M_shifted[1, 2] += oy
    # Adjust for the bbox sub-region origin (same trick as composition.py)
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
    """
    True iff candidate `new_idx` does not overlap too much with any
    already-placed fragment (threshold cfg.RECON_OVERLAP_THRESH).
    """
    if not placed:
        return True

    # Dynamic canvas anchored on the union AABB of placed + candidate
    mn_x, mn_y, mx_x, mx_y = _global_frame_bounds(
        fragments, transforms, placed | {new_idx})
    pad = 20
    canvas_w = int(mx_x - mn_x) + 2 * pad
    canvas_h = int(mx_y - mn_y) + 2 * pad
    if canvas_w <= 0 or canvas_h <= 0 \
            or canvas_w > cfg.OVERLAP_CANVAS_MAX \
            or canvas_h > cfg.OVERLAP_CANVAS_MAX:
        # Pathological — be conservative and accept; the SDT gate already
        # rejected most bad alignments upstream.
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
#  Pair-match cache — every (i, j) is matched at most once
# ══════════════════════════════════════════════════════════════════════════

class _MatchCache:
    """
    match_pair can be expensive (SW + ICP). We call it lazily as fragments
    become relevant and memoize (i, j) -> best match dict.
    """

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
            # Return a shallow copy so callers can tag direction without
            # mutating the cache.
            out = dict(cached)
            out["direction_hint"] = direction
            # If this query is from j -> i perspective, invert the transform
            # so we always return "map j onto i".
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
        # Return a properly oriented copy
        return self.match(i, j, direction)


# ══════════════════════════════════════════════════════════════════════════
#  Placement primitive: "attach j to i", respecting i's current global pose
# ══════════════════════════════════════════════════════════════════════════

def _attach(fragments: list[dict],
             transforms: dict[int, np.ndarray],
             placed: set[int], locked: dict[int, float],
             match: dict, i: int, j: int) -> bool:
    """
    Compose transforms so fragment j snaps onto fragment i:

        match["R"], match["t"]  define the LOCAL pose:
            p_j_in_i = R @ p_j + t    (in fragment A's local coords)

        If i has global pose G_i then
            G_j  =  G_i @ [R | t]

    Returns True if the attachment survives the global overlap check.
    """
    M_rel = M.affine_from_Rt(match["R"], match["translation"])
    M_j_global = transforms[i] @ M_rel
    old = transforms.get(j, np.eye(3)).copy()
    transforms[j] = M_j_global

    if not _global_overlap_ok(fragments, transforms, j, placed):
        transforms[j] = old
        return False

    placed.add(j)
    locked[j] = max(locked.get(j, 0.0), match["confidence"])
    return True


# ══════════════════════════════════════════════════════════════════════════
#  Phase II  – Corner seed batching (horizontal pivots)
# ══════════════════════════════════════════════════════════════════════════

_CORNER_HORIZONTAL_DIR = {"TL": "right", "TR": "left",
                          "BL": "right", "BR": "left"}
_CORNER_VERTICAL_DIR   = {"TL": "down",  "TR": "down",
                          "BL": "up",    "BR": "up"}


def _seed_corner(corner_name: str, corner_idx: int,
                  fragments: list[dict],
                  transforms: dict[int, np.ndarray],
                  placed: set[int], locked: dict[int, float],
                  cache: _MatchCache,
                  merge_log: list[dict]) -> int | None:
    """
    Place the corner piece at identity (its current scan pose) and
    connect its horizontal neighbor. Returns the index of the partner
    fragment if the seed succeeds, else None.
    """
    frag = fragments[corner_idx]
    want_dir = _CORNER_HORIZONTAL_DIR[corner_name]

    # Anchor the corner at identity — this fixes the global frame for
    # this cluster.  Other clusters will be reconciled in Phase IV.
    transforms[corner_idx] = np.eye(3, dtype=np.float64)
    placed.add(corner_idx)
    locked[corner_idx] = 1.0   # corner is the trusted anchor

    # Pick the best horizontal neighbor by full-edge physical fit cost
    # (lower = tighter touch, less overlap, more coverage). Prefer
    # neighbors whose `direction` field matches `want_dir`, but fall back
    # to any-direction neighbor if none are directionally tagged.
    candidates: list[tuple[float, int, dict]] = []
    for nb in frag["neighbors"]:
        j = nb["j"]
        if j == corner_idx or j in placed:
            continue
        if nb["direction"] != want_dir:
            continue
        mdict = cache.match(corner_idx, j, direction=nb["direction"])
        if mdict is None:
            continue
        candidates.append((_fit_cost(mdict), j, mdict))

    if not candidates:
        # Relaxed fallback: try any direction
        for nb in frag["neighbors"]:
            j = nb["j"]
            if j == corner_idx or j in placed:
                continue
            mdict = cache.match(corner_idx, j, direction=nb["direction"])
            if mdict is None:
                continue
            candidates.append((_fit_cost(mdict), j, mdict))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])    # lowest fit cost first

    for cost, j, mdict in candidates:
        conf = mdict["confidence"]
        ok = _attach(fragments, transforms, placed, locked, mdict, corner_idx, j)
        if not ok:
            continue
        merge_log.append({
            "phase":       "II_corner_seed",
            "corner":      corner_name,
            "anchor":      fragments[corner_idx]["id"],
            "attached":    fragments[j]["id"],
            "direction":   want_dir,
            "confidence":  round(conf, 3),
            "fit_cost":    round(cost, 2),
            "fit_coverage": round(mdict.get("fit_coverage", 0.0), 3),
            "fit_gap_px":  round(mdict.get("fit_gap_px", 0.0), 2),
            "angle_deg":   round(np.degrees(mdict["angle"]), 2),
            "rms":         round(mdict["rms"], 2),
        })
        return j
    return None


# ══════════════════════════════════════════════════════════════════════════
#  Phase III  – Vertical expansion to L-shaped clusters
# ══════════════════════════════════════════════════════════════════════════

def _expand_vertical(corner_name: str, corner_idx: int,
                      fragments: list[dict],
                      transforms: dict[int, np.ndarray],
                      placed: set[int], locked: dict[int, float],
                      cache: _MatchCache,
                      merge_log: list[dict]) -> int | None:
    """
    Attach the vertical neighbor below (TL/TR) or above (BL/BR) of the
    corner piece, forming an L-shaped 3-piece cluster.
    """
    frag = fragments[corner_idx]
    want_dir = _CORNER_VERTICAL_DIR[corner_name]

    candidates: list[tuple[float, int, dict]] = []
    for nb in frag["neighbors"]:
        j = nb["j"]
        if j == corner_idx or j in placed:
            continue
        if nb["direction"] != want_dir:
            continue
        mdict = cache.match(corner_idx, j, direction=nb["direction"])
        if mdict is None:
            continue
        candidates.append((_fit_cost(mdict), j, mdict))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])   # lowest fit cost first

    for cost, j, mdict in candidates:
        conf = mdict["confidence"]
        ok = _attach(fragments, transforms, placed, locked, mdict,
                      corner_idx, j)
        if not ok:
            continue
        merge_log.append({
            "phase":       "III_vertical_expand",
            "corner":      corner_name,
            "anchor":      fragments[corner_idx]["id"],
            "attached":    fragments[j]["id"],
            "direction":   want_dir,
            "confidence":  round(conf, 3),
            "fit_cost":    round(cost, 2),
            "fit_coverage": round(mdict.get("fit_coverage", 0.0), 3),
            "fit_gap_px":  round(mdict.get("fit_gap_px", 0.0), 2),
            "angle_deg":   round(np.degrees(mdict["angle"]), 2),
            "rms":         round(mdict["rms"], 2),
        })
        return j
    return None


# ══════════════════════════════════════════════════════════════════════════
#  Phase IV  – Perimeter frame infilling
# ══════════════════════════════════════════════════════════════════════════

def _has_factory_edge(frag: dict) -> bool:
    """A perimeter piece has at least one factory (straight) edge."""
    return any(not e["is_torn"] for e in frag["edges"])


def _is_perimeter_piece(frag_idx: int, fragments: list[dict]) -> bool:
    """Heuristic: a perimeter piece owns a straight edge that's long."""
    f = fragments[frag_idx]
    long_factory = [e for e in f["edges"]
                    if not e["is_torn"]
                    and e["length"] > cfg.PERIMETER_MIN_FACTORY_PX]
    return len(long_factory) > 0


def _infill_perimeter(fragments: list[dict],
                       transforms: dict[int, np.ndarray],
                       placed: set[int], locked: dict[int, float],
                       cache: _MatchCache,
                       merge_log: list[dict]) -> int:
    """
    Walk the perimeter of the document by attaching pieces that (a) own
    at least one factory edge, and (b) have a high-confidence match to
    an already-placed neighbor. Stop when no further perimeter attachment
    can be made.

    Returns the number of pieces attached in this phase.
    """
    added = 0
    changed = True
    while changed:
        changed = False
        # For every un-placed perimeter piece, find its best anchor
        # (lowest fit_cost among already-placed neighbors).
        best_for: dict[int, tuple[float, int, dict]] = {}
        for j in range(len(fragments)):
            if j in placed:
                continue
            if not _is_perimeter_piece(j, fragments):
                continue
            for nb in fragments[j]["neighbors"]:
                i = nb["j"]
                if i not in placed:
                    continue
                mdict = cache.match(i, j, direction=nb["direction"])
                if mdict is None:
                    continue
                cost = _fit_cost(mdict)
                prev = best_for.get(j)
                if prev is None or cost < prev[0]:
                    best_for[j] = (cost, i, mdict)

        if not best_for:
            break

        # Attach the piece with the globally lowest fit cost this round
        # (single-piece-at-a-time: no batch commits).
        ranked = sorted(best_for.items(), key=lambda kv: kv[1][0])
        picked = False
        for j, (cost, i, mdict) in ranked:
            conf = mdict["confidence"]
            if conf < cfg.PERIMETER_MIN_CONFIDENCE:
                continue
            ok = _attach(fragments, transforms, placed, locked, mdict, i, j)
            if not ok:
                continue
            added += 1
            picked = True
            changed = True
            merge_log.append({
                "phase":       "IV_perimeter",
                "anchor":      fragments[i]["id"],
                "attached":    fragments[j]["id"],
                "direction":   mdict.get("direction_hint"),
                "confidence":  round(conf, 3),
                "fit_cost":    round(cost, 2),
                "fit_coverage": round(mdict.get("fit_coverage", 0.0), 3),
                "fit_gap_px":  round(mdict.get("fit_gap_px", 0.0), 2),
                "angle_deg":   round(np.degrees(mdict["angle"]), 2),
                "rms":         round(mdict["rms"], 2),
            })
            break   # re-scan after every successful placement
        if not picked:
            break
    return added


# ══════════════════════════════════════════════════════════════════════════
#  Phase V  – Interior infilling via confidence-first priority queue
# ══════════════════════════════════════════════════════════════════════════

def _infill_interior(fragments: list[dict],
                      transforms: dict[int, np.ndarray],
                      placed: set[int], locked: dict[int, float],
                      cache: _MatchCache,
                      merge_log: list[dict]) -> int:
    """
    Sequential globally-best attachment.

    On every iteration we rebuild the list of (placed_i, unplaced_j) match
    candidates from each unplaced fragment's neighbor graph, evaluate them
    against the CURRENT placed set, and attach ONLY the single candidate
    with the lowest fit_cost (tightest overall touch: minimal overlap,
    minimal gap, maximal edge-length coverage).  Then we loop. This is
    what replaces the earlier "bunch of pieces committed at once" heapq
    behaviour — every attachment sees the up-to-date canvas.

    A blacklist of (i, j) pairs that failed the global overlap check
    prevents the loop from spinning forever on the same geometrically-
    impossible attach.
    """
    added = 0
    # Pairs that failed the dynamic-canvas overlap check — never retry.
    blacklist: set[tuple[int, int]] = set()

    while True:
        # Gather every (unplaced_j, best anchor for j) candidate, ranked
        # by physical fit_cost.  `cache.match` already applies the SDT
        # alignment gate and the MAX_ATTACH_COST upper bound, so every
        # candidate that reaches this point is physically plausible.
        best_for: dict[int, tuple[float, int, dict]] = {}
        for j in range(len(fragments)):
            if j in placed:
                continue
            # Expand the neighbor set to every placed fragment within
            # the proximity graph — this still prunes ~90% of pairs
            # versus full O(n^2).
            for nb in fragments[j]["neighbors"]:
                i = nb["j"]
                if i not in placed:
                    continue
                if (i, j) in blacklist or (j, i) in blacklist:
                    continue
                mdict = cache.match(i, j, direction=nb["direction"])
                if mdict is None:
                    continue
                conf = mdict["confidence"]
                if conf < cfg.INTERIOR_MIN_CONFIDENCE:
                    continue
                cost = _fit_cost(mdict)
                prev = best_for.get(j)
                if prev is None or cost < prev[0]:
                    best_for[j] = (cost, i, mdict)

        if not best_for:
            break

        # Pick the single globally-best candidate this round (lowest
        # fit_cost across ALL unplaced fragments).
        j, (cost, i, mdict) = min(best_for.items(), key=lambda kv: kv[1][0])
        conf = mdict["confidence"]

        ok = _attach(fragments, transforms, placed, locked, mdict, i, j)
        if not ok:
            # Geometry couldn't be reconciled with the current canvas —
            # blacklist and try the next globally-best candidate.
            blacklist.add((i, j))
            continue

        added += 1
        merge_log.append({
            "phase":       "V_interior",
            "anchor":      fragments[i]["id"],
            "attached":    fragments[j]["id"],
            "direction":   mdict.get("direction_hint"),
            "confidence":  round(conf, 3),
            "fit_cost":    round(cost, 2),
            "fit_coverage": round(mdict.get("fit_coverage", 0.0), 3),
            "fit_gap_px":  round(mdict.get("fit_gap_px", 0.0), 2),
            "fit_overlap_px": round(mdict.get("fit_overlap_px", 0.0), 2),
            "angle_deg":   round(np.degrees(mdict["angle"]), 2),
            "rms":         round(mdict["rms"], 2),
        })

    return added


def _force_fit_remaining(fragments: list[dict],
                          transforms: dict[int, np.ndarray],
                          placed: set[int], locked: dict[int, float],
                          cache: _MatchCache,
                          merge_log: list[dict]) -> int:
    """
    Low-confidence pieces (blank/smooth tears) get one final pass: for
    each un-placed piece we attach it at its BEST available pose
    regardless of confidence, subject to the global-overlap and global-
    boundary checks. This is the spec's "force fit remaining fragments
    into the only available slots" clause.
    """
    added = 0
    blacklist: set[tuple[int, int]] = set()
    while True:
        cand: list[tuple[float, int, int, dict]] = []
        for j in range(len(fragments)):
            if j in placed:
                continue
            best: tuple[float, int, dict] | None = None
            for nb in fragments[j]["neighbors"]:
                i = nb["j"]
                if i not in placed:
                    continue
                if (i, j) in blacklist or (j, i) in blacklist:
                    continue
                mdict = cache.match(i, j, direction=nb["direction"])
                if mdict is None:
                    continue
                cost = _fit_cost(mdict)
                if best is None or cost < best[0]:
                    best = (cost, i, mdict)
            if best is not None:
                cand.append((best[0], best[1], j, best[2]))

        if not cand:
            break
        # Lowest fit_cost first — closest attempt even when confidence is
        # weak, which is the whole point of this pass.
        cand.sort(key=lambda x: x[0])
        picked = False
        for cost, i, j, mdict in cand:
            conf = mdict["confidence"]
            ok = _attach(fragments, transforms, placed, locked, mdict, i, j)
            if not ok:
                blacklist.add((i, j))
                continue
            added += 1
            picked = True
            merge_log.append({
                "phase":       "V_force_fit",
                "anchor":      fragments[i]["id"],
                "attached":    fragments[j]["id"],
                "direction":   mdict.get("direction_hint"),
                "confidence":  round(conf, 3),
                "fit_cost":    round(cost, 2),
                "fit_coverage": round(mdict.get("fit_coverage", 0.0), 3),
                "fit_gap_px":  round(mdict.get("fit_gap_px", 0.0), 2),
                "angle_deg":   round(np.degrees(mdict["angle"]), 2),
                "rms":         round(mdict["rms"], 2),
            })
            break
        if not picked:
            break
    return added


# ══════════════════════════════════════════════════════════════════════════
#  Cluster jitter: post-placement snug-up against all placed neighbors
#
#  After Phase V every fragment sits at the pose that best suited its
#  single anchor at the moment it was placed. Small drift accumulates
#  along chains of attachments. This pass walks each non-corner placed
#  fragment and re-runs rigid ICP of its contour against the CONCATENATED
#  contour points of all currently-placed neighbors. Result: the whole
#  arrangement tightens, pulling untouched stretches of each tear
#  together without disturbing the corners that define the global frame.
# ══════════════════════════════════════════════════════════════════════════

def _gather_placed_neighbor_pts(j: int,
                                 fragments: list[dict],
                                 transforms: dict[int, np.ndarray],
                                 placed: set[int]) -> np.ndarray:
    """Concatenated world-frame contour points of every placed neighbor of j."""
    pts_all = []
    for nb in fragments[j]["neighbors"]:
        i = nb["j"]
        if i == j or i not in placed:
            continue
        pts_local = fragments[i]["contour"].astype(np.float64).reshape(-1, 2)
        pts_world = M.affine_apply(transforms[i], pts_local)
        pts_all.append(pts_world)
    if not pts_all:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(pts_all)


# ══════════════════════════════════════════════════════════════════════════
#  Global pose-graph bundle adjustment  (Phase 3H)
#
#  Every per-pair placement so far is LOCALLY optimal — it picked the
#  single-neighbor pose that minimised that neighbor's fit_cost. But
#  fragment j typically shares seams with TWO to FOUR placed neighbors,
#  and no single pose satisfies all of them at once. The result is
#  uniform ~1-3 px seam gaps that, across 10+ pieces, look like "tons
#  of free space" in the composed image.
#
#  Bundle adjustment re-poses every non-corner placed fragment to
#  simultaneously minimise the world-frame squared distance between
#  seam-point correspondences from ALL cached matches. Corners stay
#  pinned (they define the global gauge).  The optimisation is a
#  standard rigid-2D pose-graph with Levenberg-Marquardt.
# ══════════════════════════════════════════════════════════════════════════

def _bundle_adjust_poses(fragments: list[dict],
                          transforms: dict[int, np.ndarray],
                          placed: set[int], locked: dict[int, float],
                          corner_ids: set[int],
                          cache: "_MatchCache",
                          merge_log: list[dict]) -> int:
    """
    Joint LM refinement of every placed fragment's pose to minimise the
    sum of squared seam-coincidence residuals across all cached matches.
    Returns the number of fragments whose pose was actually updated.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return 0

    # ── Collect seam-coincidence constraints from the match cache.
    #
    # Cache key (lo, hi) with lo < hi was called as match_pair(frag_lo,
    # frag_hi), so match["matched_a"] is in fragments[lo]'s local frame
    # and match["matched_b"] is in fragments[hi]'s local frame. The
    # world-frame residual for each correspondence is therefore
    #   T_lo @ matched_a[k]  -  T_hi @ matched_b[k].
    constraints: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []
    for (lo, hi), m in cache.cache.items():
        if m is None:
            continue
        if lo not in placed or hi not in placed:
            continue
        ma = np.asarray(m.get("matched_a"), dtype=np.float64)
        mb = np.asarray(m.get("matched_b"), dtype=np.float64)
        if ma.size == 0 or mb.size == 0 or len(ma) != len(mb) or len(ma) < 3:
            continue
        # Per-constraint weight: trust tighter matches more. Fall back to
        # a modest weight if no fit_cost was recorded (old cache entries).
        fc = float(m.get("fit_cost", 10.0))
        w = 1.0 / max(fc, 1.0)
        constraints.append((lo, hi, ma, mb, float(np.sqrt(w))))

    if not constraints:
        return 0

    # Only non-corner placed fragments are free parameters.
    varying = sorted(i for i in placed if i not in corner_ids)
    if not varying:
        return 0
    idx_of = {i: k for k, i in enumerate(varying)}

    fixed_Rt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in placed:
        if i in corner_ids:
            T = transforms[i]
            fixed_Rt[i] = (T[:2, :2].copy(), T[:2, 2].copy())

    # Initial parameter vector from current transforms
    x0 = np.zeros(3 * len(varying), dtype=np.float64)
    for k, i in enumerate(varying):
        T = transforms[i]
        x0[3 * k]     = float(np.arctan2(T[1, 0], T[0, 0]))
        x0[3 * k + 1] = float(T[0, 2])
        x0[3 * k + 2] = float(T[1, 2])

    def _residuals(x: np.ndarray) -> np.ndarray:
        # Precompute each varying fragment's R, t once per LM call.
        Rs: dict[int, np.ndarray] = {}
        ts: dict[int, np.ndarray] = {}
        for k, i in enumerate(varying):
            theta = x[3 * k]
            tx    = x[3 * k + 1]
            ty    = x[3 * k + 2]
            ct = np.cos(theta); st = np.sin(theta)
            Rs[i] = np.array([[ct, -st], [st, ct]], dtype=np.float64)
            ts[i] = np.array([tx, ty], dtype=np.float64)

        parts: list[np.ndarray] = []
        for (lo, hi, ma, mb, sqw) in constraints:
            if lo in fixed_Rt:
                R_lo, t_lo = fixed_Rt[lo]
            else:
                R_lo, t_lo = Rs[lo], ts[lo]
            if hi in fixed_Rt:
                R_hi, t_hi = fixed_Rt[hi]
            else:
                R_hi, t_hi = Rs[hi], ts[hi]
            world_a = ma @ R_lo.T + t_lo
            world_b = mb @ R_hi.T + t_hi
            parts.append(sqw * (world_a - world_b).ravel())
        return np.concatenate(parts)

    initial_res = _residuals(x0)
    initial_cost = float(0.5 * np.sum(initial_res ** 2))

    t_start = time.time()
    result = least_squares(
        _residuals, x0,
        method="lm",
        max_nfev=cfg.BA_MAX_ITER,
        ftol=cfg.BA_FUNC_TOL,
        xtol=cfg.BA_FUNC_TOL,
    )
    x = result.x
    final_cost = float(0.5 * np.sum(result.fun ** 2))

    # ── Apply per-fragment with drift caps + overlap re-check.
    #
    # An LM step that wants to shift a piece by > BA_MAX_TRANSLATION_PX or
    # rotate by > BA_MAX_ROTATION_DEG is almost certainly chasing a bad
    # cached match; we reject those moves and keep the existing pose.
    old_T = {i: transforms[i].copy() for i in varying}
    proposals: list[tuple[int, float, float, np.ndarray]] = []
    for k, i in enumerate(varying):
        theta = float(x[3 * k])
        tx    = float(x[3 * k + 1])
        ty    = float(x[3 * k + 2])
        old   = old_T[i]
        old_theta = float(np.arctan2(old[1, 0], old[0, 0]))
        old_tx    = float(old[0, 2])
        old_ty    = float(old[1, 2])
        d_theta_deg = abs(np.degrees(theta - old_theta))
        d_shift_px  = float(np.hypot(tx - old_tx, ty - old_ty))
        if d_theta_deg > cfg.BA_MAX_ROTATION_DEG \
                or d_shift_px > cfg.BA_MAX_TRANSLATION_PX:
            continue
        ct = np.cos(theta); st = np.sin(theta)
        T_new = np.array([[ct, -st, tx],
                           [st,  ct, ty],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        proposals.append((i, d_theta_deg, d_shift_px, T_new))

    # Apply proposals, then individually revert any fragment whose new
    # pose violates the global overlap bound. (The others stay at their
    # BA-optimised pose.)
    for (i, _dt, _ds, T_new) in proposals:
        transforms[i] = T_new

    kept = 0
    for (i, d_theta_deg, d_shift_px, _T_new) in proposals:
        if not _global_overlap_ok(fragments, transforms, i, placed):
            transforms[i] = old_T[i]
        else:
            kept += 1

    merge_log.append({
        "phase":          "V_bundle_adjust",
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
#  Orphan rescue — unplaced fragments get one more shot against ANY placed
# ══════════════════════════════════════════════════════════════════════════

def _orphan_rescue(fragments: list[dict],
                    transforms: dict[int, np.ndarray],
                    placed: set[int], locked: dict[int, float],
                    cache: "_MatchCache",
                    merge_log: list[dict]) -> int:
    """
    Last-resort placement pass. Fragments that remained unplaced are,
    by definition, fragments whose NEIGHBOR graph didn't give them a
    workable match. Here we drop the neighbor-graph restriction and
    try each unplaced fragment against EVERY placed fragment.
    """
    if not cfg.ORPHAN_RESCUE_ENABLED:
        return 0
    added = 0
    while True:
        best: tuple[float, int, int, dict] | None = None
        for j in range(len(fragments)):
            if j in placed:
                continue
            for i in sorted(placed):
                mdict = cache.match(i, j, direction=None)
                if mdict is None:
                    continue
                cost = _fit_cost(mdict)
                if cost > cfg.ORPHAN_MAX_ATTACH_COST:
                    continue
                if best is None or cost < best[0]:
                    best = (cost, i, j, mdict)
        if best is None:
            break
        cost, i, j, mdict = best
        ok = _attach(fragments, transforms, placed, locked, mdict, i, j)
        if not ok:
            # Mark as tried and un-tryable this round by removing from cache
            # consideration — blacklist.
            key = (i, j) if i < j else (j, i)
            cache.cache[key] = None
            continue
        added += 1
        merge_log.append({
            "phase":       "V_orphan_rescue",
            "anchor":      fragments[i]["id"],
            "attached":    fragments[j]["id"],
            "direction":   mdict.get("direction_hint"),
            "confidence":  round(mdict["confidence"], 3),
            "fit_cost":    round(cost, 2),
            "fit_coverage": round(mdict.get("fit_coverage", 0.0), 3),
            "fit_gap_px":  round(mdict.get("fit_gap_px", 0.0), 2),
            "angle_deg":   round(np.degrees(mdict["angle"]), 2),
            "rms":         round(mdict["rms"], 2),
        })
    return added


# ══════════════════════════════════════════════════════════════════════════
#  Dense-boundary final jitter — uses every placed fragment within radius
# ══════════════════════════════════════════════════════════════════════════

def _wide_cluster_jitter(fragments: list[dict],
                          transforms: dict[int, np.ndarray],
                          placed: set[int], locked: dict[int, float],
                          corner_ids: set[int],
                          merge_log: list[dict],
                          max_corr_dist: float,
                          tol_shift_px: float) -> int:
    """
    Like _cluster_jitter but pulls each fragment's contour against the
    concatenated contours of EVERY placed fragment (not just neighbors),
    clipped to correspondences closer than `max_corr_dist` px. The wider
    association radius lets this pass close the sub-pixel-to-few-pixel
    residual gaps left over after bundle adjustment.
    """
    updated = 0
    for j in sorted(placed):
        if j in corner_ids:
            continue
        # Concatenate all OTHER placed fragments' world-frame contour pts
        pts_all: list[np.ndarray] = []
        for i in placed:
            if i == j:
                continue
            pts_local = fragments[i]["contour"].astype(np.float64).reshape(-1, 2)
            pts_all.append(M.affine_apply(transforms[i], pts_local))
        if not pts_all:
            continue
        neighbor_pts = np.vstack(pts_all)
        if len(neighbor_pts) < 10:
            continue

        local_pts = fragments[j]["contour"].astype(np.float64).reshape(-1, 2)
        world_pts = M.affine_apply(transforms[j], local_pts)

        R_delta, t_delta, _rms = M.icp_refine(
            src_pts=world_pts,
            dst_pts=neighbor_pts,
            R0=np.eye(2, dtype=np.float64),
            t0=np.zeros(2, dtype=np.float64),
            max_iter=cfg.ICP_MAX_ITER,
            max_correspondence_dist=max_corr_dist,
        )

        drift_deg = abs(np.degrees(
            float(np.arctan2(R_delta[1, 0], R_delta[0, 0]))))
        shift_px  = float(np.linalg.norm(t_delta))
        if drift_deg > cfg.ICP_MAX_DRIFT_DEG or shift_px > tol_shift_px:
            continue

        old = transforms[j].copy()
        M_delta = M.affine_from_Rt(R_delta, t_delta)
        transforms[j] = M_delta @ transforms[j]
        if not _global_overlap_ok(fragments, transforms, j, placed):
            transforms[j] = old
            continue
        updated += 1
        merge_log.append({
            "phase":       "V_wide_jitter",
            "fragment":    fragments[j]["id"],
            "drift_deg":   round(drift_deg, 3),
            "shift_px":    round(shift_px, 2),
        })
    return updated


def _cluster_jitter(fragments: list[dict],
                     transforms: dict[int, np.ndarray],
                     placed: set[int], locked: dict[int, float],
                     corner_ids: set[int],
                     merge_log: list[dict]) -> int:
    """
    One sweep of neighborhood-aware pose refinement.

    For each placed, non-corner fragment:
      1. Gather world-frame contour points of all placed neighbors.
      2. Run ICP of this fragment's world-frame contour against that
         neighbor point cloud (wide correspondence tolerance — we are
         looking for small corrective pulls, not large relocations).
      3. Reject the correction if it would shift the centroid by more
         than CLUSTER_JITTER_TOL_PX (safety cap against pathological pulls)
         or if it rotates the fragment by more than ICP_MAX_DRIFT_DEG.
      4. Reject if the new pose breaks the global overlap check.

    Returns the number of fragments whose pose was updated.
    """
    updated = 0
    for j in sorted(placed):
        if j in corner_ids:
            continue   # corners anchor the global frame

        neighbor_pts = _gather_placed_neighbor_pts(
            j, fragments, transforms, placed)
        if len(neighbor_pts) < 10:
            continue

        local_pts = fragments[j]["contour"].astype(np.float64).reshape(-1, 2)
        world_pts = M.affine_apply(transforms[j], local_pts)

        R0 = np.eye(2, dtype=np.float64)
        t0 = np.zeros(2, dtype=np.float64)

        # World-frame ICP: treat current placement as the initial estimate
        # and let ICP produce a delta (R, t) to apply on top.
        R_delta, t_delta, _rms = M.icp_refine(
            src_pts=world_pts,
            dst_pts=neighbor_pts,
            R0=R0, t0=t0,
            max_iter=cfg.ICP_MAX_ITER,
            max_correspondence_dist=cfg.ICP_COARSE_DIST_PX,
        )

        # Drift gates
        drift_deg = abs(np.degrees(float(np.arctan2(R_delta[1, 0], R_delta[0, 0]))))
        if drift_deg > cfg.ICP_MAX_DRIFT_DEG:
            continue
        if float(np.linalg.norm(t_delta)) > cfg.CLUSTER_JITTER_TOL_PX:
            continue

        M_delta = M.affine_from_Rt(R_delta, t_delta)
        old = transforms[j].copy()
        transforms[j] = M_delta @ transforms[j]

        # Preserve global overlap invariant
        if not _global_overlap_ok(fragments, transforms, j, placed - {j}):
            transforms[j] = old
            continue

        updated += 1
        merge_log.append({
            "phase":       "V_cluster_jitter",
            "fragment":    fragments[j]["id"],
            "drift_deg":   round(drift_deg, 3),
            "shift_px":    round(float(np.linalg.norm(t_delta)), 2),
        })
    return updated


# ══════════════════════════════════════════════════════════════════════════
#  Conflict resolution: max-weight matching on competing anchors
# ══════════════════════════════════════════════════════════════════════════

def _resolve_conflicts(fragments: list[dict],
                        cache: _MatchCache,
                        candidate_anchors: dict[int, list[tuple[int, dict]]]
                        ) -> dict[int, tuple[int, dict]]:
    """
    Given {unplaced_j: [(anchor_i, match), ...]}, return a single
    anchor -> (j, match) mapping that maximizes total confidence and
    ensures no anchor is assigned twice.

    Uses networkx max_weight_matching on a bipartite-like graph.
    """
    G = nx.Graph()
    for j, opts in candidate_anchors.items():
        for i, mdict in opts:
            u = ("j", j)
            v = ("i", i)
            G.add_edge(u, v, weight=float(mdict["confidence"]))
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
        # Recover the specific match from the candidate list
        for ii, mdict in candidate_anchors[j]:
            if ii == i:
                out[i] = (j, mdict)
                break
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Global boundary sanity check (A4/Letter ratio)
# ══════════════════════════════════════════════════════════════════════════

def _global_boundary_ok(fragments: list[dict],
                         transforms: dict[int, np.ndarray],
                         placed: set[int]) -> bool:
    """
    Roughly check that the aggregate placed fragments fit a landscape/
    portrait document aspect ratio. Pure sanity gate for pathological
    runs — we accept wide bands around A4 (1:1.414) and Letter (1:1.294).
    """
    if len(placed) < 3:
        return True
    mn_x, mn_y, mx_x, mx_y = _global_frame_bounds(fragments, transforms, placed)
    width = max(1.0, mx_x - mn_x)
    height = max(1.0, mx_y - mn_y)
    ratio = max(width, height) / min(width, height)
    return cfg.DOC_MIN_ASPECT <= ratio <= cfg.DOC_MAX_ASPECT


# ══════════════════════════════════════════════════════════════════════════
#  Debug: save intermediate reconstruction snapshots
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
    ox = -mn_x + pad
    oy = -mn_y + pad

    canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 200
    for i in sorted(placed):
        frag = fragments[i]
        Mg = transforms[i].copy()
        Mg[0, 2] += ox
        Mg[1, 2] += oy
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
#  Entry point — called from pipeline.py
# ══════════════════════════════════════════════════════════════════════════

def reconstruct(fragments: list[dict],
                image_rgb: np.ndarray,
                debug_dir: Path) -> dict[int, np.ndarray]:
    """
    Full hierarchical reconstruction. Returns a transforms dict mapping
    each fragment_index -> 3x3 homogeneous affine in the image frame.
    Fragments not placed keep the identity transform (= scan position).
    """
    recon_debug = debug_dir / "reconstruction"
    recon_debug.mkdir(parents=True, exist_ok=True)

    n = len(fragments)
    transforms: dict[int, np.ndarray] = {
        i: np.eye(3, dtype=np.float64) for i in range(n)}
    placed: set[int] = set()
    locked: dict[int, float] = {}
    merge_log: list[dict] = []

    # ── Phase 3A – Neighbor discovery ─────────────────────────────────
    print("\n  -- Phase 3A: neighbor discovery --")
    t0 = time.time()
    discover_neighbors(fragments, debug_dir=debug_dir)
    print(f"     {n} fragments, "
          f"{sum(len(f['neighbors']) for f in fragments)//2} undirected "
          f"candidate pairs "
          f"({time.time()-t0:.2f}s)")

    # ── Edge extraction + SDT pre-compute ─────────────────────────────
    print("  -- Preparing edges + interior SDTs --")
    t0 = time.time()
    M.prepare_edges_and_sdt(fragments)
    print(f"     done ({time.time()-t0:.2f}s)")

    # ── Identify corners ──────────────────────────────────────────────
    corners = identify_corners(fragments)
    print("  -- Corners (extreme-centroid heuristic):",
          {k: (fragments[v]["id"] if v is not None else None)
           for k, v in corners.items()})

    # Match cache shared across all phases
    cache = _MatchCache(fragments, image_rgb)

    step = 0

    # ── Phase 3B (II) – Corner seed batching ─────────────────────────
    print("\n  -- Phase 3B: corner seed batching (horizontal pivots) --")
    seed_partners: dict[str, int | None] = {}
    for cname in ("TL", "TR", "BL", "BR"):
        cidx = corners[cname]
        if cidx is None:
            print(f"     {cname}: no corner candidate (layout too sparse)")
            seed_partners[cname] = None
            continue
        if cidx in placed:
            seed_partners[cname] = None
            continue
        partner = _seed_corner(cname, cidx, fragments, transforms,
                                placed, locked, cache, merge_log)
        seed_partners[cname] = partner
        if partner is not None:
            last = merge_log[-1]
            print(f"     {cname}: frag {last['anchor']} <- frag {last['attached']}  "
                  f"conf={last['confidence']:.3f}  "
                  f"angle={last['angle_deg']:+.1f} deg  rms={last['rms']:.2f}")
            step += 1
            _save_step(image_rgb, fragments, transforms, placed,
                        step, f"seed_{cname}", recon_debug)
        else:
            print(f"     {cname}: anchored frag {fragments[cidx]['id']}, "
                  f"no horizontal partner match")

    # ── Phase 3C (III) – Vertical expansion to L-clusters ────────────
    print("\n  -- Phase 3C: vertical expansion (L-shaped clusters) --")
    for cname in ("TL", "TR", "BL", "BR"):
        cidx = corners[cname]
        if cidx is None:
            continue
        added = _expand_vertical(cname, cidx, fragments, transforms,
                                  placed, locked, cache, merge_log)
        if added is not None:
            last = merge_log[-1]
            print(f"     {cname}: frag {last['anchor']} <- frag {last['attached']}  "
                  f"conf={last['confidence']:.3f}  "
                  f"angle={last['angle_deg']:+.1f} deg  rms={last['rms']:.2f}")
            step += 1
            _save_step(image_rgb, fragments, transforms, placed,
                        step, f"L_{cname}", recon_debug)
        else:
            print(f"     {cname}: no vertical partner "
                  f"(cluster stays 2 pieces)")

    # ── Phase 3D (IV) – Perimeter frame infilling ────────────────────
    print("\n  -- Phase 3D: perimeter frame infilling --")
    n_perim = _infill_perimeter(fragments, transforms, placed, locked,
                                 cache, merge_log)
    print(f"     perimeter placements: {n_perim}")
    step += 1
    _save_step(image_rgb, fragments, transforms, placed,
                step, "perimeter_done", recon_debug)

    # ── Phase 3E (V) – Interior infilling ────────────────────────────
    print("\n  -- Phase 3E: interior infilling (sequential globally-best) --")
    n_interior = _infill_interior(fragments, transforms, placed, locked,
                                   cache, merge_log)
    print(f"     interior placements: {n_interior}")
    step += 1
    _save_step(image_rgb, fragments, transforms, placed,
                step, "interior_done", recon_debug)

    # ── Force-fit remaining (blank/low-confidence pieces) ───────────
    print("\n  -- Phase 3F: force-fit remaining --")
    n_forced = _force_fit_remaining(fragments, transforms, placed, locked,
                                     cache, merge_log)
    print(f"     forced placements: {n_forced}")
    if n_forced:
        step += 1
        _save_step(image_rgb, fragments, transforms, placed,
                    step, "forced_done", recon_debug)

    # ── Phase 3G – Cluster jitter (neighborhood-aware snug-up) ───────
    # The placements so far each optimised one pair at a time; small
    # residual drift accumulates across the cluster. This pass walks
    # each non-corner placed fragment and pulls it toward the combined
    # boundary of its already-placed neighbors. Corners are kept fixed
    # because they anchor the global frame.
    corner_ids = {cidx for cidx in corners.values() if cidx is not None}
    print("\n  -- Phase 3G: cluster jitter (post-placement snug-up) --")
    total_jitter = 0
    for it in range(cfg.CLUSTER_JITTER_ITERS):
        n_jitter = _cluster_jitter(fragments, transforms, placed, locked,
                                    corner_ids, merge_log)
        print(f"     iter {it+1}: {n_jitter} fragment(s) refined")
        total_jitter += n_jitter
        if n_jitter == 0:
            break
    if total_jitter > 0:
        step += 1
        _save_step(image_rgb, fragments, transforms, placed,
                    step, "jitter_done", recon_debug)

    # ── Phase 3H – Orphan rescue ─────────────────────────────────────
    # Fragments still at identity after force-fit never found a path
    # through the neighbor graph. Give them one more chance to dock
    # onto ANY placed fragment, lowest-fit-cost first.
    unplaced_before = [i for i in range(n) if i not in placed]
    if unplaced_before:
        print("\n  -- Phase 3H: orphan rescue (ignore neighbor graph) --")
        n_orphan = _orphan_rescue(fragments, transforms, placed, locked,
                                    cache, merge_log)
        print(f"     orphan placements: {n_orphan}  "
              f"(of {len(unplaced_before)} unplaced)")
        if n_orphan:
            step += 1
            _save_step(image_rgb, fragments, transforms, placed,
                        step, "orphan_done", recon_debug)

    # ── Phase 3I – Global pose-graph bundle adjustment ───────────────
    # Every piece's pose so far satisfies only its single anchoring
    # neighbor. This joint LM re-solve pulls every non-corner pose to
    # the best simultaneous fit against ALL its cached seam partners,
    # which is what actually closes the residual gaps you see in the
    # composed image. Corners stay pinned as the global gauge frame.
    if cfg.BA_ENABLED:
        print("\n  -- Phase 3I: global pose-graph bundle adjustment --")
        t_ba = time.time()
        n_ba = _bundle_adjust_poses(
            fragments, transforms, placed, locked,
            corner_ids, cache, merge_log,
        )
        if merge_log and merge_log[-1].get("phase") == "V_bundle_adjust":
            rec = merge_log[-1]
            print(f"     {rec['n_constraints']} seam constraints, "
                  f"{rec['n_varying']} varying frags, "
                  f"cost {rec['cost_initial']:.1f} -> "
                  f"{rec['cost_final']:.1f}, "
                  f"{n_ba}/{rec['n_proposed']} pose updates kept "
                  f"({time.time()-t_ba:.2f}s)")
        else:
            print(f"     skipped (no usable constraints)")
        if n_ba:
            step += 1
            _save_step(image_rgb, fragments, transforms, placed,
                        step, "bundle_adjust_done", recon_debug)

    # ── Phase 3J – Final wide jitter (post-BA snug-up) ──────────────
    # BA re-poses fragments jointly; a final ICP sweep against the
    # combined boundary of EVERY placed fragment (not just neighbors)
    # closes the residual sub-pixel / few-pixel gaps into genuine
    # edge-to-edge contact.
    print("\n  -- Phase 3J: final wide-radius cluster jitter --")
    total_final = 0
    for it in range(cfg.CLUSTER_JITTER_ITERS_FINAL):
        n_final = _wide_cluster_jitter(
            fragments, transforms, placed, locked,
            corner_ids, merge_log,
            max_corr_dist=cfg.CLUSTER_JITTER_WIDE_DIST_PX,
            tol_shift_px=cfg.CLUSTER_JITTER_TOL_PX_FINAL,
        )
        print(f"     iter {it+1}: {n_final} fragment(s) refined")
        total_final += n_final
        if n_final == 0:
            break
    if total_final > 0:
        step += 1
        _save_step(image_rgb, fragments, transforms, placed,
                    step, "final_jitter_done", recon_debug)

    # ── Global boundary sanity ───────────────────────────────────────
    if not _global_boundary_ok(fragments, transforms, placed):
        print("  ! WARNING: placed fragments exceed plausible "
              "document aspect ratio — reconstruction may be miss-aligned.")

    unplaced = [i for i in range(n) if i not in placed]
    if unplaced:
        print(f"  {len(unplaced)} fragment(s) remain unplaced "
              f"(kept at identity)")

    # ── Persist metadata ─────────────────────────────────────────────
    final = {}
    for i in range(n):
        ang, tx, ty = M.affine_angle_translation(transforms[i])
        final[fragments[i]["id"]] = {
            "angle_deg": round(np.degrees(ang), 2),
            "dx": round(tx, 1),
            "dy": round(ty, 1),
            "placed":    i in placed,
            "locked_conf": round(locked.get(i, 0.0), 3),
        }
    with open(recon_debug / "final_translations.json", "w",
              encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(recon_debug / "merge_log.json", "w", encoding="utf-8") as f:
        json.dump(merge_log, f, indent=2)

    # Match-cache stats
    with open(recon_debug / "cache_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "calls": cache.n_calls,
            "hits":  cache.n_hits,
            "total_entries": len(cache.cache),
            "successful_matches": sum(1 for v in cache.cache.values()
                                      if v is not None),
        }, f, indent=2)

    print(f"\n  Reconstruction complete: {len(placed)}/{n} placed, "
          f"{len(merge_log)} merges, "
          f"match cache {cache.n_calls} calls / {cache.n_hits} hits")
    return transforms


__all__ = ["reconstruct"]
