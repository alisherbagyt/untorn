"""
untorn.assembly
===============
Layout-agnostic global assembly by MST-growth.

The old `untorn.reconstruction` module assumed fragments arrive in a
roughly correct scan layout and used that to anchor four document
corners, then grow each corner cluster inward. That assumption breaks
outright when fragments are randomly rotated or laid out (case 2 in the
test bank), and silently mis-seeds on destroyed / partial inputs (case
3). This module replaces it.

Strategy
--------
1. Match every torn-edge vs torn-edge pair across distinct fragments
   through the four-gate matcher. Prefilter by edge-length ratio and
   paper-color compatibility to keep the set tractable.
2. Seed the MST with the single highest-confidence match — no corner
   anchor, no layout prior. The seed fragment is pinned at identity and
   acts as the global gauge.
3. Grow: repeatedly attach the highest-confidence match whose one side
   is already placed and the other is free. Conflicts (multiple free
   fragments wanting the same anchor) are resolved by networkx's
   max-weight matching on a bipartite graph.
4. Every `ASSEMBLY_GLOBAL_ROT_FIX_EVERY` placements, fit a global
   rotation that horizontalises the text baselines of every placed
   fragment. Rotate the whole cluster around its centroid. This
   cancels accumulated drift before it forks a wrong branch.
5. Bundle adjustment at the end pins only the seed; every other placed
   fragment is a free 3-DoF variable in a classic 2-D pose-graph.
6. Orphan rescue at relaxed confidence picks up fragments that never
   attracted a passing match — they get one more chance against every
   placed fragment.

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
from .io_utils import save_image
from .text_lines import _apply_affine as _apply_affine_2d


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

def _attach(fragments: list[dict],
            transforms: dict[int, np.ndarray],
            placed: set[int], locked: dict[int, float],
            match: dict, i: int, j: int) -> bool:
    """Compose transforms so fragment j sits where `match` says it should."""
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


def _enumerate_pair_candidates(fragments: list[dict]) -> list[tuple[int, int]]:
    """All i<j pairs with at least one torn edge each, paper-color compatible."""
    from .matching import _paper_lab_delta
    n = len(fragments)
    max_edges = [_max_torn_edge_length(f) for f in fragments]
    lab_deltas: list[tuple[float, int, int]] = []
    for i in range(n):
        if max_edges[i] <= 0.0:
            continue
        for j in range(i + 1, n):
            if max_edges[j] <= 0.0:
                continue
            # Edge-length ratio prefilter.
            a, b = max_edges[i], max_edges[j]
            if max(a, b) / max(min(a, b), 1e-6) > cfg.ASSEMBLY_EDGE_LENGTH_RATIO_MAX:
                continue
            d = _paper_lab_delta(fragments[i], fragments[j])
            # Keep when ΔE is finite (prefers same-paper) or unknown
            # (unknown means we can't prefilter so let the matcher decide).
            if np.isfinite(d) and d > cfg.MATCH_PAPER_COLOR_DELTA_MAX * 2.0:
                continue
            score = d if np.isfinite(d) else cfg.MATCH_PAPER_COLOR_DELTA_MAX
            lab_deltas.append((score, i, j))
    # Score-ascending so the best paper-compat pairs get their matcher call
    # before we hit ASSEMBLY_MAX_CANDIDATE_PAIRS.
    lab_deltas.sort()
    return [(i, j) for (_d, i, j) in
            lab_deltas[: cfg.ASSEMBLY_MAX_CANDIDATE_PAIRS]]


# ══════════════════════════════════════════════════════════════════════════
#  Text-line global rotation fix-up
# ══════════════════════════════════════════════════════════════════════════

def _global_text_rotation_angle(fragments: list[dict],
                                transforms: dict[int, np.ndarray],
                                placed: set[int]) -> float | None:
    """
    Median canvas-space baseline angle across every placed fragment's
    detected text lines. `None` if we don't have enough signal.
    """
    angles: list[float] = []
    for i in placed:
        lines = fragments[i].get("text_lines") or []
        if not lines:
            continue
        T = transforms[i]
        for ln in lines:
            p0 = _apply_affine_2d(T, np.asarray(ln["p0"], dtype=np.float64))
            p1 = _apply_affine_2d(T, np.asarray(ln["p1"], dtype=np.float64))
            d = p1 - p0
            if np.hypot(d[0], d[1]) < 8.0:
                continue
            # Map to [-pi/2, pi/2) so 180-flipped baselines agree.
            ang = float(np.arctan2(d[1], d[0]))
            if ang >= np.pi / 2:
                ang -= np.pi
            elif ang < -np.pi / 2:
                ang += np.pi
            angles.append(ang)
    if len(angles) < cfg.ASSEMBLY_GLOBAL_ROT_MIN_LINES:
        return None
    return float(np.median(angles))


def _apply_global_rotation(transforms: dict[int, np.ndarray],
                           placed: set[int],
                           centroid: np.ndarray,
                           delta_angle: float) -> None:
    """Rotate every placed fragment's transform by -delta around centroid."""
    ct = float(np.cos(-delta_angle))
    st = float(np.sin(-delta_angle))
    R = np.array([[ct, -st], [st, ct]], dtype=np.float64)
    cx, cy = float(centroid[0]), float(centroid[1])
    # Canvas-frame post-rotation: T' = Rot_about_centroid @ T
    for i in placed:
        T = transforms[i]
        # Move centroid to origin, rotate, move back.
        M_rot = np.eye(3, dtype=np.float64)
        M_rot[:2, :2] = R
        M_rot[0, 2] = cx - R[0, 0] * cx - R[0, 1] * cy
        M_rot[1, 2] = cy - R[1, 0] * cx - R[1, 1] * cy
        transforms[i] = M_rot @ T


def _placed_centroid(fragments: list[dict],
                     transforms: dict[int, np.ndarray],
                     placed: set[int]) -> np.ndarray:
    if not placed:
        return np.zeros(2, dtype=np.float64)
    pts = []
    for i in placed:
        c = np.asarray(fragments[i].get("centroid", (0.0, 0.0)),
                       dtype=np.float64)
        pts.append(_apply_affine_2d(transforms[i], c))
    return np.mean(np.asarray(pts), axis=0)


def _maybe_global_text_rotation(fragments: list[dict],
                                transforms: dict[int, np.ndarray],
                                placed: set[int],
                                merge_log: list[dict]) -> None:
    angle = _global_text_rotation_angle(fragments, transforms, placed)
    if angle is None:
        return
    if abs(np.degrees(angle)) < cfg.ASSEMBLY_GLOBAL_ROT_MIN_DEG:
        return
    centroid = _placed_centroid(fragments, transforms, placed)
    _apply_global_rotation(transforms, placed, centroid, angle)
    merge_log.append({
        "phase":       "mst_global_text_rotation",
        "delta_deg":   round(float(np.degrees(angle)), 3),
        "n_placed":    len(placed),
    })


# ══════════════════════════════════════════════════════════════════════════
#  Bundle adjustment — pin seed, vary every other placed fragment
# ══════════════════════════════════════════════════════════════════════════

def _bundle_adjust_poses(fragments: list[dict],
                         transforms: dict[int, np.ndarray],
                         placed: set[int], locked: dict[int, float],
                         pinned: set[int],
                         cache: _MatchCache,
                         merge_log: list[dict]) -> int:
    """
    LM refinement of every placed fragment's pose except `pinned`. The
    residual is world-frame seam-coincidence across every cached match.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return 0

    constraints: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []
    for (lo, hi), m in cache.cache.items():
        if m is None or lo not in placed or hi not in placed:
            continue
        ma = np.asarray(m.get("matched_a"), dtype=np.float64)
        mb = np.asarray(m.get("matched_b"), dtype=np.float64)
        if ma.size == 0 or mb.size == 0 or len(ma) != len(mb) or len(ma) < 3:
            continue
        fc = float(m.get("fit_cost", 10.0))
        w = 1.0 / max(fc, 1.0)
        constraints.append((lo, hi, ma, mb, float(np.sqrt(w))))

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
        for (lo, hi, ma, mb, sqw) in constraints:
            R_lo, t_lo = (fixed_Rt[lo] if lo in fixed_Rt else (Rs[lo], ts[lo]))
            R_hi, t_hi = (fixed_Rt[hi] if hi in fixed_Rt else (Rs[hi], ts[hi]))
            world_a = ma @ R_lo.T + t_lo
            world_b = mb @ R_hi.T + t_hi
            parts.append(sqw * (world_a - world_b).ravel())
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
        ok = _attach(fragments, transforms, placed, locked, mdict, i, j)
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

    # ── Phase A: enumerate candidate pairs and score each ───────────────
    print("  -- Enumerating candidate edge pairs --")
    t0 = time.time()
    candidates = _enumerate_pair_candidates(fragments)
    print(f"     {len(candidates)} pairs survived prefilter "
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

    if not ranked:
        print("  -- No candidate pair above the confidence floor. "
              "Returning identity placement.")
        _finalize_release_dinov2(fragments)
        return transforms

    # ── Phase B: seed from the single highest-confidence pair ───────────
    conf0, i0, j0 = ranked[0]
    m0 = cache.match(i0, j0)
    transforms[i0] = np.eye(3, dtype=np.float64)
    placed.add(i0)
    locked[i0] = 1.0
    seed_id = i0
    seeded = _attach(fragments, transforms, placed, locked, m0, i0, j0)
    if not seeded:
        print("  -- Seed attach failed the overlap gate. Retrying next pair.")
        for (conf_alt, i_alt, j_alt) in ranked[1:]:
            placed.clear(); transforms[i_alt] = np.eye(3)
            placed.add(i_alt); locked[i_alt] = 1.0
            m_alt = cache.match(i_alt, j_alt)
            if m_alt is None:
                continue
            if _attach(fragments, transforms, placed, locked, m_alt,
                       i_alt, j_alt):
                seed_id = i_alt
                conf0, i0, j0, m0 = conf_alt, i_alt, j_alt, m_alt
                seeded = True
                break
        if not seeded:
            print("  -- Could not seed any pair without overlap. "
                  "Falling back to identity.")
            _finalize_release_dinov2(fragments)
            return transforms

    print(f"  -- Seed: frag {fragments[i0]['id']} + "
          f"{fragments[j0]['id']} @ confidence {conf0:.2f}")
    merge_log.append({"phase": "seed",
                      "anchor": fragments[i0]["id"],
                      "attached": fragments[j0]["id"],
                      "confidence": round(conf0, 3)})
    _save_step(image_rgb, fragments, transforms, placed,
               0, "seed", recon_debug)

    # ── Phase C: MST growth ─────────────────────────────────────────────
    print("  -- Growing MST by confidence --")
    step = 0
    while step < cfg.ASSEMBLY_MAX_STEPS:
        # Collect attachments touching the placed set
        per_free: dict[int, list[tuple[int, dict]]] = {}
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
            per_free.setdefault(j_f, []).append((i_p, m))

        if not per_free:
            break

        # Resolve conflicts — a single anchor can attract multiple free
        # fragments; max-weight matching picks the globally best assignment.
        resolved = _resolve_conflicts(per_free)

        if not resolved:
            # No matching fired (e.g., single edge) — greedy pick the best.
            best: tuple[float, int, int, dict] | None = None
            for j_f, opts in per_free.items():
                for (i_p, m) in opts:
                    c = float(m["confidence"])
                    if best is None or c > best[0]:
                        best = (c, i_p, j_f, m)
            if best is None:
                break
            _c, i_p, j_f, m = best
            resolved = {i_p: (j_f, m)}

        attached_this_round = 0
        for (i_p, (j_f, m)) in resolved.items():
            if j_f in placed:
                continue
            if not _attach(fragments, transforms, placed, locked, m, i_p, j_f):
                continue
            attached_this_round += 1
            step += 1
            merge_log.append({
                "phase":       "mst_attach",
                "step":        step,
                "anchor":      fragments[i_p]["id"],
                "attached":    fragments[j_f]["id"],
                "confidence":  round(float(m["confidence"]), 3),
                "angle_deg":   round(float(np.degrees(m["angle"])), 2),
                "rms":         round(float(m["rms"]), 2),
            })
            # Periodic text-line global rotation fix-up.
            if (cfg.ASSEMBLY_GLOBAL_ROT_FIX_EVERY > 0 and
                    step % cfg.ASSEMBLY_GLOBAL_ROT_FIX_EVERY == 0):
                _maybe_global_text_rotation(
                    fragments, transforms, placed, merge_log)

        if attached_this_round == 0:
            break

    _save_step(image_rgb, fragments, transforms, placed,
               max(1, step), "mst_done", recon_debug)

    print(f"     MST grew to {len(placed)}/{n} placed in {step} steps")

    # ── Phase D: bundle adjustment (pin only the seed) ──────────────────
    print("  -- Bundle adjustment (seed-pinned) --")
    _bundle_adjust_poses(fragments, transforms, placed, locked,
                         pinned={seed_id},
                         cache=cache, merge_log=merge_log)
    _save_step(image_rgb, fragments, transforms, placed,
               max(1, step) + 1, "bundle_adjust", recon_debug)

    # ── Phase E: orphan rescue at relaxed confidence ────────────────────
    if len(placed) < n:
        print("  -- Orphan rescue at relaxed confidence --")
        n_rescued = _orphan_rescue(
            fragments, transforms, placed, locked, cache, merge_log,
            min_confidence=cfg.ASSEMBLY_ORPHAN_MIN_CONFIDENCE)
        print(f"     rescued {n_rescued}")
        _save_step(image_rgb, fragments, transforms, placed,
                   max(1, step) + 2, "orphan_rescue", recon_debug)

    # Final text-rotation pass for a cosmetic straighten.
    _maybe_global_text_rotation(fragments, transforms, placed, merge_log)
    _save_step(image_rgb, fragments, transforms, placed,
               max(1, step) + 3, "final", recon_debug)

    # ── Logs and VRAM cleanup ───────────────────────────────────────────
    with open(recon_debug / "merge_log.json", "w", encoding="utf-8") as fh:
        json.dump(merge_log, fh, indent=2)
    with open(recon_debug / "assembly_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "n_fragments":          n,
            "n_placed":             len(placed),
            "n_candidate_pairs":    len(candidates),
            "n_ranked_pairs":       len(ranked),
            "seed_fragment":        int(fragments[seed_id]["id"]),
            "match_cache_calls":    cache.n_calls,
            "match_cache_hits":     cache.n_hits,
        }, fh, indent=2)

    _finalize_release_dinov2(fragments)
    print(f"\n  Assembly complete: {len(placed)}/{n} placed, "
          f"{cache.n_calls} matcher calls / {cache.n_hits} cache hits")
    return transforms


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
