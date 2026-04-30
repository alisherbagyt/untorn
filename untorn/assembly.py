"""untorn.assembly — Phase 3 global solver.

Inputs every fragment dict that ``fragment_io.build_all`` ingested, asks
``matching.match_pair`` for the relative SE(2) pose of every plausible
torn-edge pair, then assembles the global pose graph.

Algorithm (engine-rebuild April 2026)
-------------------------------------
1. **Candidate enumeration** — paper-LAB ΔE + boundary proximity. No LBP
   fast-filter, no per-edge-rank module. With ≤40 fragments at the
   working resolution this is fast enough and never drops a real match.

2. **Pair scoring** — call ``match_pair`` once per candidate; cache the
   result keyed ``(min(i,j), max(i,j))``.

3. **Union-Find greedy merge** in ASCENDING fit_cost order, with two
   acceptance gates beyond a simple confidence sort:

   a) **Cluster-wide consistency.** When merging clusters A and B via a
      candidate (i, j), every previously cached match between cluster-A
      and cluster-B members must remain ``fit_cost`` acceptable under
      the proposed merged frame. If any cached pair degrades by more
      than a slack factor, the merge is rejected with reason
      ``cluster_consistency_violated``.

   b) **Cycle re-validation.** When the candidate would close a cycle
      (i and j already share a cluster), compare the candidate's
      relative pose to the chain pose computed from the existing
      transforms. If they disagree by more than a tight tolerance, run
      a cluster-scoped bundle adjustment to redistribute the conflict.
      Cost-monotone: revert if BA didn't actually reduce total cost.

4. **Per-attach seam refinement** — ``seam_solver.refine_pair`` polishes
   each accepted merge.

5. **Final cluster-scoped BA** — sparse LM over each cluster with dense
   edge-to-edge correspondences. Cost-monotone acceptance.

6. **Single orphan retry** with relaxed thresholds for any fragment that
   never got merged.

Output contract (unchanged for backwards compat):
    reconstruct(fragments, image_rgb, debug_dir) -> dict[int, 3x3 affine]

Unplaced fragments retain identity. ``placed`` is reported via
``assembly_summary.json``; ``merge_log.json`` records every score and
every accept/reject decision with reasons.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from . import config as cfg
from . import matching as M
from . import seam_solver
from .io_utils import save_image

logger = logging.getLogger("untorn.engine.assembly")


_CANVAS_PAD_PX = 30                  # single global padding constant
_CLUSTER_CONSISTENCY_SLACK = 1.2     # cached-pair fit_cost can grow by 20% post-merge
# Cycle handling tolerances. A cycle is "consistent" if the chain pose
# matches the cycle-edge pose within these tight tolerances. A cycle is
# "borderline" — drift worth correcting via BA — within the wider band.
# Beyond the wider band, the cycle edge is treated as a false-positive
# match and IGNORED (we trust the chain).
_CYCLE_R_TOL_TIGHT      = 0.05
_CYCLE_T_TOL_TIGHT_PX   = 5.0
_CYCLE_R_TOL_BORDERLINE = 0.15
_CYCLE_T_TOL_BORDERLINE_PX = 25.0


# ══════════════════════════════════════════════════════════════════════════
#  SE(2) helpers
# ══════════════════════════════════════════════════════════════════════════

def _affine_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M_ = np.eye(3, dtype=np.float64)
    M_[:2, :2] = R
    M_[:2,  2] = np.asarray(t, dtype=np.float64).reshape(2)
    return M_


def _Rt_from_affine(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return A[:2, :2].copy(), A[:2, 2].copy()


def _invert_se2(A: np.ndarray) -> np.ndarray:
    R, t = _Rt_from_affine(A)
    R_inv = R.T
    return _affine_from_Rt(R_inv, -R_inv @ t)


def _apply_affine(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    return (A @ homog.T).T[:, :2]


# ══════════════════════════════════════════════════════════════════════════
#  Union-Find
# ══════════════════════════════════════════════════════════════════════════

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.members = {i: [i] for i in range(n)}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> tuple[int, int, int]:
        """Merge clusters; return (new_root, root_kept, root_absorbed)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra, ra, ra
        # Always keep the larger cluster as the root (smaller absorbed)
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].extend(self.members[rb])
        absorbed = self.members.pop(rb)
        # absorbed already added to members[ra] above
        return ra, ra, rb

    def cluster_members(self, x: int) -> list[int]:
        return list(self.members[self.find(x)])

    def roots(self) -> list[int]:
        return [r for r in self.members.keys()]


# ══════════════════════════════════════════════════════════════════════════
#  Candidate enumeration
# ══════════════════════════════════════════════════════════════════════════

_PAPER_LAB_DELTA_MAX = 35.0
_BOUNDARY_PROXIMITY_PX = 200.0


def _has_torn_edge(frag: dict) -> bool:
    return any(e.get("is_torn") for e in frag.get("edges", []) or [])


def _paper_lab_delta(a: dict, b: dict) -> float:
    la = a.get("paper_lab"); lb = b.get("paper_lab")
    if la is None or lb is None:
        return 0.0   # neutral when missing
    return float(np.linalg.norm(np.asarray(la) - np.asarray(lb)))


def _boundary_proximity(a: dict, b: dict) -> float:
    ba = a.get("boundary_pixels"); bb = b.get("boundary_pixels")
    if ba is None or bb is None or len(ba) == 0 or len(bb) == 0:
        return float("inf")
    return float(cKDTree(np.asarray(ba, dtype=np.float64))
                 .query(np.asarray(bb, dtype=np.float64), k=1)[0]
                 .min())


def _enumerate_candidates(fragments: list[dict]) -> list[tuple[int, int]]:
    """All (i, j) pairs i<j where both frags have torn edges, paper-LAB
    is compatible, and boundary pixels come within ``_BOUNDARY_PROXIMITY_PX``.
    """
    n = len(fragments)
    out: list[tuple[int, int]] = []
    for i in range(n):
        if not _has_torn_edge(fragments[i]):
            continue
        for j in range(i + 1, n):
            if not _has_torn_edge(fragments[j]):
                continue
            if _paper_lab_delta(fragments[i], fragments[j]) > _PAPER_LAB_DELTA_MAX:
                continue
            if _boundary_proximity(fragments[i], fragments[j]) > _BOUNDARY_PROXIMITY_PX:
                continue
            out.append((i, j))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Match cache
# ══════════════════════════════════════════════════════════════════════════

class _MatchCache:
    """Caches match_pair results keyed (lo, hi) where lo, hi are FRAGMENT
    INDICES into the fragments list (NOT frag["id"]). Always returns the
    pose oriented as (frag_lo -> frag_hi, R applies to frag_hi)."""

    def __init__(self, fragments: list[dict], image_rgb: np.ndarray):
        self._fragments = fragments
        self._image = image_rgb
        self._cache: dict[tuple[int, int], dict | None] = {}

    @staticmethod
    def _key(i: int, j: int) -> tuple[int, int]:
        return (min(i, j), max(i, j))

    def get(self, i: int, j: int) -> dict | None:
        return self._cache.get(self._key(i, j))

    def put(self, i: int, j: int, m: dict | None) -> None:
        self._cache[self._key(i, j)] = m

    def match(self, i: int, j: int) -> dict | None:
        """Compute or fetch the match between fragment-indices i and j.

        The returned dict's R, t are oriented (j -> i): R @ p_j + t = p_i.
        match_pair already returns this orientation when we pass
        (frag_a=fragments[i], frag_b=fragments[j]).
        """
        key = self._key(i, j)
        if key in self._cache:
            return self._cache[key]
        m = M.match_pair(self._fragments[key[0]], self._fragments[key[1]],
                         self._image)
        # When key reorders (i > j), m.R, m.t map fragments[key[1]] (was j)
        # onto fragments[key[0]] (was i). That's still consistent with the
        # cache contract: "match for (lo, hi) means hi -> lo".
        self._cache[key] = m
        return m

    def items(self):
        return self._cache.items()

    def cached_pairs_between(self, members_a: list[int], members_b: list[int],
                              restrict_to: set | None = None
                              ) -> list[tuple[int, int, dict]]:
        """Return cached matches whose (lo, hi) pair has one member in A
        and the other in B. If ``restrict_to`` is given (as a set of
        (lo, hi) tuples), only those keys are returned — used by the
        cluster-consistency check and BA so they consider ONLY trusted
        merge-accepted edges, never false-positive cached candidates."""
        sa, sb = set(members_a), set(members_b)
        out = []
        for (lo, hi), m in self._cache.items():
            if restrict_to is not None and (lo, hi) not in restrict_to:
                continue
            if m is None or m.get("reject_reason") is not None:
                continue
            if (lo in sa and hi in sb) or (lo in sb and hi in sa):
                out.append((lo, hi, m))
        return out


# ══════════════════════════════════════════════════════════════════════════
#  Pose application + cluster moves
# ══════════════════════════════════════════════════════════════════════════

def _compose_match(transforms: dict[int, np.ndarray],
                   anchor_idx: int, free_idx: int,
                   match: dict, lo: int) -> np.ndarray:
    """Compute the new global transform for ``free_idx`` such that the
    match's relative pose holds.

    ``match`` was retrieved with cache key (lo, hi) where lo = min(i, j).
    Its (R, t) maps fragment ``hi`` onto fragment ``lo``. So:

        if anchor=lo, free=hi:    T_free_new = T_anchor @ M_match
        if anchor=hi, free=lo:    T_free_new = T_anchor @ inv(M_match)
    """
    R = match["R"]; t = match["t"]
    M_lo_to_hi_inv = _affine_from_Rt(R, t)   # hi -> lo
    if anchor_idx == lo:
        return transforms[anchor_idx] @ M_lo_to_hi_inv
    else:
        return transforms[anchor_idx] @ _invert_se2(M_lo_to_hi_inv)


def _global_relative_pose(transforms: dict[int, np.ndarray],
                           i_idx: int, j_idx: int) -> np.ndarray:
    """Pose mapping fragment j's local frame onto fragment i's local frame
    USING THE CURRENT GLOBAL TRANSFORMS. Useful for cycle consistency."""
    return _invert_se2(transforms[i_idx]) @ transforms[j_idx]


def _move_cluster(transforms: dict[int, np.ndarray],
                  members: list[int], D: np.ndarray) -> None:
    """Pre-compose every member's transform by D."""
    for k in members:
        transforms[k] = D @ transforms[k]


# ══════════════════════════════════════════════════════════════════════════
#  Cluster consistency check
# ══════════════════════════════════════════════════════════════════════════

def _cluster_consistency(cache: _MatchCache,
                          members_a: list[int], members_b: list[int],
                          fragments: list[dict],
                          transforms: dict[int, np.ndarray],
                          accepted_pairs: set,
                          slack: float = _CLUSTER_CONSISTENCY_SLACK,
                          ) -> tuple[bool, str]:
    """For every ACCEPTED-AT-MERGE match between cluster A and cluster B,
    verify fit_cost stays acceptable under the *current* global transforms.

    We restrict to merge-accepted pairs (not all cached) so a stale
    false-positive candidate cannot block a legitimate merge.
    """
    cap = float(cfg.MAX_ATTACH_COST) * slack
    for (lo, hi, m) in cache.cached_pairs_between(members_a, members_b,
                                                   restrict_to=accepted_pairs):
        # Recompute fit_cost under current global frames.
        # Map fragment hi's torn edge into fragment lo's frame as
        # implied by global transforms, then evaluate edge fit.
        T_lo = transforms[lo]
        T_hi = transforms[hi]
        # Pose hi -> lo under current globals = inv(T_lo) @ T_hi
        rel = _invert_se2(T_lo) @ T_hi
        R_rel, t_rel = _Rt_from_affine(rel)
        edge_lo = fragments[lo]["edges"][m["edge_i"] if lo == m["frag_i"] else m["edge_j"]]
        edge_hi = fragments[hi]["edges"][m["edge_j"] if lo == m["frag_i"] else m["edge_i"]]
        try:
            fit = M.evaluate_edge_fit(fragments[lo], fragments[hi],
                                       edge_lo, edge_hi, R_rel, t_rel)
        except Exception:
            continue
        if float(fit["fit_cost"]) > cap:
            return False, f"cluster_consistency_violated_pair({lo},{hi})"
    return True, ""


# ══════════════════════════════════════════════════════════════════════════
#  Cycle consistency
# ══════════════════════════════════════════════════════════════════════════

def _cycle_classify(transforms: dict[int, np.ndarray],
                     i_idx: int, j_idx: int,
                     match: dict, lo: int) -> tuple[str, dict]:
    """Classify a cycle-closing edge by comparing chain pose to candidate
    pose. Returns (label, diag) where label is one of:

      "consistent"   - within tight tolerances; cycle is informative confirmation
      "borderline"   - within wider tolerances; chain drift worth BA-correcting
      "false_match"  - way off; the cycle edge is a false positive (ignore)
    """
    chain = _global_relative_pose(transforms, i_idx, j_idx)
    M_lo_to_hi_inv = _affine_from_Rt(match["R"], match["t"])  # hi -> lo
    if i_idx == lo:
        proposed_i_to_j = _invert_se2(M_lo_to_hi_inv)
    else:
        proposed_i_to_j = M_lo_to_hi_inv
    R_chain, t_chain = _Rt_from_affine(chain)
    R_prop, t_prop = _Rt_from_affine(proposed_i_to_j)
    R_diff = float(np.linalg.norm(R_chain - R_prop))
    t_diff = float(np.linalg.norm(t_chain - t_prop))
    diag = {"R_diff": round(R_diff, 4), "t_diff": round(t_diff, 2)}
    if R_diff <= _CYCLE_R_TOL_TIGHT and t_diff <= _CYCLE_T_TOL_TIGHT_PX:
        return "consistent", diag
    if R_diff <= _CYCLE_R_TOL_BORDERLINE and t_diff <= _CYCLE_T_TOL_BORDERLINE_PX:
        return "borderline", diag
    return "false_match", diag


# ══════════════════════════════════════════════════════════════════════════
#  Cluster-scoped bundle adjustment (cost-monotone)
# ══════════════════════════════════════════════════════════════════════════

def _cluster_total_cost(cache: _MatchCache,
                         fragments: list[dict],
                         transforms: dict[int, np.ndarray],
                         members: list[int],
                         accepted_pairs: set | None = None) -> float:
    """Sum of fit_costs across cluster-internal MERGE-ACCEPTED pairs
    (not all cached pairs — false positives in the cache must not poison
    the cost) evaluated under the current global transforms."""
    total = 0.0
    n_pairs = 0
    for (lo, hi, m) in cache.cached_pairs_between(members, members,
                                                   restrict_to=accepted_pairs):
        T_lo = transforms[lo]
        T_hi = transforms[hi]
        rel = _invert_se2(T_lo) @ T_hi
        R_rel, t_rel = _Rt_from_affine(rel)
        edge_lo = fragments[lo]["edges"][m["edge_i"] if lo == m["frag_i"] else m["edge_j"]]
        edge_hi = fragments[hi]["edges"][m["edge_j"] if lo == m["frag_i"] else m["edge_i"]]
        try:
            fit = M.evaluate_edge_fit(fragments[lo], fragments[hi],
                                       edge_lo, edge_hi, R_rel, t_rel)
        except Exception:
            continue
        total += float(fit["fit_cost"])
        n_pairs += 1
    if n_pairs == 0:
        return 0.0
    return float(total)


def _ba_cluster(cache: _MatchCache,
                fragments: list[dict],
                transforms: dict[int, np.ndarray],
                members: list[int],
                pinned: int,
                accepted_pairs: set | None = None,
                ) -> dict:
    """LM bundle adjustment over the cluster, pinning ``pinned``.

    Cost-monotone: snapshots all transforms; if final cost >= initial
    cost, reverts and returns ``{"reverted": True}``.

    Variables: each non-pinned member contributes 3 parameters
    (theta, dx, dy) representing a left-pre-composition adjustment to
    its current global transform.

    Residuals: for every cached match (lo, hi) within the cluster, take
    the SW correspondences ``matched_a`` (in lo's frame) and
    ``matched_b`` (in hi's frame). Predicted lo-frame point is
    ``T_lo'^-1 @ T_hi' @ matched_b``. Residual = predicted - matched_a.
    """
    if len(members) < 2:
        return {"skipped": "single_member"}

    init_cost = _cluster_total_cost(cache, fragments, transforms, members,
                                     accepted_pairs=accepted_pairs)
    snapshot = {k: transforms[k].copy() for k in members}

    free = [k for k in members if k != pinned]
    if not free:
        return {"skipped": "all_pinned"}
    idx_map = {k: i for i, k in enumerate(free)}

    pairs = cache.cached_pairs_between(members, members,
                                        restrict_to=accepted_pairs)
    if not pairs:
        return {"skipped": "no_pairs"}

    def _x_to_transforms(x: np.ndarray) -> dict[int, np.ndarray]:
        t_out = {pinned: snapshot[pinned].copy()}
        for k, idx in idx_map.items():
            theta = float(x[3 * idx + 0])
            dx = float(x[3 * idx + 1])
            dy = float(x[3 * idx + 2])
            c = np.cos(theta); s = np.sin(theta)
            D = np.array([[c, -s, dx],
                          [s,  c, dy],
                          [0,  0, 1.0]], dtype=np.float64)
            t_out[k] = D @ snapshot[k]
        return t_out

    def _residuals(x: np.ndarray) -> np.ndarray:
        T_now = _x_to_transforms(x)
        out = []
        for (lo, hi, m) in pairs:
            mA = np.asarray(m.get("matched_a"), dtype=np.float64)  # in lo's local
            mB = np.asarray(m.get("matched_b"), dtype=np.float64)  # in hi's local
            if mA.size == 0 or mB.size == 0 or len(mA) != len(mB):
                continue
            rel = _invert_se2(T_now[lo]) @ T_now[hi]   # hi -> lo
            pred = _apply_affine(rel, mB)
            out.append((pred - mA).ravel())
        if not out:
            return np.zeros(1, dtype=np.float64)
        return np.concatenate(out)

    x0 = np.zeros(3 * len(free), dtype=np.float64)
    try:
        res = least_squares(
            _residuals, x0, method="lm",
            max_nfev=int(getattr(cfg, "BA_MAX_ITER", 50)) * 3,
            xtol=float(getattr(cfg, "BA_FUNC_TOL", 1e-6)),
        )
    except Exception as exc:
        for k, T in snapshot.items():
            transforms[k] = T
        return {"reverted": True, "reason": f"solver_error:{exc}"}

    proposed = _x_to_transforms(res.x)
    for k, T in proposed.items():
        transforms[k] = T
    final_cost = _cluster_total_cost(cache, fragments, transforms, members,
                                      accepted_pairs=accepted_pairs)
    if not np.isfinite(final_cost) or final_cost >= init_cost:
        for k, T in snapshot.items():
            transforms[k] = T
        return {"reverted": True,
                 "init_cost": round(init_cost, 3),
                 "final_cost": round(final_cost, 3),
                 "reason": "no_improvement"}
    return {"reverted": False,
             "init_cost": round(init_cost, 3),
             "final_cost": round(final_cost, 3),
             "delta": round(init_cost - final_cost, 3),
             "iters": int(getattr(res, "nfev", 0))}


# ══════════════════════════════════════════════════════════════════════════
#  Seam-solver wrapper
# ══════════════════════════════════════════════════════════════════════════

def _seam_refine(cache: _MatchCache,
                 fragments: list[dict],
                 transforms: dict[int, np.ndarray],
                 lo: int, hi: int, m: dict) -> dict:
    """Run seam_solver on a freshly accepted merge; update the cached
    match's R, t and the global transform of the absorbed fragment.

    After refinement, re-evaluate ``evaluate_edge_fit`` at the new (R, t)
    and store the post-seam ``fit_cost`` on the diag dict. Caller uses
    that to decide whether to keep the merge.
    """
    edge_a_idx = m["edge_i"] if m["frag_i"] == lo else m["edge_j"]
    edge_b_idx = m["edge_j"] if m["frag_j"] == hi else m["edge_i"]
    R0, t0 = m["R"], m["t"]
    R_new, t_new, diag = seam_solver.refine_pair(
        fragments[lo], fragments[hi], edge_a_idx, edge_b_idx, R0, t0)
    if diag.get("reason"):
        # solver declined; re-evaluate at original pose for the post-seam
        # quality gate so the caller still gets a fit_cost decision.
        edge_lo = fragments[lo]["edges"][edge_a_idx]
        edge_hi = fragments[hi]["edges"][edge_b_idx]
        try:
            fit = M.evaluate_edge_fit(fragments[lo], fragments[hi],
                                       edge_lo, edge_hi, R0, t0)
            diag["post_seam_fit_cost"] = round(float(fit["fit_cost"]), 3)
        except Exception:
            diag["post_seam_fit_cost"] = float(m.get("fit_cost", float("inf")))
        return diag
    m["R"] = R_new
    m["t"] = t_new
    # Update hi's global transform to reflect the refined relative pose.
    transforms[hi] = transforms[lo] @ _affine_from_Rt(R_new, t_new)
    # Re-evaluate fit_cost at the refined pose.
    edge_lo = fragments[lo]["edges"][edge_a_idx]
    edge_hi = fragments[hi]["edges"][edge_b_idx]
    try:
        fit = M.evaluate_edge_fit(fragments[lo], fragments[hi],
                                   edge_lo, edge_hi, R_new, t_new)
        diag["post_seam_fit_cost"] = round(float(fit["fit_cost"]), 3)
    except Exception:
        diag["post_seam_fit_cost"] = float(m.get("fit_cost", float("inf")))
    return diag


# ══════════════════════════════════════════════════════════════════════════
#  Debug artifact writers
# ══════════════════════════════════════════════════════════════════════════

def _summarise_match(m: dict | None) -> dict:
    if m is None:
        return {"accepted": False, "reason": "matcher_returned_none"}
    if m.get("reject_reason"):
        return {"accepted": False, "reason": m["reject_reason"]}
    return {
        "accepted": True,
        "fit_cost": round(float(m.get("fit_cost", float("inf"))), 3),
        "confidence": round(float(m.get("confidence", 0.0)), 3),
        "rms": round(float(m.get("rms", 0.0)), 2),
        "angle_deg": round(float(np.degrees(m.get("angle", 0.0))), 2),
        "orientation": m.get("orientation", "?"),
        "edge_i": int(m.get("edge_i", -1)),
        "edge_j": int(m.get("edge_j", -1)),
    }


def _save_step_image(fragments: list[dict],
                     transforms: dict[int, np.ndarray],
                     placed: set[int],
                     image_rgb: np.ndarray,
                     out_path: Path,
                     pad: int = _CANVAS_PAD_PX) -> None:
    """Render every placed fragment by warping its mask+image patch by its
    transform. The resulting PNG goes to ``out_path``."""
    if not placed:
        save_image(image_rgb, str(out_path))
        return
    # Compute global bbox.
    mn_x = mn_y = float("inf")
    mx_x = mx_y = float("-inf")
    for k in placed:
        x, y, w, h = fragments[k]["bbox"]
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                           dtype=np.float64)
        warped = _apply_affine(transforms[k], corners)
        mn_x = min(mn_x, float(warped[:, 0].min()))
        mn_y = min(mn_y, float(warped[:, 1].min()))
        mx_x = max(mx_x, float(warped[:, 0].max()))
        mx_y = max(mx_y, float(warped[:, 1].max()))
    cx = int(np.floor(mn_x - pad))
    cy = int(np.floor(mn_y - pad))
    cw = int(np.ceil(mx_x - cx + pad))
    ch = int(np.ceil(mx_y - cy + pad))
    canvas = np.full((ch, cw, 3), 245, dtype=np.uint8)
    for k in placed:
        T = transforms[k].copy()
        T[0, 2] -= cx
        T[1, 2] -= cy
        M_2x3 = T[:2, :].astype(np.float64)
        warped_img = cv2.warpAffine(
            image_rgb, M_2x3, (cw, ch),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(245, 245, 245))
        warped_mask = cv2.warpAffine(
            fragments[k]["mask"], M_2x3, (cw, ch),
            flags=cv2.INTER_NEAREST, borderValue=0) > 127
        canvas[warped_mask] = warped_img[warped_mask]
    save_image(canvas, str(out_path))


# ══════════════════════════════════════════════════════════════════════════
#  Public entry: reconstruct
# ══════════════════════════════════════════════════════════════════════════

def reconstruct(fragments: list[dict],
                image_rgb: np.ndarray,
                debug_dir: Path) -> dict[int, np.ndarray]:
    """Assemble fragments into a global pose graph.

    Returns ``{fragment_index: 3x3 SE(2) affine}``. Unplaced fragments
    keep the identity transform.
    """
    debug_dir = Path(debug_dir)
    recon_debug = debug_dir / "reconstruction"
    recon_debug.mkdir(parents=True, exist_ok=True)

    n = len(fragments)
    transforms: dict[int, np.ndarray] = {
        i: np.eye(3, dtype=np.float64) for i in range(n)}
    placed: set[int] = set()
    merge_log: list[dict] = []
    summary: dict = {}

    try:
        if n < 2:
            logger.info("only %d fragment(s); nothing to assemble", n)
            _write_artifacts(fragments, transforms, placed, merge_log,
                              recon_debug, status="trivial",
                              summary=summary)
            return transforms

        # ── Feature prep ────────────────────────────────────────────
        t0 = time.time()
        M.prepare_edges_and_sdt(fragments, image_rgb=image_rgb)
        logger.info("prepare_edges_and_sdt done in %.2fs", time.time() - t0)

        if cfg.DINOV2_ENABLED:
            try:
                from .appearance import attach_dinov2_features_all
                t0 = time.time()
                attach_dinov2_features_all(fragments, image_rgb)
                logger.info("DINOv2 features attached in %.2fs",
                            time.time() - t0)
            except Exception as exc:
                logger.warning("DINOv2 feature extraction failed: %s — "
                               "engine will use paper-LAB + strip-NCC only",
                               exc)

        cache = _MatchCache(fragments, image_rgb)

        # ── Candidate enumeration ───────────────────────────────────
        t0 = time.time()
        candidates = _enumerate_candidates(fragments)
        logger.info("enumerated %d candidate pairs in %.2fs",
                    len(candidates), time.time() - t0)

        # ── Score every candidate ───────────────────────────────────
        t0 = time.time()
        scored: list[tuple[float, int, int]] = []
        for (i, j) in candidates:
            m = cache.match(i, j)
            entry = {
                "phase": "score", "i": i, "j": j,
                "result": _summarise_match(m),
            }
            merge_log.append(entry)
            if m is None:
                continue
            scored.append((float(m["fit_cost"]), i, j))
        scored.sort()
        logger.info("scored %d candidates -> %d acceptable in %.2fs",
                    len(candidates), len(scored), time.time() - t0)

        if not scored:
            logger.warning("no candidate match was accepted; returning identity")
            summary["status"] = "no_acceptable_pairs"
            _write_artifacts(fragments, transforms, placed, merge_log,
                              recon_debug, status="no_acceptable_pairs",
                              summary=summary)
            return transforms

        # ── Union-Find greedy merge ────────────────────────────────
        uf = _UnionFind(n)
        # The set of (lo, hi) pair keys we have actually committed to as
        # part of the spanning tree. Cluster consistency and BA only
        # consider these — the cache also holds false-positive candidates
        # we explicitly want to ignore for global pose optimisation.
        accepted_pairs: set[tuple[int, int]] = set()

        for fit_cost, i, j in scored:
            ri = uf.find(i); rj = uf.find(j)
            m = cache.get(i, j)
            if m is None:
                continue
            lo = min(i, j)

            # CYCLE: i and j already share a cluster
            if ri == rj:
                label, cdiag = _cycle_classify(transforms, i, j, m, lo)
                if label == "consistent":
                    merge_log.append({"phase": "cycle_consistent",
                                       "i": i, "j": j,
                                       "fit_cost": round(fit_cost, 2),
                                       "diag": cdiag})
                    continue
                if label == "false_match":
                    # Treat as false positive; do NOT BA on a false target.
                    merge_log.append({"phase": "cycle_false_match",
                                       "i": i, "j": j,
                                       "fit_cost": round(fit_cost, 2),
                                       "diag": cdiag})
                    continue
                # Borderline: add this edge to the trusted set and let
                # cluster BA redistribute the small chain drift.
                accepted_pairs.add((min(i, j), max(i, j)))
                members = uf.cluster_members(i)
                ba = _ba_cluster(cache, fragments, transforms,
                                  members, pinned=members[0],
                                  accepted_pairs=accepted_pairs)
                merge_log.append({"phase": "cycle_borderline_ba",
                                   "i": i, "j": j,
                                   "fit_cost": round(fit_cost, 2),
                                   "diag": cdiag, "ba": ba})
                continue

            # CROSS-CLUSTER MERGE: validate consistency with cached pairs.
            members_a = uf.cluster_members(i)
            members_b = uf.cluster_members(j)

            # Anchor = the member whose cluster is currently larger
            # (pose stays the same; the smaller cluster gets transformed).
            if len(members_a) >= len(members_b):
                anchor_idx, free_idx = i, j
                anchor_members, absorbed_members = members_a, members_b
            else:
                anchor_idx, free_idx = j, i
                anchor_members, absorbed_members = members_b, members_a

            # Tentative move: compute D so free's transform satisfies the
            # match, then propagate to absorbed_members.
            T_free_proposed = _compose_match(transforms,
                                              anchor_idx, free_idx, m, lo)
            D = T_free_proposed @ _invert_se2(transforms[free_idx])

            # Tentatively apply.
            snapshot = {k: transforms[k].copy() for k in absorbed_members}
            _move_cluster(transforms, absorbed_members, D)

            # Cluster consistency check (uses only merge-accepted edges).
            ok, reason = _cluster_consistency(
                cache, anchor_members, absorbed_members,
                fragments, transforms, accepted_pairs)
            if not ok:
                # Revert and reject this merge.
                for k, T in snapshot.items():
                    transforms[k] = T
                merge_log.append({"phase": "merge_reject",
                                   "i": i, "j": j,
                                   "fit_cost": round(fit_cost, 2),
                                   "reason": reason})
                continue

            # Tentatively commit so seam refinement has the right context;
            # we may still revert below if post-seam fit_cost is too high.
            tentative_T = {k: transforms[k].copy() for k in absorbed_members}
            tentative_T_anchor = transforms[anchor_idx].copy()

            # Per-attach seam refinement (updates m and transforms[hi]).
            seam_diag = _seam_refine(cache, fragments, transforms,
                                      lo, max(i, j), m)
            post_fit = float(seam_diag.get("post_seam_fit_cost",
                                            fit_cost))

            # Post-seam quality gate: if the refined seam still doesn't
            # actually fit the two edges together, this is a wrong-pair
            # merge dressed up by ICP. Revert.
            post_cap = float(getattr(cfg, "POST_SEAM_FIT_COST_MAX", 15.0))
            if post_fit > post_cap:
                # revert
                transforms[anchor_idx] = tentative_T_anchor
                for k, T in tentative_T.items():
                    transforms[k] = T
                merge_log.append({"phase": "merge_reject",
                                   "i": i, "j": j,
                                   "fit_cost": round(fit_cost, 2),
                                   "post_seam_fit_cost": round(post_fit, 2),
                                   "reason": "post_seam_quality"})
                continue

            # Commit for real.
            uf.union(i, j)
            placed.add(i); placed.add(j)
            placed.update(absorbed_members)
            placed.update(anchor_members)
            accepted_pairs.add((min(i, j), max(i, j)))
            merge_log.append({"phase": "merge",
                               "i": i, "j": j,
                               "fit_cost": round(fit_cost, 2),
                               "post_seam_fit_cost": round(post_fit, 2),
                               "seam": seam_diag})

        # ── Final cluster-scoped BA (cost-monotone) ─────────────────
        for root in uf.roots():
            members = uf.cluster_members(root)
            if len(members) < 2:
                continue
            ba = _ba_cluster(cache, fragments, transforms,
                              members, pinned=members[0],
                              accepted_pairs=accepted_pairs)
            merge_log.append({"phase": "final_ba",
                               "root": root,
                               "members": members,
                               "ba": ba})

        # ── Cluster reconciliation: attempt to bridge separate
        # clusters with their lowest-fit-cost cached pair, relaxed
        # post-seam threshold. This fixes cases where MAX_ATTACH_COST
        # rejected the only true bridge between two otherwise-correct
        # subgraphs (common on real photos where the worst-aligned seam
        # in the document has fit_cost > the cap).
        reconciled = True
        relax_post_cap = float(getattr(cfg, "POST_SEAM_FIT_COST_MAX",
                                         15.0)) * 1.6
        while reconciled:
            reconciled = False
            roots = sorted(uf.roots(),
                            key=lambda r: -len(uf.cluster_members(r)))
            if len(roots) < 2:
                break
            # For every pair of clusters, find the lowest-fit-cost cached
            # cross-cluster pair.
            for ci, ri in enumerate(roots):
                if reconciled:
                    break
                members_i = set(uf.cluster_members(ri))
                for rj in roots[ci + 1:]:
                    members_j = set(uf.cluster_members(rj))
                    bridges: list[tuple[float, int, int, dict]] = []
                    for (lo, hi), m in cache.items():
                        if m is None or m.get("reject_reason") is not None:
                            continue
                        if (lo in members_i and hi in members_j) or \
                                (lo in members_j and hi in members_i):
                            bridges.append((float(m["fit_cost"]), lo, hi, m))
                    if not bridges:
                        continue
                    bridges.sort()
                    fc, lo, hi, m = bridges[0]
                    # Try the merge with relaxed post-seam threshold.
                    if lo in members_i:
                        anchor_idx = lo; free_idx = hi
                        absorbed_root = rj
                    else:
                        anchor_idx = hi; free_idx = lo
                        absorbed_root = rj
                    absorbed_members = (list(members_j) if absorbed_root == rj
                                          else list(members_i))
                    snapshot = {k: transforms[k].copy() for k in absorbed_members}
                    T_free_proposed = _compose_match(transforms,
                                                      anchor_idx, free_idx,
                                                      m, lo)
                    D = T_free_proposed @ _invert_se2(transforms[free_idx])
                    _move_cluster(transforms, absorbed_members, D)

                    seam_diag = _seam_refine(cache, fragments, transforms,
                                              lo, hi, m)
                    post_fit = float(seam_diag.get("post_seam_fit_cost", fc))
                    if post_fit > relax_post_cap:
                        for k, T in snapshot.items():
                            transforms[k] = T
                        merge_log.append({"phase": "reconcile_reject",
                                           "lo": lo, "hi": hi,
                                           "fit_cost": round(fc, 2),
                                           "post_seam_fit_cost": round(post_fit, 2)})
                        continue
                    uf.union(anchor_idx, free_idx)
                    accepted_pairs.add((lo, hi))
                    placed.update(absorbed_members)
                    placed.update(uf.cluster_members(anchor_idx))
                    merge_log.append({"phase": "reconcile_merge",
                                       "lo": lo, "hi": hi,
                                       "fit_cost": round(fc, 2),
                                       "post_seam_fit_cost": round(post_fit, 2)})
                    reconciled = True
                    break

        # ── Orphan retry pass (relaxed thresholds) ──────────────────
        unplaced = [k for k in range(n) if k not in placed]
        if unplaced and placed:
            for k in list(unplaced):
                best = None
                best_anchor = None
                for a in placed:
                    m = cache.match(k, a) if k != a else None
                    if m is None:
                        continue
                    fc = float(m["fit_cost"])
                    if fc > cfg.MAX_ATTACH_COST * 1.5:
                        continue
                    if best is None or fc < best["fit_cost"]:
                        best = m; best_anchor = a
                if best is None:
                    continue
                lo = min(k, best_anchor); hi = max(k, best_anchor)
                free_idx = k
                T_free_proposed = _compose_match(transforms,
                                                  best_anchor, free_idx,
                                                  best, lo)
                transforms[free_idx] = T_free_proposed
                placed.add(k)
                accepted_pairs.add((lo, hi))
                merge_log.append({"phase": "orphan_attach",
                                   "k": k, "anchor": best_anchor,
                                   "fit_cost": round(float(best["fit_cost"]), 2)})

        # ── Final debug step image ──────────────────────────────────
        try:
            _save_step_image(fragments, transforms, placed, image_rgb,
                              recon_debug / "step_final.png")
        except Exception as exc:
            logger.warning("step_final.png failed: %s", exc)

        summary["status"] = "ok"
        summary["n_fragments"] = n
        summary["n_placed"] = len(placed)
        summary["n_clusters"] = sum(1 for r in uf.roots()
                                      if len(uf.cluster_members(r)) >= 2)
        summary["scored_pairs"] = len(scored)
        _write_artifacts(fragments, transforms, placed, merge_log,
                          recon_debug, status="ok", summary=summary)
        logger.info("reconstruct: placed %d / %d fragments",
                    len(placed), n)
        return transforms
    finally:
        _release_dinov2(fragments)


# ══════════════════════════════════════════════════════════════════════════
#  Cleanup + artifact writers
# ══════════════════════════════════════════════════════════════════════════

def _release_dinov2(fragments: list[dict]) -> None:
    for f in fragments:
        f.pop("dinov2", None)


def _write_artifacts(fragments: list[dict],
                     transforms: dict[int, np.ndarray],
                     placed: set[int],
                     merge_log: list[dict],
                     recon_debug: Path,
                     status: str,
                     summary: dict) -> None:
    # merge_log
    with open(recon_debug / "merge_log.json", "w") as f:
        json.dump(merge_log, f, indent=2, default=_jsonable)

    # final_transforms.json — full 3x3 per fragment id
    out_transforms: dict[str, list[list[float]]] = {}
    out_translations: list[dict] = []
    for k, T in transforms.items():
        fid = fragments[k]["id"]
        out_transforms[str(fid)] = [list(map(float, row)) for row in T.tolist()]
        angle, tx, ty = M.affine_angle_translation(T)
        out_translations.append({
            "id": int(fid),
            "placed": int(k in placed),
            "angle_deg": round(float(np.degrees(angle)), 3),
            "dx": round(float(tx), 3),
            "dy": round(float(ty), 3),
        })
    with open(recon_debug / "final_transforms.json", "w") as f:
        json.dump(out_transforms, f, indent=2)
    with open(recon_debug / "final_translations.json", "w") as f:
        json.dump(out_translations, f, indent=2)

    # assembly_summary.json
    summary = dict(summary)
    summary.setdefault("status", status)
    summary.setdefault("n_fragments", len(fragments))
    summary.setdefault("n_placed", len(placed))
    summary["placed"] = sorted(int(fragments[k]["id"]) for k in placed)
    with open(recon_debug / "assembly_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


__all__ = ["reconstruct"]
