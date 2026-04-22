"""
untorn.neighbors
================
Phase 3A: Proximity-based neighbor discovery.

Assumption
----------
Fragments are supplied in a roughly correct initial layout (same scan,
pieces laid out approximately on the document). We exploit that layout
as a prior: real seams only happen between spatially adjacent pieces,
so we never need a global O(n^2) edge search.

What this module produces
-------------------------
For every fragment we attach:

  frag["neighbors"] : list[dict]  one entry per candidate neighbor
      {
        "j":        int,          index into fragments
        "direction":str,          "right"|"left"|"down"|"up"
        "centroid_dist": float,   L2 distance between centroids
        "edge_dist":     float,   minimum boundary-to-boundary distance
        "via":           str,     "knn" | "delaunay" | "both"
      }

The graph is (a) a kNN graph on centroids (k = cfg.NEIGHBOR_K), unioned
with (b) a Delaunay triangulation over centroids. kNN handles "close but
many neighbors" cases; Delaunay is robust for sparse edge pieces on the
perimeter where kNN can miss a diagonal neighbor.

We also classify each neighbor as one of the four cardinal directions
by its offset vector relative to the reference centroid. A neighbor is
only kept if it lies in its direction's 90 degree cone.

All structures use scipy.spatial.cKDTree / Delaunay — zero extra deps.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree, Delaunay

from . import config as cfg


# ── Directional classification ─────────────────────────────────────────────
# A neighbor at offset (dx, dy) falls into one of 4 cones (image coords,
# y positive downwards):
#   right : |dx| >= |dy| and dx > 0
#   left  : |dx| >= |dy| and dx < 0
#   down  : |dy| >  |dx| and dy > 0
#   up    : |dy| >  |dx| and dy < 0

def _direction(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


_OPPOSITE = {"right": "left", "left": "right", "down": "up", "up": "down"}


# ── Boundary distance (edge-to-edge, not centroid) ─────────────────────────

def _min_boundary_distance(pts_a: np.ndarray, pts_b: np.ndarray,
                           tree_b: cKDTree | None = None) -> float:
    """
    Min Euclidean distance from any boundary pixel of A to any boundary
    pixel of B. Queries a pre-built KDTree on B (cheap: O(|A| log |B|)).
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("inf")
    if tree_b is None:
        tree_b = cKDTree(pts_b)
    d, _ = tree_b.query(pts_a, k=1)
    return float(d.min())


# ── Main entry: build the neighbor graph ──────────────────────────────────

def discover_neighbors(fragments: list[dict],
                        debug_dir=None) -> list[dict]:
    """
    Populate frag["neighbors"] for every fragment.

    Args:
        fragments: list of frag dicts with at least "centroid",
                   "boundary_pixels". Mutated in place.
        debug_dir: optional Path for writing neighbors.json.

    Returns the same list for chaining.
    """
    n = len(fragments)
    if n == 0:
        return fragments

    centroids = np.array([f["centroid"] for f in fragments], dtype=np.float64)

    # ── kNN on centroids ──────────────────────────────────────────────
    k = min(cfg.NEIGHBOR_K + 1, n)          # +1 to exclude self
    tree_c = cKDTree(centroids)
    dists, idx = tree_c.query(centroids, k=k)

    # ── Delaunay backbone (robust for edge pieces) ────────────────────
    delaunay_edges: set[tuple[int, int]] = set()
    if n >= 4:
        try:
            tri = Delaunay(centroids)
            for simp in tri.simplices:
                for a, b in ((simp[0], simp[1]),
                             (simp[1], simp[2]),
                             (simp[2], simp[0])):
                    lo, hi = (int(a), int(b)) if a < b else (int(b), int(a))
                    delaunay_edges.add((lo, hi))
        except Exception:
            # Degenerate (collinear) — skip Delaunay and rely on kNN.
            delaunay_edges = set()
    elif n == 3:
        delaunay_edges = {(0, 1), (0, 2), (1, 2)}
    elif n == 2:
        delaunay_edges = {(0, 1)}

    # ── Pre-build boundary KDTrees (for edge distance) ────────────────
    boundary_trees = [cKDTree(f["boundary_pixels"])
                      if len(f.get("boundary_pixels", [])) > 0 else None
                      for f in fragments]

    # ── Union kNN + Delaunay → candidate pair set ─────────────────────
    pair_sources: dict[tuple[int, int], set[str]] = {}

    for i in range(n):
        for rank in range(1, k):                # skip self (rank 0)
            j = int(idx[i, rank])
            lo, hi = (i, j) if i < j else (j, i)
            pair_sources.setdefault((lo, hi), set()).add("knn")

    for (lo, hi) in delaunay_edges:
        pair_sources.setdefault((lo, hi), set()).add("delaunay")

    # ── Build per-fragment neighbor lists ─────────────────────────────
    per_frag: list[list[dict]] = [[] for _ in range(n)]

    for (lo, hi), srcs in pair_sources.items():
        ci = centroids[lo]
        cj = centroids[hi]
        dv = cj - ci
        cd = float(np.linalg.norm(dv))
        if cd < 1e-6:
            continue

        via = "both" if len(srcs) >= 2 else next(iter(srcs))

        # Edge distance gate — reject pairs whose boundaries are far apart.
        ed = _min_boundary_distance(
            fragments[lo]["boundary_pixels"],
            fragments[hi]["boundary_pixels"],
            tree_b=boundary_trees[hi],
        )
        if ed > cfg.NEIGHBOR_MAX_EDGE_DIST_PX:
            continue

        dir_ij = _direction(dv[0], dv[1])         # as seen from lo
        dir_ji = _OPPOSITE[dir_ij]

        per_frag[lo].append({
            "j": hi, "direction": dir_ij,
            "centroid_dist": cd, "edge_dist": ed, "via": via,
        })
        per_frag[hi].append({
            "j": lo, "direction": dir_ji,
            "centroid_dist": cd, "edge_dist": ed, "via": via,
        })

    # Sort each list by edge distance (closest physical neighbor first)
    for lst in per_frag:
        lst.sort(key=lambda d: (d["edge_dist"], d["centroid_dist"]))

    for i, f in enumerate(fragments):
        f["neighbors"] = per_frag[i]

    # ── Debug dump ────────────────────────────────────────────────────
    if debug_dir is not None:
        import json
        from pathlib import Path
        out = Path(debug_dir) / "reconstruction"
        out.mkdir(parents=True, exist_ok=True)
        payload = []
        for i, f in enumerate(fragments):
            payload.append({
                "id": f["id"],
                "centroid": [round(float(f["centroid"][0]), 1),
                             round(float(f["centroid"][1]), 1)],
                "neighbors": [{
                    "j":              fragments[nb["j"]]["id"],
                    "direction":      nb["direction"],
                    "centroid_dist":  round(nb["centroid_dist"], 1),
                    "edge_dist":      round(nb["edge_dist"], 1),
                    "via":            nb["via"],
                } for nb in f["neighbors"]],
            })
        with open(out / "neighbors.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    return fragments


# ── Corner identification ─────────────────────────────────────────────────
# TL = arg min (cx + cy)
# TR = arg min (-cx + cy) == arg max (cx - cy)
# BL = arg max (-cx + cy) == arg min (cx - cy)
# BR = arg max (cx + cy)
#
# This works as long as the rough scan layout is axis-aligned (our
# preprocessing assumption). If the four extrema are not distinct — e.g.,
# fewer than 4 fragments, or a skewed scan — we gracefully degrade: a
# corner may be None and the caller skips that branch.

def identify_corners(fragments: list[dict]) -> dict[str, int | None]:
    """
    Returns {"TL": i, "TR": i, "BL": i, "BR": i} where each value is the
    index of the fragment nearest that corner, or None if the layout
    doesn't support that corner (e.g., n < 4).
    """
    n = len(fragments)
    if n == 0:
        return {"TL": None, "TR": None, "BL": None, "BR": None}

    centroids = np.array([f["centroid"] for f in fragments], dtype=np.float64)
    cx, cy = centroids[:, 0], centroids[:, 1]

    tl = int(np.argmin(cx + cy))
    tr = int(np.argmax(cx - cy))
    bl = int(np.argmin(cx - cy))
    br = int(np.argmax(cx + cy))

    # If fewer than 4 fragments, extrema collide — null out duplicates
    # from BR, BL, TR in that order (keep TL deterministically).
    out = {"TL": tl, "TR": tr, "BL": bl, "BR": br}
    seen = set()
    ordered_keys = ["TL", "TR", "BL", "BR"]
    unique_out: dict[str, int | None] = {}
    for key in ordered_keys:
        val = out[key]
        if val in seen:
            unique_out[key] = None
        else:
            unique_out[key] = val
            seen.add(val)
    return unique_out


__all__ = ["discover_neighbors", "identify_corners"]
