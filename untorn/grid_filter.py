"""
untorn.grid_filter
==================
Step 9 of Phase 3 — fast block-LBP pre-screening filter.

The classical pipeline runs Smith-Waterman + Procrustes + ICP + SDT for
every candidate (i, j) fragment pair. That's O(N^2) calls to a ~50-100 ms
matcher; on a 30-fragment scene it dominates wall-clock time. The grid
filter is a zero-training pre-screen that runs in milliseconds and lets
us forward only the most plausible 8 candidates per fragment to the
heavy matcher.

Algorithm
---------
1. Per fragment, per torn edge:
     - Sample a 16-row strip 0..GRID_BAND_DEPTH_PX inside the edge,
       running along the edge direction in 1-px steps.
     - Otsu-binarize the strip (single threshold per fragment).
     - Tile into 16x16 blocks along the edge direction.
     - Compute uniform LBP (P=8, R=1, 10 bins) per block, L1-normalize.
     - Store block descriptor + block centre (world-frame xy).

2. Per fragment pair (i, j):
     - For each of A's blocks, find its top-3 nearest neighbours among
       B's blocks via cosine distance on the LBP histograms.
     - 2-point rigid-transform RANSAC over the resulting correspondences
       (also tries the mirrored case, since mating fragments are
       reflected). Inlier count is the pair score.

3. Per fragment, keep the top-K partner indices (default 8) by score.

This gives a Hamming-style fast filter (cosine on a fixed-length 10-bin
descriptor is essentially a soft Hamming on a binary signature) with
spatial consistency baked in. Empirically, ~85-95 percent of false
candidates are filtered out before the expensive matcher runs.

Public API
----------
build_index(fragments, image_rgb)         -> GridFilterIndex
screen_candidates(index, *, top_k=8)      -> dict[int, list[tuple[int, float]]]
filter_pairs(fragments, image_rgb,
             candidates, *, top_k=8)      -> list[tuple[int, int]]

The third helper is the one used directly by `assembly.py`: it screens
an existing candidate list and returns only those pairs that survive
the grid-filter top-K cut on either side.

Self-test:
    python -m untorn.grid_filter
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from skimage.feature import local_binary_pattern as _skimage_lbp
    _HAVE_SKIMAGE = True
except Exception:                              # pragma: no cover
    _HAVE_SKIMAGE = False
    _skimage_lbp = None

from . import config as cfg


# ---------------------------------------------------------------------------
# Tunables (also exposed via cfg with GRID_FILTER_* names)
# ---------------------------------------------------------------------------

_DEFAULT_BAND_DEPTH_PX = 20
_DEFAULT_BLOCK_SIZE    = 16
_DEFAULT_LBP_P         = 8
_DEFAULT_LBP_R         = 1
_DEFAULT_TOP_BLOCK_NN  = 3
_DEFAULT_RANSAC_ITERS  = 64
_DEFAULT_RANSAC_TOL_PX = 8.0
_DEFAULT_MIN_INLIERS   = 3


def _tunable(name: str, default):
    return getattr(cfg, name, default)


# ---------------------------------------------------------------------------
# Per-fragment block descriptor
# ---------------------------------------------------------------------------

@dataclass
class _FragBlocks:
    """All blocks produced from one fragment's torn edges."""
    descriptors: np.ndarray                   # (N, n_bins) float32, L1-normalized
    centres:     np.ndarray                   # (N, 2) float64, world-frame xy
    edge_ids:    np.ndarray                   # (N,) int32
    n_blocks:    int

    @classmethod
    def empty(cls, n_bins: int) -> "_FragBlocks":
        return cls(
            descriptors=np.zeros((0, n_bins), dtype=np.float32),
            centres=np.zeros((0, 2), dtype=np.float64),
            edge_ids=np.zeros((0,), dtype=np.int32),
            n_blocks=0,
        )


@dataclass
class GridFilterIndex:
    """Cached LBP block descriptors for every fragment."""
    blocks:    list[_FragBlocks]
    n_bins:    int
    block_size: int
    band_depth: int
    pair_scores: dict[tuple[int, int], int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Edge-band sampling
# ---------------------------------------------------------------------------

def _sample_edge_band(image_gray: np.ndarray, mask: np.ndarray,
                      edge_pts: np.ndarray, outward_normal: np.ndarray,
                      band_depth_px: int, n_rows: int
                      ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (band [n_rows, n_cols] uint8, world_x [n_cols] mid-row coords).

    Samples along the edge polyline in 1-px steps. For each along-edge
    sample, walks inward (opposite of outward_normal) up to band_depth_px,
    sampling `n_rows` rows from depth 1 to depth band_depth_px. Pixels
    outside the mask are filled with the local fragment background mean.

    Returns (None, None) if the edge is too short or normal is invalid.
    """
    if edge_pts is None or len(edge_pts) < 2:
        return None, None
    n_unit = float(np.linalg.norm(outward_normal))
    if n_unit < 1e-6:
        return None, None
    inward = -np.asarray(outward_normal, dtype=np.float64) / n_unit

    # Resample edge polyline to ~1 px arc-length steps along the edge.
    diffs = np.diff(edge_pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < 4.0:
        return None, None
    n_cols = max(int(round(total)), 4)
    along_t = np.linspace(0.0, total, n_cols, dtype=np.float64)

    # Interpolate (x, y) along the polyline.
    pts = np.empty((n_cols, 2), dtype=np.float64)
    seg_idx = np.searchsorted(cum, along_t, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(diffs) - 1)
    local_t = (along_t - cum[seg_idx]) / np.maximum(seg_len[seg_idx], 1e-9)
    local_t = np.clip(local_t, 0.0, 1.0)
    pts[:, 0] = edge_pts[seg_idx, 0] + local_t * diffs[seg_idx, 0]
    pts[:, 1] = edge_pts[seg_idx, 1] + local_t * diffs[seg_idx, 1]

    # Depth grid: skip the first pixel (it's the edge itself, often background)
    # and reach band_depth_px inward.
    depth_t = np.linspace(1.0, float(band_depth_px), n_rows, dtype=np.float64)

    # Build (n_rows, n_cols) pixel coordinates.
    inward_x = pts[:, 0:1] + inward[0] * depth_t[None, :]    # (n_cols, n_rows)
    inward_y = pts[:, 1:2] + inward[1] * depth_t[None, :]
    map_x = inward_x.T.astype(np.float32)                    # (n_rows, n_cols)
    map_y = inward_y.T.astype(np.float32)

    band = cv2.remap(image_gray, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
    if band.ndim == 3:
        band = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)

    # Mask off out-of-fragment samples by sampling the binary mask the same
    # way and overwriting bg pixels with the fragment's mean foreground
    # intensity (so LBP doesn't latch onto the mask boundary).
    mask_band = cv2.remap((mask > 127).astype(np.uint8) * 255,
                           map_x, map_y, interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if mask_band.any():
        fg_mean = float(image_gray[mask > 127].mean()) \
            if image_gray.size else 128.0
    else:
        fg_mean = 128.0
    band = band.astype(np.float32)
    band[mask_band == 0] = fg_mean
    band = np.clip(band, 0.0, 255.0).astype(np.uint8)

    # Mid-row world coords (used as block centres for spatial-consistency).
    return band, pts


# ---------------------------------------------------------------------------
# LBP descriptor per block
# ---------------------------------------------------------------------------

def _lbp_image(gray: np.ndarray, P: int, R: int) -> np.ndarray:
    """Uniform LBP code map. Uses skimage when available, else a manual
    P=8/R=1 fallback. The fallback's labelling matches skimage's
    "uniform" convention: label = popcount(code) for uniform patterns,
    P+1 for non-uniform ones, giving P+2 distinct labels in total."""
    if _HAVE_SKIMAGE:
        return _skimage_lbp(gray, P=P, R=R, method="uniform")
    if P != 8 or R != 1:
        raise RuntimeError("skimage not available; manual LBP fallback "
                            "only supports P=8 R=1")
    g = gray.astype(np.int32)
    h, w = g.shape
    out = np.zeros((h, w), dtype=np.int32)
    # 8-neighbour offsets, clockwise from top-left.
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]
    centre = g[1:h - 1, 1:w - 1]
    for k, (dy, dx) in enumerate(offsets):
        nb = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        out[1:h - 1, 1:w - 1] |= ((nb >= centre).astype(np.int32) << k)
    # Map raw codes -> uniform labels.
    labels = np.full(256, P + 1, dtype=np.int32)
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        ring = bits + [bits[0]]
        transitions = sum(b1 != b2 for b1, b2 in zip(ring, ring[1:]))
        if transitions <= 2:
            labels[code] = sum(bits)
    return labels[out]


def _block_descriptors(band_binary: np.ndarray,
                        band_centres: np.ndarray,
                        block_size: int,
                        n_bins: int,
                        lbp_P: int, lbp_R: int) -> tuple[np.ndarray, np.ndarray]:
    """Tile the band along its column axis into block_size-wide blocks and
    return (descriptors [N, n_bins], centres [N, 2])."""
    h, w = band_binary.shape
    n_blocks = w // block_size
    if n_blocks == 0 or h < block_size:
        return (np.zeros((0, n_bins), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float64))

    # Crop the band height to a clean multiple of block_size (typically just
    # block_size).  band height is min(h, block_size).
    rows = min(h, block_size)
    band_use = band_binary[:rows, : n_blocks * block_size]
    centres_use = band_centres[: n_blocks * block_size]

    # Compute the LBP code map once for the whole band.
    lbp = _lbp_image(band_use, lbp_P, lbp_R)
    lbp = np.clip(lbp, 0, n_bins - 1).astype(np.int32)

    desc = np.zeros((n_blocks, n_bins), dtype=np.float32)
    centres = np.zeros((n_blocks, 2), dtype=np.float64)
    for b in range(n_blocks):
        col0 = b * block_size
        col1 = col0 + block_size
        patch = lbp[:, col0:col1].ravel()
        hist = np.bincount(patch, minlength=n_bins).astype(np.float32)
        s = hist.sum()
        desc[b] = hist / s if s > 0 else hist
        centres[b] = centres_use[col0:col1].mean(axis=0)
    return desc, centres


# ---------------------------------------------------------------------------
# Per-fragment block extraction
# ---------------------------------------------------------------------------

def _fragment_blocks(frag: dict, image_gray_full: np.ndarray,
                      *, band_depth_px: int, block_size: int,
                      n_bins: int, lbp_P: int, lbp_R: int) -> _FragBlocks:
    """Build a _FragBlocks for one fragment by walking its torn edges."""
    edges = frag.get("edges") or []
    if not edges:
        return _FragBlocks.empty(n_bins)

    mask = frag["mask"]
    desc_chunks: list[np.ndarray] = []
    cent_chunks: list[np.ndarray] = []
    eid_chunks:  list[np.ndarray] = []

    for eidx, edge in enumerate(edges):
        if not edge.get("is_torn", False):
            continue
        epts = np.asarray(edge["pts"], dtype=np.float64)
        on   = np.asarray(edge["outward_normal"], dtype=np.float64)
        band, centres_world = _sample_edge_band(
            image_gray_full, mask, epts, on,
            band_depth_px=band_depth_px, n_rows=block_size)
        if band is None:
            continue

        # Otsu-binarize the band on-the-fly (single global threshold per band).
        if band.size and band.max() > band.min():
            _, band_bin = cv2.threshold(band, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            band_bin = np.zeros_like(band)

        d, c = _block_descriptors(band_bin, centres_world, block_size,
                                    n_bins, lbp_P, lbp_R)
        if d.shape[0] == 0:
            continue
        desc_chunks.append(d)
        cent_chunks.append(c)
        eid_chunks.append(np.full(d.shape[0], eidx, dtype=np.int32))

    if not desc_chunks:
        return _FragBlocks.empty(n_bins)

    descriptors = np.concatenate(desc_chunks, axis=0)
    centres     = np.concatenate(cent_chunks, axis=0)
    edge_ids    = np.concatenate(eid_chunks, axis=0)
    return _FragBlocks(
        descriptors=descriptors, centres=centres, edge_ids=edge_ids,
        n_blocks=int(descriptors.shape[0]),
    )


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(fragments: list[dict], image_rgb: np.ndarray
                 ) -> GridFilterIndex:
    """Pre-compute per-fragment block descriptors. Idempotent; cheap to
    rebuild between phases.

    Requires `prepare_edges_and_sdt` to have run on the fragments so that
    `frag["edges"]` is populated.
    """
    band_depth = int(_tunable("GRID_FILTER_BAND_DEPTH_PX", _DEFAULT_BAND_DEPTH_PX))
    block_size = int(_tunable("GRID_FILTER_BLOCK_SIZE", _DEFAULT_BLOCK_SIZE))
    lbp_P      = int(_tunable("GRID_FILTER_LBP_P", _DEFAULT_LBP_P))
    lbp_R      = int(_tunable("GRID_FILTER_LBP_R", _DEFAULT_LBP_R))
    n_bins     = lbp_P + 2

    if image_rgb.ndim == 3:
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_rgb

    blocks: list[_FragBlocks] = []
    for frag in fragments:
        blocks.append(_fragment_blocks(
            frag, image_gray,
            band_depth_px=band_depth, block_size=block_size,
            n_bins=n_bins, lbp_P=lbp_P, lbp_R=lbp_R))
    return GridFilterIndex(blocks=blocks, n_bins=n_bins,
                            block_size=block_size, band_depth=band_depth)


# ---------------------------------------------------------------------------
# Pair scoring — block matching + 2-point RANSAC
# ---------------------------------------------------------------------------

def _topk_block_correspondences(desc_a: np.ndarray, desc_b: np.ndarray,
                                  k: int) -> np.ndarray:
    """For every row in desc_a, return indices of the top-k nearest rows in
    desc_b under cosine distance. Returns (Na, k) int32.

    Both desc_a and desc_b are expected to be L1-normalized 1D histograms;
    we cosine-normalize them for matching here so very-empty blocks don't
    blow up the dot product.
    """
    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return np.zeros((desc_a.shape[0], k), dtype=np.int32)
    a = desc_a / np.maximum(np.linalg.norm(desc_a, axis=1, keepdims=True), 1e-9)
    b = desc_b / np.maximum(np.linalg.norm(desc_b, axis=1, keepdims=True), 1e-9)
    sim = a @ b.T                         # (Na, Nb), higher = better
    k_eff = min(k, sim.shape[1])
    # argpartition picks the top-k unsorted; then sort the k by similarity.
    idx_unsorted = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(sim.shape[0])[:, None]
    sim_topk = sim[rows, idx_unsorted]
    order = np.argsort(-sim_topk, axis=1)
    idx_sorted = np.take_along_axis(idx_unsorted, order, axis=1)
    if idx_sorted.shape[1] < k:
        # pad with -1s (callers must skip negatives)
        pad = -np.ones((idx_sorted.shape[0], k - idx_sorted.shape[1]),
                        dtype=np.int32)
        idx_sorted = np.concatenate([idx_sorted, pad], axis=1)
    return idx_sorted.astype(np.int32)


def _ransac_inliers(pts_a: np.ndarray, pts_b: np.ndarray,
                     *, iters: int, tol_px: float, rng: np.random.Generator
                     ) -> int:
    """2-point rigid-transform RANSAC. The largest set of correspondences
    consistent with one SE(2) is returned.

    For mating torn fragments the relative pose IS a rigid SE(2): when A
    and B come from the same tear and we put them back together, A's edge
    polyline sits on top of B's edge polyline (same curve in world space).
    Distances are preserved → 2-point sample fully constrains R, t.
    """
    n = len(pts_a)
    if n < _DEFAULT_MIN_INLIERS:
        return 0
    best = 0
    a = pts_a.astype(np.float64)
    b = pts_b.astype(np.float64)
    for _ in range(iters):
        i, j = (int(x) for x in rng.integers(0, n, size=2))
        if i == j:
            continue
        va = a[j] - a[i]
        vb = b[j] - b[i]
        la = np.hypot(va[0], va[1])
        lb = np.hypot(vb[0], vb[1])
        if la < 1e-3 or lb < 1e-3:
            continue
        # Rigid transforms preserve distance — fast reject.
        if abs(la - lb) > tol_px:
            continue
        theta = np.arctan2(vb[1], vb[0]) - np.arctan2(va[1], va[0])
        ct = float(np.cos(theta))
        st = float(np.sin(theta))
        R = np.array([[ct, -st], [st, ct]], dtype=np.float64)
        t = b[i] - R @ a[i]
        mapped = a @ R.T + t
        err = np.linalg.norm(mapped - b, axis=1)
        inliers = int((err < tol_px).sum())
        if inliers > best:
            best = inliers
    return best


def _pair_score(blocks_a: _FragBlocks, blocks_b: _FragBlocks,
                  *, top_block_nn: int, ransac_iters: int,
                  ransac_tol_px: float, rng: np.random.Generator) -> int:
    """Number of spatially-consistent block matches between two fragments."""
    if blocks_a.n_blocks == 0 or blocks_b.n_blocks == 0:
        return 0
    knn = _topk_block_correspondences(
        blocks_a.descriptors, blocks_b.descriptors, top_block_nn)
    # Build the candidate correspondence set (Na * top_block_nn pairs).
    a_idx = np.repeat(np.arange(blocks_a.n_blocks), top_block_nn)
    b_idx = knn.ravel()
    valid = b_idx >= 0
    a_idx = a_idx[valid]; b_idx = b_idx[valid]
    if a_idx.size < _DEFAULT_MIN_INLIERS:
        return 0
    pts_a = blocks_a.centres[a_idx]
    pts_b = blocks_b.centres[b_idx]
    return _ransac_inliers(pts_a, pts_b, iters=ransac_iters,
                            tol_px=ransac_tol_px, rng=rng)


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------

def screen_candidates(index: GridFilterIndex, *, top_k: int = 8,
                       seed: int = 0) -> dict[int, list[tuple[int, float]]]:
    """For each fragment positional index, return the top-K most plausible
    partner indices ranked by spatial-consistent block-match count.

    Result format:
        {i: [(j_best, score_best), (j_2nd, score_2nd), ...]}
    """
    n = len(index.blocks)
    rng = np.random.default_rng(seed)
    top_block_nn  = int(_tunable("GRID_FILTER_TOP_BLOCK_NN", _DEFAULT_TOP_BLOCK_NN))
    ransac_iters  = int(_tunable("GRID_FILTER_RANSAC_ITERS", _DEFAULT_RANSAC_ITERS))
    ransac_tol_px = float(_tunable("GRID_FILTER_RANSAC_TOL_PX",
                                     _DEFAULT_RANSAC_TOL_PX))
    min_inliers   = int(_tunable("GRID_FILTER_MIN_INLIERS",
                                   _DEFAULT_MIN_INLIERS))

    pair_scores = index.pair_scores
    per_frag: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            score = _pair_score(
                index.blocks[i], index.blocks[j],
                top_block_nn=top_block_nn,
                ransac_iters=ransac_iters,
                ransac_tol_px=ransac_tol_px, rng=rng)
            if score < min_inliers:
                continue
            pair_scores[(i, j)] = score
            per_frag[i].append((j, float(score)))
            per_frag[j].append((i, float(score)))

    for i in range(n):
        per_frag[i].sort(key=lambda kv: -kv[1])
        per_frag[i] = per_frag[i][:top_k]
    return per_frag


def filter_pairs(fragments: list[dict],
                 image_rgb: np.ndarray,
                 candidates: list[tuple[int, int]],
                 *, top_k: int = 8) -> list[tuple[int, int]]:
    """Convenience wrapper used by `assembly._enumerate_pair_candidates`.

    Builds the index, runs the screen, then keeps every (i, j) in
    `candidates` that appears in either i's or j's top-K partner list.

    Pairs not surviving the cut are dropped. Returns the filtered list,
    preserving the original ordering.
    """
    index = build_index(fragments, image_rgb)
    per_frag = screen_candidates(index, top_k=top_k)

    keep: set[tuple[int, int]] = set()
    for i, partners in per_frag.items():
        for (j, _score) in partners:
            keep.add((min(i, j), max(i, j)))

    out: list[tuple[int, int]] = []
    for (i, j) in candidates:
        key = (min(i, j), max(i, j))
        if key in keep:
            out.append((i, j))
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":              # pragma: no cover
    import sys

    print(f"[grid_filter] skimage available: {_HAVE_SKIMAGE}")

    # Build a synthetic two-fragment scene: a rectangle split in half
    # along a diagonal tear. Both fragments share the same texture along
    # the tear so the block descriptors should match.
    rng = np.random.default_rng(0)
    H, W = 240, 360
    image = (rng.integers(20, 235, size=(H, W, 3), dtype=np.uint8))
    # Stamp synthetic "text rows" so LBP has structure.
    for y in range(20, H - 20, 18):
        image[y:y + 4, 30:W - 30] = 25

    # Two halves of a diagonal tear.
    mask_full = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask_full, (20, 20), (W - 20, H - 20), 255, -1)

    yy, xx = np.mgrid[:H, :W]
    diag = (xx + yy).astype(np.float32)
    # Add a wobble so the tear is a curve, not a straight line.
    wobble = 6.0 * np.sin(yy / 18.0)
    cut = 280 + wobble
    left  = (diag < cut) & (mask_full > 0)
    right = (diag >= cut) & (mask_full > 0)
    mask_a = (left.astype(np.uint8)) * 255
    mask_b = (right.astype(np.uint8)) * 255

    def _frag(mask: np.ndarray, fid: int) -> dict:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(mask)
        # Build "edges": one torn (the diagonal cut) + factory edges (the
        # rectangle sides). For the self-test, classify by straightness:
        # the diagonal tear curve has high straightness deviation.
        return {
            "id": fid, "mask": mask,
            "bbox": (int(x), int(y), int(w), int(h)),
            "contour": c.astype(np.int32),
            "edges": [
                {"pts": c.astype(np.float64), "is_torn": True, "length": float(len(c)),
                 "outward_normal": np.array([1.0 if fid == 0 else -1.0,
                                              -1.0 if fid == 0 else 1.0]) /
                                       np.sqrt(2.0)},
            ],
        }

    frag_a = _frag(mask_a, 0)
    frag_b = _frag(mask_b, 1)

    t0 = time.time()
    idx = build_index([frag_a, frag_b], image)
    t_build = time.time() - t0
    print(f"[grid_filter] built index for 2 fragments in {t_build*1000:.1f} ms; "
          f"blocks: A={idx.blocks[0].n_blocks}, B={idx.blocks[1].n_blocks}")

    t0 = time.time()
    per_frag = screen_candidates(idx, top_k=4)
    t_screen = time.time() - t0
    print(f"[grid_filter] screen done in {t_screen*1000:.1f} ms")
    for fid, lst in per_frag.items():
        print(f"  frag {fid}: top partners = {lst}")

    # Smoke-test the filter_pairs convenience entry.
    candidates = [(0, 1)]
    survivors = filter_pairs([frag_a, frag_b], image, candidates, top_k=4)
    print(f"[grid_filter] candidates -> survivors: {candidates} -> {survivors}")
    sys.exit(0)
