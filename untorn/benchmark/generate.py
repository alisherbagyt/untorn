"""
untorn.benchmark.generate
=========================
Synthesise torn-paper scan inputs with ground-truth poses.

Pipeline:
    1. Load a clean source image (document, page, etc.)
    2. Scatter N Voronoi seeds inside the image.
    3. Build Voronoi polygons clipped to the image rectangle.
    4. Roughen each polygon edge with midpoint-displacement noise so
       boundaries look torn, not laser-cut.
    5. Mask each region out of the source image → fragment RGBA.
    6. Randomly rotate + translate each fragment onto a dark canvas.
    7. Persist composite scan + ground-truth JSON with per-fragment
       source polygon, centroid, applied rotation, applied translation.

The composite image is exactly what a fresh scan of torn fragments looks
like to the UNTORN pipeline. Ground truth lets us score recovery error
automatically.
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from scipy.spatial import Voronoi


# ───────────────────────────────────────────────────────────────────────────
#  Config
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class TearGeneratorConfig:
    """Parameters that control synthetic tear generation."""

    # Number of fragments to tear the source into
    n_fragments: int = 8

    # Canvas size relative to source: 1.5 -> canvas is 1.5x source in each dim
    canvas_scale: float = 1.6

    # Background RGB colour of the synthetic scan (dark to match real scans)
    bg_color: tuple = (30, 30, 30)

    # Midpoint-displacement iterations for roughening edges (more = more tears)
    roughen_iters: int = 5

    # Max perpendicular noise amplitude (pixels) per midpoint split
    roughen_amplitude: float = 6.0

    # Max rotation per fragment (radians)
    max_rotation_rad: float = math.radians(25.0)

    # Margin between fragments on the canvas (pixels) to avoid overlap
    min_gap_px: int = 20

    # Fraction of a polygon that must lie on the source border to drop it.
    # None disables dropping entirely. 0.75 = keep anything with ≥25% interior.
    drop_border_frac_threshold: Optional[float] = 0.75

    # Resize source so longest edge is this many pixels before tearing.
    # Keeps synthetic scans under WORKING_MAX_DIM (=1500) so Phase-0 doesn't
    # downscale and mangle the coordinate frame. Set to None to disable.
    resize_source_longest_edge: Optional[int] = 900

    # RNG seed (None → non-deterministic)
    seed: Optional[int] = None


# ───────────────────────────────────────────────────────────────────────────
#  Voronoi polygon generation
# ───────────────────────────────────────────────────────────────────────────

def _clip_polygon_to_rect(polygon: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Sutherland-Hodgman clip a polygon to the rectangle [0, w) x [0, h).
    Returns polygon as (N, 2) float32 array, possibly empty.
    """
    def clip_edge(poly, xin, yin, xout, yout, side):
        """Clip polygon to one rect edge."""
        out = []
        if len(poly) == 0:
            return poly
        for i in range(len(poly)):
            curr = poly[i]
            prev = poly[i - 1]
            curr_in = side(curr)
            prev_in = side(prev)
            if curr_in:
                if not prev_in:
                    out.append(_intersect(prev, curr, xin, yin, xout, yout))
                out.append(curr)
            elif prev_in:
                out.append(_intersect(prev, curr, xin, yin, xout, yout))
        return np.asarray(out, dtype=np.float32) if out else np.empty((0, 2),
                                                                     dtype=np.float32)

    def _intersect(p1, p2, xin, yin, xout, yout):
        """Edge intersection with the clip-line (xin,yin)->(xout,yout)."""
        x1, y1 = p1
        x2, y2 = p2
        dx1 = xout - xin
        dy1 = yout - yin
        dx2 = x2 - x1
        dy2 = y2 - y1
        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-12:
            return p1
        t = ((xin - x1) * dy1 - (yin - y1) * dx1) / -denom
        return np.array([x1 + t * dx2, y1 + t * dy2], dtype=np.float32)

    poly = np.asarray(polygon, dtype=np.float32)
    # Left edge x >= 0
    poly = clip_edge(poly, 0, 0, 0, 1, lambda p: p[0] >= 0)
    # Right edge x <= w
    poly = clip_edge(poly, w, 0, w, 1, lambda p: p[0] <= w)
    # Top y >= 0
    poly = clip_edge(poly, 0, 0, 1, 0, lambda p: p[1] >= 0)
    # Bottom y <= h
    poly = clip_edge(poly, 0, h, 1, h, lambda p: p[1] <= h)
    return poly


def _voronoi_polygons(w: int, h: int, n_seeds: int,
                      rng: np.random.Generator) -> list[np.ndarray]:
    """
    Compute Voronoi regions for `n_seeds` random points inside [0,w] x [0,h],
    clipped to the image rectangle.

    Adds 4 ghost seeds at the corners (far outside) so outer regions close.
    """
    # Real seeds inside the image, well-separated via Mitchell best-candidate
    seeds = _mitchell_best_candidate(w, h, n_seeds, rng)

    # Ghost seeds VERY far away so real Voronoi regions stay bounded
    far = 10 * max(w, h)
    ghosts = np.array(
        [[-far, -far], [w + far, -far], [-far, h + far], [w + far, h + far]],
        dtype=np.float64,
    )
    all_seeds = np.vstack([seeds, ghosts])
    vor = Voronoi(all_seeds)

    polys = []
    for region_idx in vor.point_region[: len(seeds)]:
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        poly = vor.vertices[region]
        clipped = _clip_polygon_to_rect(poly, w, h)
        if len(clipped) >= 3:
            polys.append(clipped)
    return polys


def _mitchell_best_candidate(w: int, h: int, n: int,
                              rng: np.random.Generator,
                              n_candidates: int = 20) -> np.ndarray:
    """Mitchell's best-candidate algorithm: picks n well-separated points."""
    pts = [np.array([rng.uniform(0, w), rng.uniform(0, h)])]
    for _ in range(n - 1):
        best = None
        best_dist = -1.0
        for _ in range(n_candidates):
            c = np.array([rng.uniform(0, w), rng.uniform(0, h)])
            d = min(float(np.linalg.norm(c - p)) for p in pts)
            if d > best_dist:
                best_dist = d
                best = c
        pts.append(best)
    return np.asarray(pts, dtype=np.float64)


def _roughen_edge(p1: np.ndarray, p2: np.ndarray,
                  iters: int, amplitude: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Midpoint displacement: split segment p1-p2 into pieces, pushing each
    midpoint perpendicular to the segment by a shrinking random amount.
    Produces a rough tear-like boundary between the two endpoints.
    """
    pts = [p1.astype(np.float32), p2.astype(np.float32)]
    amp = float(amplitude)
    for _ in range(iters):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            mid = 0.5 * (a + b)
            d = b - a
            length = float(np.linalg.norm(d))
            if length < 1.5:
                new_pts.append(b)
                continue
            perp = np.array([-d[1], d[0]], dtype=np.float32) / length
            offset = rng.uniform(-amp, amp)
            new_pts.append(mid + offset * perp)
            new_pts.append(b)
        pts = new_pts
        amp *= 0.55   # subdivisions grow smoother
    return np.asarray(pts, dtype=np.float32)


def _roughen_polygon(poly: np.ndarray,
                     iters: int, amplitude: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Apply midpoint-displacement roughening to every edge of a polygon."""
    if len(poly) < 3:
        return poly
    out = []
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        seg = _roughen_edge(a, b, iters, amplitude, rng)
        # Skip final pt of each segment to avoid duplicating vertices
        out.extend(seg[:-1].tolist())
    return np.asarray(out, dtype=np.float32)


def _border_vertex_fraction(poly: np.ndarray, w: int, h: int,
                             border_tol: float = 2.0) -> float:
    """
    Return the fraction of polygon vertices lying exactly on the source
    border. Polygons with many border vertices are mostly synthesised edge
    rather than real torn edge, so they produce ugly fragments.
    """
    on_border = (
        (poly[:, 0] < border_tol) |
        (poly[:, 0] > w - border_tol) |
        (poly[:, 1] < border_tol) |
        (poly[:, 1] > h - border_tol)
    )
    return float(np.mean(on_border)) if len(poly) else 1.0


# ───────────────────────────────────────────────────────────────────────────
#  Fragment rasterisation + canvas composition
# ───────────────────────────────────────────────────────────────────────────

def _rasterize_fragment(image: np.ndarray, poly: np.ndarray) -> dict:
    """
    Return a cropped RGBA fragment from `image` using `poly` as the mask.
    Output keys:
        rgba (H, W, 4) uint8
        bbox_in_source (x, y, w, h)
        polygon_local (N, 2) float32  polygon in the cropped frame
        centroid_in_source (x, y)     centroid of the polygon in source coords
    """
    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    if mask.sum() == 0:
        return None

    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop_img = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]

    rgba = np.zeros((crop_img.shape[0], crop_img.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = crop_img
    rgba[..., 3] = crop_mask

    poly_local = poly.copy()
    poly_local[:, 0] -= x0
    poly_local[:, 1] -= y0

    M = cv2.moments(crop_mask, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx_local = M["m10"] / M["m00"]
    cy_local = M["m01"] / M["m00"]

    return {
        "rgba": rgba,
        "bbox_in_source": (x0, y0, x1 - x0, y1 - y0),
        "polygon_local": poly_local,
        "polygon_source": poly.copy(),
        "centroid_in_source": (x0 + cx_local, y0 + cy_local),
        "centroid_local": (cx_local, cy_local),
    }


def _place_fragment(canvas: np.ndarray,
                    rgba: np.ndarray,
                    centroid_local: tuple,
                    rotation_rad: float,
                    target_center: tuple) -> np.ndarray:
    """
    Rotate `rgba` by `rotation_rad` around its centroid, then place it on
    `canvas` so that the rotated centroid lands at `target_center`.
    Returns the affine (2x3) that maps rgba-local coords → canvas coords.
    """
    h, w = rgba.shape[:2]
    cx, cy = centroid_local

    # Build the affine in two parts: rotate around (cx,cy), then translate so
    # (cx,cy) lands on target_center.
    cos_t = math.cos(rotation_rad)
    sin_t = math.sin(rotation_rad)
    tx = target_center[0] - (cos_t * cx - sin_t * cy)
    ty = target_center[1] - (sin_t * cx + cos_t * cy)
    M = np.array([[cos_t, -sin_t, tx], [sin_t, cos_t, ty]], dtype=np.float32)

    ch, cw = canvas.shape[:2]
    warped = cv2.warpAffine(rgba, M, (cw, ch),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    alpha = warped[..., 3:4].astype(np.float32) / 255.0
    canvas[...] = (warped[..., :3].astype(np.float32) * alpha +
                   canvas.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return M


def _try_pack(bboxes: list[tuple], canvas_w: int, canvas_h: int,
              gap: int, rng: np.random.Generator,
              attempts_per_frag: int = 800) -> Optional[list[tuple]]:
    """
    Attempt a strictly non-overlapping packing. Fragments are bounded by their
    rotated-bbox inscribed circle (radius = diag/2 + gap). Largest first.
    Returns the list of (cx, cy) centres (same order as input `bboxes`) on
    success, or None if any fragment fails to place.
    """
    order = sorted(range(len(bboxes)),
                   key=lambda i: -math.hypot(bboxes[i][0], bboxes[i][1]))
    placements = [None] * len(bboxes)
    occupied = []
    for idx in order:
        bw, bh = bboxes[idx]
        half = int(math.hypot(bw, bh)) // 2 + gap
        placed = False
        if canvas_w - 2 * half < 1 or canvas_h - 2 * half < 1:
            return None
        for _ in range(attempts_per_frag):
            cx = int(rng.integers(half, canvas_w - half))
            cy = int(rng.integers(half, canvas_h - half))
            if all(math.hypot(cx - ox, cy - oy) >= (half + r)
                   for (ox, oy, r) in occupied):
                placements[idx] = (cx, cy)
                occupied.append((cx, cy, half))
                placed = True
                break
        if not placed:
            return None
    return placements


def _pack_fragments_on_canvas(bboxes: list[tuple],
                              canvas_w: int, canvas_h: int,
                              gap: int, rng: np.random.Generator,
                              allow_grow: bool = True,
                              max_grow_iters: int = 6,
                              ) -> tuple:
    """
    Strictly non-overlapping packer. Grows the canvas (proportionally) if the
    initial size cannot fit all fragments — fragments overlapping on the scan
    is the single biggest source of false "merged" segmentations in earlier
    benchmark runs.

    Returns (canvas_w, canvas_h, placements). `placements[i]` is the (cx, cy)
    centre for fragment i.
    """
    w, h = int(canvas_w), int(canvas_h)
    for it in range(max_grow_iters):
        res = _try_pack(bboxes, w, h, gap, rng)
        if res is not None:
            return w, h, res
        if not allow_grow:
            break
        w = int(round(w * 1.18))
        h = int(round(h * 1.18))

    # Last-ditch: pack what we can, place remainders on the expanded canvas at
    # random (may overlap). Log, don't silently corrupt the scan.
    print(f"    [pack] WARNING: could not fit all {len(bboxes)} fragments "
          f"without overlap even at {w}x{h}; accepting possibly-overlapping "
          f"placement.")
    placements = []
    occupied = []
    for bw, bh in bboxes:
        half = int(math.hypot(bw, bh)) // 2 + gap
        half_w_margin = max(half, 1)
        cx = int(rng.integers(half_w_margin, max(w - half_w_margin,
                                                 half_w_margin + 1)))
        cy = int(rng.integers(half_w_margin, max(h - half_w_margin,
                                                 half_w_margin + 1)))
        placements.append((cx, cy))
        occupied.append((cx, cy, half))
    return w, h, placements


# ───────────────────────────────────────────────────────────────────────────
#  Public API
# ───────────────────────────────────────────────────────────────────────────

def generate_case(source_image: np.ndarray,
                  output_dir: Path,
                  case_id: str,
                  cfg: TearGeneratorConfig = None) -> dict:
    """
    Tear a source image into N fragments and composite them on a canvas.

    Writes:
        {output_dir}/{case_id}/scan.png        -- the synthetic scan input
        {output_dir}/{case_id}/truth.json      -- ground truth
        {output_dir}/{case_id}/source.png      -- copy of the source image
        {output_dir}/{case_id}/fragments/*.png -- each fragment RGBA (debug)

    Returns the ground-truth dict.
    """
    cfg = cfg or TearGeneratorConfig()
    output_dir = Path(output_dir)
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "fragments").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed if cfg.seed is not None
                                else _deterministic_seed(case_id))

    if source_image.dtype != np.uint8:
        source_image = np.clip(source_image, 0, 255).astype(np.uint8)

    # Optional: resize source so the synthetic scan stays under WORKING_MAX_DIM
    if cfg.resize_source_longest_edge is not None:
        h0, w0 = source_image.shape[:2]
        longest = max(h0, w0)
        if longest > cfg.resize_source_longest_edge:
            scale = cfg.resize_source_longest_edge / longest
            source_image = cv2.resize(
                source_image,
                (int(round(w0 * scale)), int(round(h0 * scale))),
                interpolation=cv2.INTER_AREA,
            )
    h, w = source_image.shape[:2]

    # 1. Voronoi polygons (oversample so border-dropping still leaves N)
    n_seeds = max(cfg.n_fragments + 2,
                  int(round(cfg.n_fragments * 1.5)))
    polys = _voronoi_polygons(w, h, n_seeds, rng)

    # 2. Drop border-dominated polygons: rank by border-vertex fraction,
    #    keep the n_fragments polygons with the LEAST border contamination.
    thresh = cfg.drop_border_frac_threshold
    if thresh is not None:
        scored = sorted(polys, key=lambda p: _border_vertex_fraction(p, w, h))
        polys = [p for p in scored
                 if _border_vertex_fraction(p, w, h) < thresh]
    polys = polys[: cfg.n_fragments]

    # 3. Roughen each polygon's edges
    rough_polys = [_roughen_polygon(p, cfg.roughen_iters,
                                     cfg.roughen_amplitude, rng)
                   for p in polys]

    # 4. Rasterise fragments
    rasters = []
    for p in rough_polys:
        r = _rasterize_fragment(source_image, p)
        if r is not None and r["rgba"].shape[0] > 10 and r["rgba"].shape[1] > 10:
            rasters.append(r)

    if len(rasters) < 2:
        raise RuntimeError(f"Only {len(rasters)} usable fragments produced; "
                           f"increase canvas or reduce roughening.")

    # 5. Build the canvas and place fragments.
    # Start from cfg.canvas_scale × source, but let the packer grow the
    # canvas if fragments would otherwise overlap (previously, overlap on
    # the scan caused SAM to merge multiple fragments into one mask, which
    # silently destroyed ground-truth labels during evaluation).
    bboxes = [r["bbox_in_source"][2:] for r in rasters]
    canvas_w0 = int(w * cfg.canvas_scale)
    canvas_h0 = int(h * cfg.canvas_scale)
    canvas_w, canvas_h, placements = _pack_fragments_on_canvas(
        bboxes, canvas_w0, canvas_h0, cfg.min_gap_px, rng, allow_grow=True)
    canvas = np.full((canvas_h, canvas_w, 3), cfg.bg_color, dtype=np.uint8)

    fragments_meta = []
    for idx, (r, target_center) in enumerate(zip(rasters, placements)):
        rot = float(rng.uniform(-cfg.max_rotation_rad, cfg.max_rotation_rad))
        M = _place_fragment(canvas, r["rgba"], r["centroid_local"],
                             rot, target_center)

        # Save fragment RGBA for debugging
        cv2.imwrite(str(case_dir / "fragments" / f"frag_{idx:02d}.png"),
                    cv2.cvtColor(r["rgba"], cv2.COLOR_RGBA2BGRA))

        # Affine from rgba-local coords → canvas coords. We want the map from
        # SOURCE coords → canvas coords for evaluation: local = source - bbox_xy,
        # so canvas = M @ (source - bbox_xy, 1)
        bx, by, bw, bh = r["bbox_in_source"]
        # Combine translation from source → local into the affine.
        # source2local = translate(-bx, -by); local2canvas = M.
        # M_full = M @ T(-bx, -by)
        M_full = np.zeros((2, 3), dtype=np.float64)
        M_full[:, :2] = M[:, :2]
        M_full[:, 2] = M[:, 2] - M[:, :2] @ np.array([bx, by], dtype=np.float64)

        fragments_meta.append({
            "id":                     idx,
            "bbox_in_source":         list(map(int, r["bbox_in_source"])),
            "centroid_in_source":     list(map(float, r["centroid_in_source"])),
            "centroid_local":         list(map(float, r["centroid_local"])),
            "applied_rotation_rad":   rot,
            "applied_rotation_deg":   math.degrees(rot),
            "target_center_on_canvas": list(map(int, target_center)),
            # Forward affine: source coords → canvas coords
            "affine_source_to_canvas": M_full.tolist(),
            # Polygon in SOURCE coordinate frame
            "polygon_source":         r["polygon_source"].astype(float).tolist(),
        })

    # 6. Save outputs.
    # NOTE: the scan file is named with the case_id so that the UNTORN
    # pipeline's per-image debug dir (derived from input stem) is unique
    # across cases in a benchmark run — otherwise every case overwrites
    # `data/debug/scan/` and the evaluator reads stale predictions.
    scan_path   = case_dir / f"{case_id}_scan.png"
    source_path = case_dir / "source.png"
    truth_path  = case_dir / "truth.json"

    cv2.imwrite(str(scan_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(source_path), cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))

    truth = {
        "case_id":       case_id,
        "source_path":   str(source_path),
        "scan_path":     str(scan_path),
        "source_size":   [int(w), int(h)],
        "canvas_size":   [int(canvas_w), int(canvas_h)],
        "n_fragments":   len(fragments_meta),
        "generator_config": asdict(cfg),
        "fragments":     fragments_meta,
    }
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    return truth


def _deterministic_seed(case_id: str) -> int:
    """Turn a case_id string into a 32-bit integer seed."""
    digest = hashlib.md5(case_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")
