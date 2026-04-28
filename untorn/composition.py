"""
untorn.composition
==================
Phase 4 — polygon-clip composition (Step 6).

For every canvas pixel the WINNER is the fragment with the largest
interior-edge-distance, looked up in the warped *interior signed-distance
transform* (`_sdt_interior`, populated by ``fragment_io``). Concretely:

    for each fragment i, warp:
        mask_i           uint8 boolean
        sdt_i            float32 — distance from each pixel to fragment's
                         outline, measured INSIDE fragment i's local frame
                         BEFORE warping (so arbitrary affine warps preserve
                         the relative ordering of "deep" vs "edge" pixels)
        rgb_i            colour-harmonised paper crop

    canvas pixel x:
        winner = argmax_i { sdt_warped_i(x)  if mask_warped_i(x) > 127 }
        canvas[x] = rgb_warped_winner(x)
        coverage[x] = (winner is defined)

Then, for every uncovered canvas pixel inside the document hull whose
distance to the nearest covered pixel is at most
``COMP_SEAM_FILL_MAX_PX``, we copy the colour of that nearest covered
pixel. This is a Voronoi-of-fragments fill: small seam-zone gaps are
*physically* filled from neighbour pixels rather than alpha-blurred or
hallucinated by an inpainter. Anything farther away is left for
``gap_fill`` to classify as a real hole.

Per-fragment LAB harmonisation is preserved from the prior implementation:
each fragment's paper LAB is shifted toward the global median of placed
fragments, with the shift attenuated near ink so text contrast stays
true.

Returns ``{canvas, coverage, gap_mask, crop_bbox}`` — the same contract
the old module had, so pipeline.py and gap_fill.py are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import config as cfg
from .io_utils import save_image


# ══════════════════════════════════════════════════════════════════════════
#  Color harmonisation helpers (kept verbatim from the prior implementation)
# ══════════════════════════════════════════════════════════════════════════

def _fragment_paper_lab(image_rgb: np.ndarray, mask: np.ndarray,
                        ink_thresh: int, safe_margin_px: int
                        ) -> tuple[float, float, float] | None:
    """LAB mean of paper-only pixels for one fragment, or None."""
    if image_rgb.size == 0 or mask.size == 0:
        return None
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    paper = (mask > 127) & (gray >= ink_thresh)
    if safe_margin_px > 0:
        interior = cv2.erode(
            (mask > 127).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=safe_margin_px,
        ) > 0
        paper &= interior
    if int(paper.sum()) < 50:
        return None
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat = lab[paper]
    return (float(flat[:, 0].mean()),
            float(flat[:, 1].mean()),
            float(flat[:, 2].mean()))


def _apply_lab_shift(image_rgb: np.ndarray, mask: np.ndarray,
                     delta_lab: tuple[float, float, float],
                     ink_thresh: int) -> np.ndarray:
    """Add `delta_lab` to LAB channels inside `mask`, attenuated near ink."""
    if image_rgb.size == 0:
        return image_rgb
    dl, da, db = (float(delta_lab[0]),
                  float(delta_lab[1]),
                  float(delta_lab[2]))
    if abs(dl) < 0.1 and abs(da) < 0.1 and abs(db) < 0.1:
        return image_rgb
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    alpha = np.clip((L - ink_thresh) / 60.0, 0.0, 1.0)
    m = (mask > 127).astype(np.float32)
    w = alpha * m
    lab[..., 0] = np.clip(L + dl * w, 0.0, 255.0)
    lab[..., 1] = np.clip(lab[..., 1] + da * w, 0.0, 255.0)
    lab[..., 2] = np.clip(lab[..., 2] + db * w, 0.0, 255.0)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


# ══════════════════════════════════════════════════════════════════════════
#  Interior SDT lookup for the winner-take-all priority
# ══════════════════════════════════════════════════════════════════════════

def _interior_sdt(frag: dict, bx: int, by: int, bw: int, bh: int
                  ) -> np.ndarray:
    """
    Interior SDT for the bbox crop. Uses the cached
    ``frag["_sdt_interior"]`` (populated by fragment_io) when available;
    otherwise computes it from the mask. Outside the mask the SDT is 0,
    so warped non-mask pixels never beat any in-mask pixel for ownership.
    """
    cached = frag.get("_sdt_interior")
    if cached is not None and cached.shape == frag["mask"].shape:
        return cached[by:by + bh, bx:bx + bw].astype(np.float32)
    sub_mask = frag["mask"][by:by + bh, bx:bx + bw]
    fg = (sub_mask > 127).astype(np.uint8) * 255
    return cv2.distanceTransform(fg, cv2.DIST_L2, 3).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  Main entry
# ══════════════════════════════════════════════════════════════════════════

def compose_final(image_rgb: np.ndarray, fragments: list[dict],
                  transforms: dict, debug_dir: Path) -> dict:
    """
    Polygon-clip composition. Returns:
        canvas    HxWx3 uint8 RGB
        coverage  HxW uint8 (0/255) where any pixel is owned by a fragment
                  OR was Voronoi-filled from a near neighbour
        gap_mask  HxW uint8 (or None) for true holes that gap_fill should
                  forward to LaMa
        crop_bbox (x, y, w, h) tight crop of content
    """
    comp_debug = debug_dir / "composition"
    comp_debug.mkdir(parents=True, exist_ok=True)

    n = len(fragments)
    SS = max(1, int(getattr(cfg, "COMP_SUPERSAMPLE", 2)))
    ink_thresh = int(getattr(cfg, "COMP_INK_THRESH", 140))
    lab_enabled = bool(getattr(cfg, "COMP_LAB_HARMONISE_ENABLED", True))
    seam_fill_max_px = float(getattr(cfg, "COMP_SEAM_FILL_MAX_PX", 6.0))

    # ── 1. Canvas bounds ────────────────────────────────────────────────
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for i in range(n):
        Mt = transforms[i]
        bx, by, bw, bh = fragments[i]["bbox"]
        corners = np.array([
            [bx,      by,      1],
            [bx + bw, by,      1],
            [bx + bw, by + bh, 1],
            [bx,      by + bh, 1],
        ], dtype=np.float64)
        tc = (Mt[:2, :] @ corners.T).T
        min_x = min(min_x, tc[:, 0].min())
        min_y = min(min_y, tc[:, 1].min())
        max_x = max(max_x, tc[:, 0].max())
        max_y = max(max_y, tc[:, 1].max())

    pad = 10
    canvas_w = int(max_x - min_x) + 2 * pad
    canvas_h = int(max_y - min_y) + 2 * pad
    ox = -min_x + pad
    oy = -min_y + pad

    # Clamp supersampling against the configured max canvas (memory cap).
    ss_canvas_max = int(getattr(cfg, "OVERLAP_CANVAS_MAX", 12000))
    while SS > 1 and (canvas_w * SS > ss_canvas_max
                      or canvas_h * SS > ss_canvas_max):
        SS -= 1

    cw2 = canvas_w * SS
    ch2 = canvas_h * SS
    print(f"  Canvas: {canvas_w} x {canvas_h}  (supersample {SS}x -> "
          f"{cw2} x {ch2})")

    # ── 2. Per-fragment LAB stats and global target ─────────────────────
    per_frag_lab: list[tuple[float, float, float] | None] = []
    if lab_enabled:
        for i in range(n):
            frag = fragments[i]
            bx, by, bw, bh = frag["bbox"]
            sub_img = image_rgb[by:by + bh, bx:bx + bw]
            sub_mask = frag["mask"][by:by + bh, bx:bx + bw]
            per_frag_lab.append(
                _fragment_paper_lab(sub_img, sub_mask, ink_thresh, 5))
    else:
        per_frag_lab = [None] * n

    known = [lab for lab in per_frag_lab if lab is not None]
    if len(known) >= 2:
        arr = np.asarray(known, dtype=np.float32)
        target_lab = (float(np.median(arr[:, 0])),
                      float(np.median(arr[:, 1])),
                      float(np.median(arr[:, 2])))
    else:
        target_lab = None

    # ── 3. Allocate supersampled buffers ────────────────────────────────
    # canvas_hi: paper-coloured background. best_sdt: depth ranking. We
    # never blend; a pixel either belongs to one fragment (whose RGB
    # wins) or stays paper-coloured for now (Voronoi-filled later).
    canvas_hi = np.full((ch2, cw2, 3), 240, dtype=np.uint8)
    best_sdt = np.zeros((ch2, cw2), dtype=np.float32)   # 0 = "no claim yet"
    owned = np.zeros((ch2, cw2), dtype=bool)             # any fragment claimed

    lab_shifts_applied: dict[int, tuple[float, float, float]] = {}

    # We want every fragment to participate in the priority arbitration
    # — order doesn't change the result because winner-take-all is
    # commutative in argmax. Iterate by index.
    for i in range(n):
        frag = fragments[i]
        Mt = transforms[i].copy()

        bx, by, bw, bh = frag["bbox"]
        sub_mask_full = frag["mask"][by:by + bh, bx:bx + bw]
        m = sub_mask_full > 127
        if not np.any(m):
            continue

        # Masked source crop, with LAB harmonisation.
        sub_img = np.zeros((bh, bw, 3), dtype=np.uint8)
        sub_img[m] = image_rgb[by:by + bh, bx:bx + bw][m]
        lab_mean = per_frag_lab[i]
        if lab_enabled and target_lab is not None and lab_mean is not None:
            delta = (target_lab[0] - lab_mean[0],
                     target_lab[1] - lab_mean[1],
                     target_lab[2] - lab_mean[2])
            sub_img = _apply_lab_shift(sub_img, sub_mask_full, delta,
                                       ink_thresh)
            lab_shifts_applied[i] = delta

        # Per-fragment interior SDT in the FRAGMENT's local frame. Warping
        # this through the same affine preserves the priority (depth) at
        # the pixel level — winner-take-all picks the fragment whose
        # local-frame interior depth is largest at the contested pixel.
        sub_sdt = _interior_sdt(frag, bx, by, bw, bh)

        # Build the supersampled affine that includes the canvas offset
        # and the bbox-relative origin (same trick as before).
        M_ss = np.eye(3, dtype=np.float64)
        M_ss[:2, :2] = SS * Mt[:2, :2]
        M_ss[0, 2]   = SS * (Mt[0, 2] + ox)
        M_ss[1, 2]   = SS * (Mt[1, 2] + oy)
        origin = M_ss @ np.array([bx, by, 1.0], dtype=np.float64)
        M_sub = M_ss.copy()
        M_sub[0, 2] = origin[0]
        M_sub[1, 2] = origin[1]
        M_2x3 = M_sub[:2, :].astype(np.float32)

        # Bilinear warps. Mask is bilinear too so its boundary anti-
        # aliases; we'll threshold below for ownership.
        warped_img = cv2.warpAffine(
            sub_img, M_2x3, (cw2, ch2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        warped_mask = cv2.warpAffine(
            sub_mask_full, M_2x3, (cw2, ch2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_sdt = cv2.warpAffine(
            sub_sdt, M_2x3, (cw2, ch2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        # Winner update: take this fragment when it's in-mask AND its
        # depth is strictly larger than anything claimed before.
        in_frag = warped_mask > 127
        better = in_frag & (warped_sdt > best_sdt)
        if not np.any(better):
            # Even if it doesn't win interior pixels, the fragment may be
            # the only claimant at the seam. Allow it to take any pixel
            # the previous winners haven't actually claimed.
            unclaimed_take = in_frag & (~owned)
            if not np.any(unclaimed_take):
                continue
            canvas_hi[unclaimed_take] = warped_img[unclaimed_take]
            best_sdt[unclaimed_take] = warped_sdt[unclaimed_take]
            owned[unclaimed_take] = True
            continue

        canvas_hi[better] = warped_img[better]
        best_sdt[better] = warped_sdt[better]
        owned[better] = True

    # ── 4. Downsample to 1× canvas ──────────────────────────────────────
    if SS > 1:
        canvas = cv2.resize(canvas_hi, (canvas_w, canvas_h),
                            interpolation=cv2.INTER_AREA)
        # Ownership downsample: a pixel is covered iff a majority of its
        # SS×SS supersample cells were covered.
        owned_u8 = owned.astype(np.uint8) * 255
        owned_lo = cv2.resize(owned_u8, (canvas_w, canvas_h),
                              interpolation=cv2.INTER_AREA)
        coverage = (owned_lo > 127).astype(np.uint8) * 255
    else:
        canvas = canvas_hi
        coverage = (owned.astype(np.uint8) * 255)

    save_image(canvas, str(comp_debug / "01_raw_composite.png"))
    save_image(np.stack([coverage] * 3, axis=-1),
               str(comp_debug / "02_coverage_pre_voronoi.png"))

    # ── 5. Voronoi seam-zone fill ───────────────────────────────────────
    # Build the document hull as a tight bound for "where could content go"
    # (any uncovered pixel outside the hull is just empty space, never a
    # gap). Inside the hull, find uncovered pixels and either Voronoi-fill
    # them from the nearest covered pixel (small seam gaps) or report them
    # to gap_fill as a true hole.
    hull_mask = _document_hull_mask(coverage)
    interior_uncovered = (hull_mask > 0) & (coverage == 0)

    n_voronoi = 0
    n_true_hole = 0
    if interior_uncovered.any():
        # distance_transform_edt operates on a boolean array where TRUE =
        # background and computes distance from each TRUE pixel to the
        # nearest FALSE pixel. We want distance from uncovered to covered,
        # so the input is (coverage == 0).
        gap_input = (coverage == 0)
        dist_from_cov, indices = distance_transform_edt(
            gap_input, return_indices=True)
        fill_mask = interior_uncovered & (dist_from_cov <= seam_fill_max_px)
        true_hole_mask = interior_uncovered & (dist_from_cov > seam_fill_max_px)

        if fill_mask.any():
            ys, xs = np.where(fill_mask)
            src_ys = indices[0, ys, xs]
            src_xs = indices[1, ys, xs]
            canvas[ys, xs] = canvas[src_ys, src_xs]
            coverage[fill_mask] = 255
            n_voronoi = int(fill_mask.sum())

        if true_hole_mask.any():
            n_true_hole = int(true_hole_mask.sum())
            gap_mask = (true_hole_mask.astype(np.uint8) * 255)
            # A 1px close to keep tiny isolated cells from blowing up the
            # repair-mask perimeter; LaMa is happier with cohesive holes.
            kernel = np.ones((3, 3), np.uint8)
            gap_mask = cv2.morphologyEx(gap_mask, cv2.MORPH_CLOSE, kernel)
        else:
            gap_mask = None
    else:
        gap_mask = None

    save_image(np.stack([coverage] * 3, axis=-1),
               str(comp_debug / "03_coverage_post_voronoi.png"))
    if gap_mask is not None:
        save_image(np.stack([gap_mask] * 3, axis=-1),
                   str(comp_debug / "04_gap_mask.png"))

    # ── 6. Crop bbox ────────────────────────────────────────────────────
    rx, ry, rw, rh = _content_crop(canvas, canvas_w, canvas_h)
    crop_bbox = (int(rx), int(ry), int(rw), int(rh))
    print(f"  Composition: {n_voronoi:,} px Voronoi-filled, "
          f"{n_true_hole:,} px true hole; crop {rw}x{rh}")

    meta = {
        "canvas_w": canvas_w, "canvas_h": canvas_h,
        "supersample": SS,
        "offset_x": round(float(ox), 1), "offset_y": round(float(oy), 1),
        "voronoi_filled_px": n_voronoi,
        "true_hole_px": n_true_hole,
        "seam_fill_max_px": seam_fill_max_px,
        "gap_pixels_detected": n_true_hole,    # back-compat for old debug UI
        "crop_bbox": list(crop_bbox),
        "final_w": rw, "final_h": rh,
        "lab_target": (list(map(float, target_lab))
                       if target_lab is not None else None),
        "lab_shifts": {str(k): [round(v[0], 2), round(v[1], 2),
                                round(v[2], 2)]
                       for k, v in lab_shifts_applied.items()},
    }
    with open(comp_debug / "composition_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "canvas":    canvas,
        "coverage":  coverage,
        "gap_mask":  gap_mask,
        "crop_bbox": crop_bbox,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Helpers (private)
# ══════════════════════════════════════════════════════════════════════════

def _document_hull_mask(coverage: np.ndarray) -> np.ndarray:
    """Convex hull of `coverage` as a uint8 mask. Used as the "where could
    document pixels live" bound for gap classification."""
    pts = np.column_stack(np.where(coverage > 0))
    if len(pts) < 3:
        return np.zeros_like(coverage, dtype=np.uint8)
    hull = cv2.convexHull(pts[:, ::-1])
    m = np.zeros(coverage.shape, dtype=np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    return m


def _content_crop(canvas: np.ndarray, canvas_w: int, canvas_h: int
                  ) -> tuple[int, int, int, int]:
    """Tight crop bbox (x, y, w, h) around non-paper pixels of `canvas`."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.vstack(contours)
        rx, ry, rw, rh = cv2.boundingRect(all_pts)
        margin = 15
        rx = max(0, rx - margin)
        ry = max(0, ry - margin)
        rw = min(canvas_w - rx, rw + 2 * margin)
        rh = min(canvas_h - ry, rh + 2 * margin)
    else:
        rx, ry, rw, rh = 0, 0, canvas_w, canvas_h
    return int(rx), int(ry), int(rw), int(rh)
