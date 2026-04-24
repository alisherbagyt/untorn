"""
untorn.composition
==================
Phase 4: Compose the final reconstructed image from fragment transforms.

The old composition used a hard integer-pixel warp + `cv2.warpAffine` +
direct pixel assignment. Even when every fragment was aligned correctly
upstream, this produced visible white seams because (a) sub-pixel pose
errors round up into visible gaps, (b) each fragment carries slightly
different paper color and jumps abruptly at every seam, and (c) the
fragment boundary is a hard edge with no feathering.

The new pipeline fixes all three:

  1. **Supersampled warp**. Every fragment is warped onto a canvas that
     is COMP_SUPERSAMPLE times larger than the output, with
     `cv2.INTER_LINEAR`. This gives effectively sub-pixel placement for
     free and anti-aliases the mask boundary.
  2. **Feathered alpha**. Each fragment's alpha is a smoothstep over the
     inward distance-transform, so the hard mask edge becomes a ~1 px
     cosine ramp. Composition is then an alpha blend instead of a hard
     assignment — neighbouring fragments overlap in their feather zones
     and the seam dissolves.
  3. **LAB color harmonisation**. Each fragment's paper-only pixels
     (inside mask, above the ink-threshold, >=5 px from boundary) give a
     per-fragment LAB mean. We shift each fragment's LAB channels to the
     global median, attenuating the shift near dark (ink) pixels so we
     never posterise text.

The function returns the same dict the legacy module did so pipeline.py
is unchanged: canvas, coverage, gap_mask, crop_bbox.
"""

from __future__ import annotations

import json
import numpy as np
import cv2
from pathlib import Path

from . import config as cfg
from .io_utils import save_image


# ══════════════════════════════════════════════════════════════════════════
#  Color harmonisation helpers
# ══════════════════════════════════════════════════════════════════════════

def _fragment_paper_lab(image_rgb: np.ndarray, mask: np.ndarray,
                        ink_thresh: int, safe_margin_px: int
                        ) -> tuple[float, float, float] | None:
    """Return LAB mean of paper-only pixels for one fragment, or None."""
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
    """
    Add `delta_lab` to the LAB channels of `image_rgb` inside `mask`, with
    attenuation near ink so dark strokes don't drift in color.

    Attenuation: linear ramp in L between `ink_thresh` (alpha = 0) and
    `ink_thresh + 60` (alpha = 1). Pixels brighter than that are fully
    shifted.
    """
    if image_rgb.size == 0:
        return image_rgb
    dl, da, db = (float(delta_lab[0]),
                  float(delta_lab[1]),
                  float(delta_lab[2]))
    if abs(dl) < 0.1 and abs(da) < 0.1 and abs(db) < 0.1:
        return image_rgb
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    # Attenuate shift where L is near ink (dark) so text keeps its contrast.
    alpha = np.clip((L - ink_thresh) / 60.0, 0.0, 1.0)
    m = (mask > 127).astype(np.float32)
    w = alpha * m
    lab[..., 0] = np.clip(L            + dl * w, 0.0, 255.0)
    lab[..., 1] = np.clip(lab[..., 1]  + da * w, 0.0, 255.0)
    lab[..., 2] = np.clip(lab[..., 2]  + db * w, 0.0, 255.0)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


# ══════════════════════════════════════════════════════════════════════════
#  Feathered alpha from mask
# ══════════════════════════════════════════════════════════════════════════

def _feathered_alpha(mask01: np.ndarray, feather_px: float) -> np.ndarray:
    """
    Turn a binary mask into a soft alpha that is 1.0 in the interior and
    fades smoothly to 0 inside the last `feather_px` pixels of the
    boundary. Uses a cosine smoothstep on the inward distance-transform.
    """
    if mask01.dtype != np.uint8:
        mask01 = (mask01 > 0).astype(np.uint8)
    if mask01.sum() == 0:
        return np.zeros(mask01.shape, dtype=np.float32)
    dist = cv2.distanceTransform(mask01, cv2.DIST_L2, 3)
    t = np.clip(dist / max(feather_px, 1e-6), 0.0, 1.0)
    # Cosine smoothstep: 0.5*(1 - cos(pi*t))
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  Main entry
# ══════════════════════════════════════════════════════════════════════════

def compose_final(image_rgb: np.ndarray, fragments: list[dict],
                  transforms: dict, debug_dir: Path) -> dict:
    """
    Place every fragment onto a single canvas using its affine transform,
    with supersampled warp, feathered alpha compositing, and per-fragment
    LAB color harmonisation. Returns the same contract the old module did:
        canvas    HxWx3 uint8 RGB
        coverage  HxW uint8 (0/255) where any fragment is placed
        gap_mask  HxW uint8 (may be None) for the inpainter
        crop_bbox (x, y, w, h) tight crop of content
    """
    comp_debug = debug_dir / "composition"
    comp_debug.mkdir(parents=True, exist_ok=True)

    n = len(fragments)
    SS = max(1, int(getattr(cfg, "COMP_SUPERSAMPLE", 2)))
    feather_px = float(getattr(cfg, "COMP_FEATHER_PX", 1.5))
    ink_thresh = int(getattr(cfg, "COMP_INK_THRESH", 140))
    lab_enabled = bool(getattr(cfg, "COMP_LAB_HARMONISE_ENABLED", True))

    # ── 1. Canvas bounds from transformed bboxes ─────────────────────────
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for i in range(n):
        M = transforms[i]
        bx, by, bw, bh = fragments[i]["bbox"]
        corners = np.array([
            [bx,      by,      1],
            [bx + bw, by,      1],
            [bx + bw, by + bh, 1],
            [bx,      by + bh, 1],
        ], dtype=np.float64)
        tc = (M[:2, :] @ corners.T).T
        min_x = min(min_x, tc[:, 0].min())
        min_y = min(min_y, tc[:, 1].min())
        max_x = max(max_x, tc[:, 0].max())
        max_y = max(max_y, tc[:, 1].max())

    pad = 10
    canvas_w = int(max_x - min_x) + 2 * pad
    canvas_h = int(max_y - min_y) + 2 * pad
    ox = -min_x + pad
    oy = -min_y + pad

    # Cap to sane memory. Supersampling quadratically grows the canvas —
    # back off if we'd blow through OVERLAP_CANVAS_MAX.
    ss_canvas_max = int(getattr(cfg, "OVERLAP_CANVAS_MAX", 12000))
    while SS > 1 and (canvas_w * SS > ss_canvas_max
                      or canvas_h * SS > ss_canvas_max):
        SS -= 1

    cw2 = canvas_w * SS
    ch2 = canvas_h * SS
    print(f"  Canvas: {canvas_w} x {canvas_h}  (supersample {SS}x -> "
          f"{cw2} x {ch2})")

    # ── 2. Per-fragment paper-LAB stats and global target ────────────────
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

    # ── 3. Allocate supersampled canvas + alpha ──────────────────────────
    # Canvas is blank (paper-ish off-white). Using 240 as a "close to white"
    # fill means any uncovered area reads as paper, not harsh white.
    canvas_hi = np.full((ch2, cw2, 3), 240, dtype=np.uint8)
    alpha_hi = np.zeros((ch2, cw2), dtype=np.float32)
    # Accumulators for source-over alpha blending
    rgb_acc = np.zeros((ch2, cw2, 3), dtype=np.float32)
    a_acc = np.zeros((ch2, cw2), dtype=np.float32)

    # Sort by area descending so smaller pieces paint LAST (on top).
    # Without this, tiny fragments get buried under large ones that warp
    # across them.
    order = sorted(range(n),
                   key=lambda i: fragments[i].get("area",
                       int((fragments[i]["mask"] > 127).sum())),
                   reverse=True)

    lab_shifts_applied: dict[int, tuple[float, float, float]] = {}

    for i in order:
        frag = fragments[i]
        M = transforms[i].copy()
        # Supersampled affine: we want p_dst_hi = SS * (M @ p_src + (ox, oy))
        # for every source pixel. That means BOTH the linear part and the
        # translation get multiplied by SS:
        #   M_ss[:2,:2] = SS * M[:2,:2]
        #   M_ss[:2, 2] = SS * (M[:2, 2] + (ox, oy))
        M_ss = np.eye(3, dtype=np.float64)
        M_ss[:2, :2] = SS * M[:2, :2]
        M_ss[0, 2]   = SS * (M[0, 2] + ox)
        M_ss[1, 2]   = SS * (M[1, 2] + oy)

        bx, by, bw, bh = frag["bbox"]
        sub_img = np.zeros((bh, bw, 3), dtype=np.uint8)
        sub_mask_full = frag["mask"][by:by + bh, bx:bx + bw]
        m = sub_mask_full > 127
        if not np.any(m):
            continue
        sub_img[m] = image_rgb[by:by + bh, bx:bx + bw][m]

        # Per-fragment LAB color correction
        lab_mean = per_frag_lab[i]
        if lab_enabled and target_lab is not None and lab_mean is not None:
            delta = (target_lab[0] - lab_mean[0],
                     target_lab[1] - lab_mean[1],
                     target_lab[2] - lab_mean[2])
            sub_img = _apply_lab_shift(sub_img, sub_mask_full, delta,
                                        ink_thresh)
            lab_shifts_applied[i] = delta

        # Adjust affine so the sub-image origin is at (bx, by)
        origin = M_ss @ np.array([bx, by, 1.0], dtype=np.float64)
        M_sub = M_ss.copy()
        M_sub[0, 2] = origin[0]
        M_sub[1, 2] = origin[1]
        M_2x3 = M_sub[:2, :].astype(np.float32)

        # Warp image with bilinear interpolation into supersampled canvas
        warped_img = cv2.warpAffine(
            sub_img, M_2x3, (cw2, ch2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        # Warp mask with linear interpolation too — this gives a gentle
        # boundary around the 127 threshold that we can feather.
        warped_mask01 = cv2.warpAffine(
            sub_mask_full, M_2x3, (cw2, ch2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Convert to feathered alpha in [0, 1]
        bin01 = (warped_mask01 > 127).astype(np.uint8)
        alpha_f = _feathered_alpha(bin01, feather_px * SS)

        # Source-over compositing: out = src * alpha_src + dst * (1 - alpha_src)
        a3 = alpha_f[..., None]
        rgb_acc = rgb_acc * (1.0 - a3) + warped_img.astype(np.float32) * a3
        a_acc = a_acc + alpha_f * (1.0 - a_acc)
        alpha_hi = np.maximum(alpha_hi, alpha_f)

    # Blend accumulator onto the paper-colored canvas
    a3 = alpha_hi[..., None]
    canvas_hi_f = canvas_hi.astype(np.float32) * (1.0 - a3) + rgb_acc * a3
    canvas_hi = np.clip(canvas_hi_f, 0, 255).astype(np.uint8)

    # ── 4. Downsample to 1x canvas ───────────────────────────────────────
    if SS > 1:
        canvas = cv2.resize(canvas_hi, (canvas_w, canvas_h),
                            interpolation=cv2.INTER_AREA)
        alpha_lo = cv2.resize(alpha_hi, (canvas_w, canvas_h),
                              interpolation=cv2.INTER_AREA)
    else:
        canvas = canvas_hi
        alpha_lo = alpha_hi

    coverage = (alpha_lo > 0.05).astype(np.uint8) * 255

    save_image(canvas, str(comp_debug / "01_raw_composite.png"))
    save_image(np.stack([coverage] * 3, axis=-1),
               str(comp_debug / "02_coverage_mask.png"))

    # ── 5. Classical quick-fill for thin gaps, same as before ───────────
    hull_pts = np.column_stack(np.where(coverage > 0))
    gap_mask = None
    gap_px = 0
    if len(hull_pts) >= 3:
        hull_pts_xy = hull_pts[:, ::-1]
        hull = cv2.convexHull(hull_pts_xy)
        hull_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        gap_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(coverage))
        kernel = np.ones((3, 3), np.uint8)
        gap_mask = cv2.dilate(gap_mask, kernel, iterations=2)
        gap_px = int(np.sum(gap_mask > 0))
        save_image(np.stack([gap_mask] * 3, axis=-1),
                   str(comp_debug / "03_gap_mask.png"))

    # ── 6. Crop bbox ────────────────────────────────────────────────────
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
    crop_bbox = (int(rx), int(ry), int(rw), int(rh))
    print(f"  Crop: {rw} x {rh}  "
          f"(canvas {canvas_w} x {canvas_h}, gap pixels {gap_px:,})")

    meta = {
        "canvas_w": canvas_w, "canvas_h": canvas_h,
        "supersample": SS,
        "offset_x": round(float(ox), 1), "offset_y": round(float(oy), 1),
        "gap_pixels_detected": gap_px,
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
