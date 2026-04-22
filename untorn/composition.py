"""
untorn.composition
==================
Phase 4: Compose the final reconstructed image from fragment transforms.

Supports full affine transforms (rotation + translation) per fragment.
"""

import json
import numpy as np
import cv2
from pathlib import Path

from . import config as cfg
from .io_utils import save_image


def compose_final(image_rgb: np.ndarray, fragments: list[dict],
                  transforms: dict, debug_dir: Path) -> dict:
    """
    Place all fragments onto a single canvas using their masks and affine
    transforms.  Fill thin inter-fragment gaps with classical inpaint.

    Args:
        image_rgb:  HxWx3 source image.
        fragments:  list of fragment dicts (with mask, bbox, contour, centroid).
        transforms: dict mapping fragment_index → 3x3 homogeneous affine matrix.
        debug_dir:  per-job debug directory.

    Returns a dict with:
        canvas:    HxWx3 uint8, uncropped RGB composite (with classical gap fill).
        coverage:  HxW uint8, non-zero where any fragment is placed.
        gap_mask:  HxW uint8, gap pixels fed into the classical inpaint (may be None).
        crop_bbox: (x, y, w, h) tight crop of content on the canvas.
    """
    comp_debug = debug_dir / "composition"
    comp_debug.mkdir(parents=True, exist_ok=True)

    h, w = image_rgb.shape[:2]
    n = len(fragments)

    # ── Compute canvas bounds from transformed bboxes ─────────────────
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for i in range(n):
        M = transforms[i]
        bx, by, bw, bh = fragments[i]["bbox"]
        corners = np.array([
            [bx,    by,    1],
            [bx+bw, by,    1],
            [bx+bw, by+bh, 1],
            [bx,    by+bh, 1],
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

    print(f"  Canvas size: {canvas_w} x {canvas_h}")

    # ── Place fragments via warpAffine ────────────────────────────────
    canvas   = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    coverage = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # Sort by area descending so smaller pieces overlay on top
    order = sorted(range(n), key=lambda i: fragments[i]["area"], reverse=True)

    for i in order:
        frag = fragments[i]
        M = transforms[i].copy()
        # Add canvas offset
        M[0, 2] += ox
        M[1, 2] += oy

        # Use bbox sub-region for efficiency
        bx, by, bw, bh = frag["bbox"]
        sub_mask = frag["mask"][by:by+bh, bx:bx+bw]
        sub_img = np.zeros((bh, bw, 3), dtype=np.uint8)
        m = sub_mask > 127
        if not np.any(m):
            continue
        sub_img[m] = image_rgb[by:by+bh, bx:bx+bw][m]

        # Adjust affine for sub-image origin at (bx, by)
        origin = M @ np.array([bx, by, 1.0], dtype=np.float64)
        M_sub = M.copy()
        M_sub[0, 2] = origin[0]
        M_sub[1, 2] = origin[1]
        M_2x3 = M_sub[:2, :].astype(np.float32)

        warped_img  = cv2.warpAffine(sub_img,  M_2x3, (canvas_w, canvas_h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
        warped_mask = cv2.warpAffine(sub_mask, M_2x3, (canvas_w, canvas_h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

        mw = warped_mask > 127
        canvas[mw]   = warped_img[mw]
        coverage[mw] = 255

    save_image(canvas, str(comp_debug / "01_raw_composite.png"))
    save_image(np.stack([coverage]*3, axis=-1),
               str(comp_debug / "02_coverage_mask.png"))

    # ── Gap inpainting ────────────────────────────────────────────────
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
        print(f"  Gap pixels to inpaint: {gap_px}")

        save_image(np.stack([gap_mask]*3, axis=-1),
                   str(comp_debug / "03_gap_mask.png"))

        if gap_px > 0:
            canvas = cv2.inpaint(canvas, gap_mask, cfg.INPAINT_RADIUS,
                                 cv2.INPAINT_TELEA)

    save_image(canvas, str(comp_debug / "04_inpainted.png"))

    # ── Compute crop bbox ─────────────────────────────────────────────
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
    print(f"  Crop bbox: {rw} x {rh}  (canvas was {canvas_w} x {canvas_h})")

    # Save composition metadata
    meta = {
        "canvas_w": canvas_w, "canvas_h": canvas_h,
        "offset_x": round(float(ox), 1), "offset_y": round(float(oy), 1),
        "gap_pixels_inpainted": gap_px,
        "crop_bbox": list(crop_bbox),
        "final_w": rw, "final_h": rh,
    }
    with open(comp_debug / "composition_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "canvas":    canvas,
        "coverage":  coverage,
        "gap_mask":  gap_mask,
        "crop_bbox": crop_bbox,
    }
