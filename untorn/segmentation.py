"""
untorn.segmentation
===================
Phase 1: Fragment segmentation using SAM 2.1 Automatic Mask Generator.

Universal background-agnostic approach:
  1. Auto-detect background color by sampling image corners.
  2. Filter SAM masks by LAB-space color distance from background
     (not by area ratio) — works for any paper color on any background.
  3. Containment-aware hole filling for degraded paper with physical holes.
  4. Relative minimum area threshold (fraction of image) instead of
     absolute pixel count.
  5. Generous upper-area bound replaces the brittle BG_AREA_RATIO_THRESH.
"""

import json
import os
import numpy as np
import cv2
import torch
from pathlib import Path
from . import config as cfg
from .io_utils import save_image, save_mask


# ─────────────────────────────────────────────────────────────────────────────
# SAM loader (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def load_mask_generator():
    """Instantiate SAM 2.1 Automatic Mask Generator with our parameters."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    print("  Loading SAM 2.1 model ...")
    _device = os.environ.get("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam2(cfg.SAM2_CONFIG, cfg.SAM2_CHECKPOINT, device=_device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=cfg.SAM2_POINTS_PER_SIDE,
        pred_iou_thresh=cfg.SAM2_PRED_IOU_THRESH,
        stability_score_thresh=cfg.SAM2_STABILITY_THRESH,
        min_mask_region_area=cfg.SAM2_MIN_MASK_AREA,
        crop_n_layers=cfg.SAM2_CROP_N_LAYERS,
        crop_n_points_downscale_factor=max(cfg.SAM2_CROP_N_POINTS, 1),
    )
    print("  SAM 2.1 ready.")
    return mask_generator


# ─────────────────────────────────────────────────────────────────────────────
# Background auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def _sample_background_color(image_rgb: np.ndarray,
                              corner_frac: float = 0.05) -> np.ndarray:
    """
    Estimate background color by sampling the four image corners.

    Corners are almost always background in archive scan setups.
    Returns mean LAB color of sampled corner pixels as float32 array [L, a, b].
    """
    h, w = image_rgb.shape[:2]
    ch = max(1, int(h * corner_frac))
    cw = max(1, int(w * corner_frac))

    corners = [
        image_rgb[:ch,   :cw],    # top-left
        image_rgb[:ch,   -cw:],   # top-right
        image_rgb[-ch:,  :cw],    # bottom-left
        image_rgb[-ch:,  -cw:],   # bottom-right
    ]
    corner_pixels = np.concatenate(
        [c.reshape(-1, 3) for c in corners], axis=0
    ).astype(np.uint8)

    # Convert to LAB for perceptually uniform distance
    corner_lab = cv2.cvtColor(
        corner_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)

    # Use median (robust to scanner-edge artifacts)
    bg_lab = np.median(corner_lab, axis=0)
    return bg_lab


def _mask_mean_lab(mask_bool: np.ndarray,
                   image_lab: np.ndarray) -> np.ndarray:
    """Return mean LAB color of image pixels inside mask."""
    pixels = image_lab[mask_bool]
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    return pixels.mean(axis=0).astype(np.float32)


def _lab_distance(color_a: np.ndarray, color_b: np.ndarray) -> float:
    """Euclidean distance in LAB space (perceptual ΔE approx)."""
    return float(np.linalg.norm(color_a - color_b))


# ─────────────────────────────────────────────────────────────────────────────
# Overlap / containment merging
# ─────────────────────────────────────────────────────────────────────────────

def _merge_overlapping_masks(masks: list[dict],
                              iou_thresh: float = 0.4) -> list[dict]:
    """
    Merge masks that overlap significantly.
    SAM often produces multiple overlapping masks for the same fragment.
    """
    if not masks:
        return masks

    n = len(masks)
    merged = [False] * n
    result = []

    for i in range(n):
        if merged[i]:
            continue
        current_mask = masks[i]["segmentation"].copy()
        current_area = current_mask.sum()

        for j in range(i + 1, n):
            if merged[j]:
                continue
            other_mask = masks[j]["segmentation"]
            intersection = np.logical_and(current_mask, other_mask).sum()
            smaller_area = min(current_area, other_mask.sum())
            if smaller_area > 0 and intersection / smaller_area > iou_thresh:
                current_mask = np.logical_or(current_mask, other_mask)
                current_area = current_mask.sum()
                merged[j] = True

        masks[i]["segmentation"] = current_mask
        masks[i]["area"] = int(current_area)
        result.append(masks[i])

    return result


def _fill_contained_holes(paper_masks: list[dict],
                           bg_masks: list[dict],
                           h: int, w: int) -> list[dict]:
    """
    For degraded paper (Image 3): SAM generates masks for the background
    color showing through physical holes in the paper.  If a background-
    colored mask is spatially *contained* within a paper mask, fill it in
    (union) rather than treating it as a separate fragment.

    paper_masks : kept (paper-colored) masks — modified in-place
    bg_masks    : rejected (background-colored) masks
    Returns updated paper_masks.
    """
    for pm in paper_masks:
        pmask = pm["segmentation"]
        pm_area = pmask.sum()
        if pm_area == 0:
            continue

        for bm in bg_masks:
            bmask = bm["segmentation"]
            bm_area = bmask.sum()
            if bm_area == 0:
                continue

            # "Contained" = almost all of bg_mask pixels lie inside paper_mask
            intersection = np.logical_and(pmask, bmask).sum()
            containment_ratio = intersection / bm_area if bm_area > 0 else 0

            if containment_ratio > cfg.HOLE_CONTAINMENT_THRESH:
                # Fill the hole: union bg mask into paper mask
                pmask = np.logical_or(pmask, bmask)

        pm["segmentation"] = pmask
        pm["area"] = int(pmask.sum())

    return paper_masks


# ─────────────────────────────────────────────────────────────────────────────
# Morphological cleanup (unchanged logic, kernel sizes now DPI-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _morphological_cleanup(mask_uint8: np.ndarray,
                            image_area: int) -> np.ndarray:
    """
    Clean a fragment mask:
      - Close small holes (text punching through)
      - Remove small islands
      - Smooth boundary

    Kernel sizes scale with image resolution so the same physical operation
    applies regardless of DPI.
    """
    # Scale kernels relative to image diagonal — ~15px at 1 Mpx, larger for HQ scans
    diag = (image_area ** 0.5)
    close_r = max(5,  int(diag * 0.012))   # close: ~1.2% of diagonal
    open_r  = max(3,  int(diag * 0.005))   # open:  ~0.5% of diagonal

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_r * 2 + 1, close_r * 2 + 1))
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_r * 2 + 1,  open_r * 2 + 1))

    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(cleaned,    cv2.MORPH_OPEN,  kernel_open)
    # Note: previously we Gaussian-blurred and re-thresholded the mask here.
    # That blur softened the very edge curvature the matcher reads later
    # (and the boundary-refine step already snaps each pixel to the local
    # gradient ridge). Dropped in Step 1 cleanup.
    cleaned = (cleaned > 127).astype(np.uint8) * 255
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def segment_fragments(image_rgb: np.ndarray,
                      debug_dir: Path) -> list[dict]:
    """
    Segment paper fragments from background.

    Returns list of fragment dicts:
        {id, mask, bbox, area, contour, centroid}

    Saves extensive debug output to debug_dir/segmentation/
    """
    seg_debug = debug_dir / "segmentation"
    seg_debug.mkdir(parents=True, exist_ok=True)

    h, w = image_rgb.shape[:2]
    total_px = h * w
    image_area = total_px  # alias for clarity

    # Derived thresholds (relative to image size)
    min_fragment_area = int(total_px * cfg.MIN_FRAGMENT_AREA_FRAC)
    max_fragment_area = int(total_px * cfg.MAX_FRAGMENT_AREA_FRAC)

    # ── Pre-compute LAB image once ─────────────────────────────────────────
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # ── Auto-detect background color ──────────────────────────────────────
    bg_lab = _sample_background_color(image_rgb,
                                       corner_frac=cfg.BG_CORNER_SAMPLE_FRAC)
    print(f"  Background LAB: L={bg_lab[0]:.1f}  a={bg_lab[1]:.1f}"
          f"  b={bg_lab[2]:.1f}")

    # Save a corner-sample visualisation
    _save_corner_debug(image_rgb, bg_lab, seg_debug,
                       frac=cfg.BG_CORNER_SAMPLE_FRAC)

    # ── Run SAM 2.1 ────────────────────────────────────────────────────────
    print("  Running SAM 2.1 automatic mask generation ...")
    mask_gen = load_mask_generator()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw_masks = mask_gen.generate(image_rgb)
    print(f"  SAM 2.1 produced {len(raw_masks)} raw masks")

    # ── Score every mask by appearance ────────────────────────────────────
    scored = []
    for m in raw_masks:
        mask_bool = m["segmentation"]
        area = int(mask_bool.sum())
        mean_lab = _mask_mean_lab(mask_bool, image_lab)
        dist_from_bg = _lab_distance(mean_lab, bg_lab)
        scored.append({
            **m,
            "area": area,
            "_mean_lab": mean_lab,
            "_bg_dist": dist_from_bg,
        })

    # Save raw mask metadata (now includes bg_dist)
    raw_meta = []
    for i, m in enumerate(scored):
        raw_meta.append({
            "idx": i,
            "area": m["area"],
            "area_ratio": round(m["area"] / total_px, 4),
            "predicted_iou": round(float(m["predicted_iou"]), 4),
            "stability_score": round(float(m["stability_score"]), 4),
            "bbox": [int(x) for x in m["bbox"]],
            "bg_dist_lab": round(m["_bg_dist"], 2),
            "mean_lab": [round(float(v), 1) for v in m["_mean_lab"]],
        })
    with open(seg_debug / "raw_masks_meta.json", "w") as f:
        json.dump(raw_meta, f, indent=2)

    # Save raw SAM overlay
    _save_mask_overlay(image_rgb, scored,
                       str(seg_debug / "01_raw_sam_overlay.png"))

    # ── Partition: paper vs background ────────────────────────────────────
    paper_candidates = []
    bg_candidates    = []

    for m in scored:
        area = m["area"]

        # Absolute size guard: skip sub-pixel noise
        if area < cfg.SAM2_MIN_MASK_AREA:
            continue

        # Relative size upper bound: reject true full-image background mask
        if area > max_fragment_area:
            bg_candidates.append(m)
            print(f"    Reject mask area={area} "
                  f"(> max {max_fragment_area}, full-image bg)")
            continue

        # Appearance gate: is this mask paper-colored or background-colored?
        if m["_bg_dist"] >= cfg.BG_DIST_LAB_THRESH:
            paper_candidates.append(m)
        else:
            bg_candidates.append(m)
            print(f"    Reject mask area={area} "
                  f"bg_dist={m['_bg_dist']:.1f} < {cfg.BG_DIST_LAB_THRESH} "
                  f"(looks like background)")

    print(f"  Paper candidates: {len(paper_candidates)} | "
          f"Background candidates: {len(bg_candidates)}")

    # Save appearance-filtered overlay
    _save_mask_overlay(image_rgb, paper_candidates,
                       str(seg_debug / "02_after_appearance_filter.png"))

    # ── Relative area minimum ──────────────────────────────────────────────
    paper_candidates = [m for m in paper_candidates
                        if m["area"] >= min_fragment_area]
    print(f"  After relative-area minimum ({min_fragment_area}px): "
          f"{len(paper_candidates)} masks")

    # ── Hole filling: bg masks contained within paper masks ───────────────
    paper_candidates = _fill_contained_holes(
        paper_candidates, bg_candidates, h, w)
    print(f"  After hole filling: {len(paper_candidates)} masks")

    # ── Merge overlapping paper masks ─────────────────────────────────────
    paper_candidates.sort(key=lambda m: m["area"], reverse=True)
    paper_candidates = _merge_overlapping_masks(paper_candidates, iou_thresh=0.4)
    print(f"  After overlap merge: {len(paper_candidates)} masks")

    # Limit to reasonable count
    paper_candidates = paper_candidates[:cfg.MAX_FRAGMENTS]

    # ── Morphological cleanup & contour extraction ─────────────────────────
    colors = [(255, 0, 0), (0, 200, 0), (0, 80, 255), (255, 200, 0),
              (255, 0, 200), (0, 220, 220), (160, 0, 0), (0, 100, 0),
              (0, 0, 160), (140, 120, 0)]

    fragments = []
    for i, m in enumerate(paper_candidates):
        mask_raw = m["segmentation"].astype(np.uint8) * 255
        save_mask(mask_raw, str(seg_debug / f"03_mask_raw_{i:02d}.png"))

        mask_clean = _morphological_cleanup(mask_raw, image_area)
        save_mask(mask_clean, str(seg_debug / f"04_mask_clean_{i:02d}.png"))

        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"    Fragment {i}: no contour found, skipping")
            continue

        contour = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(contour)
        if area < min_fragment_area:
            continue

        # Recompute mask from contour (fills remaining interior holes)
        mask_final = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_final, [contour], -1, 255, cv2.FILLED)
        save_mask(mask_final, str(seg_debug / f"05_mask_final_{i:02d}.png"))

        x, y, bw, bh = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + bw / 2, y + bh / 2

        fragments.append({
            "id":       i,
            "mask":     mask_final,
            "bbox":     (x, y, bw, bh),
            "area":     int(area),
            "contour":  contour,
            "centroid": np.array([cx, cy]),
        })

    print(f"  Final fragment count: {len(fragments)}")

    # ── Final debug overlay ────────────────────────────────────────────────
    final_overlay = image_rgb.copy()
    frag_meta     = []

    for frag in fragments:
        c = colors[frag["id"] % len(colors)]
        cv2.drawContours(final_overlay, [frag["contour"]], -1, c, 3)
        cx, cy = frag["centroid"].astype(int)
        cv2.putText(final_overlay, str(frag["id"]),
                    (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)
        frag_meta.append({
            "id":        frag["id"],
            "area":      frag["area"],
            "bbox_xywh": list(frag["bbox"]),
            "centroid":  [round(float(frag["centroid"][0]), 1),
                          round(float(frag["centroid"][1]), 1)],
        })

    save_image(final_overlay,
               str(seg_debug / "06_final_fragments_overlay.png"))
    with open(seg_debug / "fragments_meta.json", "w") as f:
        json.dump(frag_meta, f, indent=2)

    # Individual fragment crops
    for frag in fragments:
        x, y, bw, bh = frag["bbox"]
        crop = image_rgb[y:y + bh, x:x + bw].copy()
        mask_crop = frag["mask"][y:y + bh, x:x + bw]
        crop[mask_crop == 0] = 0
        save_image(crop, str(seg_debug / f"07_crop_{frag['id']:02d}.png"))

    return fragments


# ─────────────────────────────────────────────────────────────────────────────
# Debug helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_mask_overlay(image_rgb: np.ndarray,
                       masks: list[dict],
                       path: str) -> None:
    colors = [(255, 0, 0), (0, 200, 0), (0, 80, 255), (255, 200, 0),
              (255, 0, 200), (0, 220, 220), (160, 0, 0), (0, 100, 0),
              (0, 0, 160), (140, 120, 0)]
    overlay = image_rgb.copy()
    for i, m in enumerate(masks):
        c = colors[i % len(colors)]
        mb = m["segmentation"]
        overlay[mb] = (
            overlay[mb].astype(np.float32) * 0.5 +
            np.array(c, dtype=np.float32) * 0.5
        ).astype(np.uint8)
    save_image(overlay, path)


def _save_corner_debug(image_rgb: np.ndarray,
                        bg_lab: np.ndarray,
                        seg_debug: Path,
                        frac: float = 0.05) -> None:
    """Draw corner sample regions and print detected bg color."""
    h, w = image_rgb.shape[:2]
    ch = max(1, int(h * frac))
    cw = max(1, int(w * frac))
    vis = image_rgb.copy()
    for (r0, r1, c0, c1) in [
        (0,   ch,  0,  cw),
        (0,   ch,  w - cw, w),
        (h - ch, h, 0,  cw),
        (h - ch, h, w - cw, w),
    ]:
        cv2.rectangle(vis, (c0, r0), (c1, r1), (0, 255, 0), 3)

    # Render detected bg color as a swatch
    swatch_h, swatch_w = 60, 200
    bg_bgr = cv2.cvtColor(
        np.array([[bg_lab]], dtype=np.float32), cv2.COLOR_LAB2BGR
    )
    bg_bgr_uint8 = np.clip(bg_bgr[0, 0], 0, 255).astype(np.uint8)

    swatch = np.full((swatch_h, swatch_w, 3),
                     bg_bgr_uint8.tolist(), dtype=np.uint8)
    cv2.putText(swatch, "BG color",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    vis[:swatch_h, :swatch_w] = cv2.cvtColor(swatch, cv2.COLOR_BGR2RGB)

    save_image(vis, str(seg_debug / "00_background_detection.png"))
