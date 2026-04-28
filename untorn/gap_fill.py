"""
untorn.gap_fill
===============
Phase 5 front-end — classify holes in the composed canvas, build a
text-aware repair mask, and dispatch to the LaMa backend.

Post-Step-6 the composition step does the work that used to fall on this
module for "small" holes: any uncovered pixel within
``COMP_SEAM_FILL_MAX_PX`` of a placed fragment is Voronoi-filled from
the nearest covered pixel, so by the time gap_fill runs every remaining
hole is genuinely far from any fragment. Hole classification therefore
collapses to two cases:
    edge hole   — touches the canvas border; just cropped out by composition
    interior hole — handed to LaMa for hallucination, with `medium` /
                    `large` distinction kept only for context-expansion
                    sizing and missing-fragment reporting.

Public entry:
    inpaint_gaps(canvas, coverage, debug_dir, *, refine=False)
       -> {"canvas": cleaned, "meta": {...}}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config as cfg
from .io_utils import save_image, save_mask
from . import inpainting


# ══════════════════════════════════════════════════════════════════════════
#  Hole detection and classification
# ══════════════════════════════════════════════════════════════════════════

def _document_hull_mask(coverage: np.ndarray) -> np.ndarray:
    """Convex hull of coverage as a uint8 mask — proxy for 'document area'."""
    pts = np.column_stack(np.where(coverage > 0))
    if len(pts) < 3:
        return np.zeros_like(coverage, dtype=np.uint8)
    hull = cv2.convexHull(pts[:, ::-1])
    m = np.zeros(coverage.shape, dtype=np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    return m


def _classify_holes(coverage: np.ndarray,
                    edge_touch_px: int,
                    small_frac: float,
                    medium_frac: float
                    ) -> tuple[np.ndarray, list[dict]]:
    """
    Find each not-covered connected component inside the document hull,
    label it as edge / small / medium / large, and return:
      * interior_mask : uint8 mask of all HOLE components that should be
                        repaired (edge holes excluded).
      * reports       : per-component dicts with area_frac, kind, bbox.
    """
    h, w = coverage.shape[:2]
    hull = _document_hull_mask(coverage)
    doc_area = max(int((hull > 0).sum()), 1)
    holes_all = cv2.bitwise_and(hull, cv2.bitwise_not(coverage))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes_all,
                                                           connectivity=8)

    interior = np.zeros_like(coverage, dtype=np.uint8)
    reports: list[dict] = []
    for cid in range(1, n):
        x, y, bw, bh, area = stats[cid]
        comp = (labels == cid).astype(np.uint8) * 255
        touches_edge = (
            x <= edge_touch_px
            or y <= edge_touch_px
            or (x + bw) >= (w - edge_touch_px)
            or (y + bh) >= (h - edge_touch_px)
        )
        area_frac = float(area) / float(doc_area)
        kind = "edge" if touches_edge else (
            "small" if area_frac < small_frac
            else "medium" if area_frac < medium_frac
            else "large")
        if kind != "edge":
            interior = cv2.bitwise_or(interior, comp)
        reports.append({
            "id":        int(cid),
            "bbox":      [int(x), int(y), int(bw), int(bh)],
            "area_px":   int(area),
            "area_frac": round(area_frac, 5),
            "kind":      kind,
        })
    return interior, reports


# ══════════════════════════════════════════════════════════════════════════
#  Repair-mask synthesis — seam ring + classified holes
# ══════════════════════════════════════════════════════════════════════════

def _build_repair_mask(canvas_rgb: np.ndarray,
                       coverage: np.ndarray,
                       holes_interior: np.ndarray,
                       reports: list[dict],
                       band_px: int,
                       ink_thresh: int,
                       medium_context_px: int
                       ) -> np.ndarray:
    """
    Build a text-aware repair mask covering both:
      (a) the symmetric ring around fragment coverage (seam scar), and
      (b) every interior hole, with medium/large holes dilated by
          `medium_context_px` so LaMa has surrounding context.
    Dark (ink) pixels are subtracted so LaMa doesn't repaint text.
    """
    # Seam ring — delegate to the existing text-aware builder.
    ring = inpainting.build_scar_mask(
        canvas_rgb, coverage, gap_mask=None,
        band_px=band_px, ink_threshold=ink_thresh,
    )

    # Build a per-hole mask: small holes as-is; medium/large dilated for
    # context. We don't dilate edge holes (they were dropped).
    holes_contextual = holes_interior.copy()
    for r in reports:
        if r["kind"] in ("medium", "large"):
            comp_mask = np.zeros_like(coverage, dtype=np.uint8)
            x, y, bw, bh = r["bbox"]
            # We can't extract the exact component again without the
            # labels; fall back to drawing its bbox + dilating the union
            # mask. The union dilation is effectively the same.
            comp_mask[y:y + bh, x:x + bw] = holes_interior[y:y + bh, x:x + bw]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            expanded = cv2.dilate(comp_mask, kernel,
                                  iterations=medium_context_px)
            holes_contextual = cv2.bitwise_or(holes_contextual, expanded)

    repair = cv2.bitwise_or(ring, holes_contextual)

    # Subtract ink pixels so we don't repaint real text.
    gray = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2GRAY)
    ink = (gray < ink_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ink = cv2.dilate(ink, kernel, iterations=1)
    repair = cv2.bitwise_and(repair, cv2.bitwise_not(ink))
    repair = cv2.morphologyEx(repair, cv2.MORPH_CLOSE, kernel)

    # Drop speckle
    if getattr(inpainting, "INPAINT_MIN_MASK_PX", 8) > 0:
        thresh = inpainting.INPAINT_MIN_MASK_PX
        n, labels, stats, _ = cv2.connectedComponentsWithStats(repair)
        keep = np.zeros_like(repair)
        for cid in range(1, n):
            if stats[cid, cv2.CC_STAT_AREA] >= thresh:
                keep[labels == cid] = 255
        repair = keep

    return repair


# ══════════════════════════════════════════════════════════════════════════
#  Public entry — Phase 5 orchestrator
# ══════════════════════════════════════════════════════════════════════════

def inpaint_gaps(canvas_rgb: np.ndarray,
                 coverage: np.ndarray,
                 debug_dir: Path,
                 *,
                 refine: bool = False) -> dict:
    """
    Phase 5 orchestrator: classify holes, build a text-aware repair
    mask, run LaMa.

    Returns a dict:
        canvas       HxWx3 uint8 cleaned RGB (same size as input)
        meta         the aggregated metadata written to
                     <debug_dir>/inpainting/inpainting_meta.json
    """
    out_dir = debug_dir / "inpainting"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(canvas_rgb, str(out_dir / "01_before.png"))

    # ── Classify holes ─────────────────────────────────────────────────
    edge_touch_px   = int(getattr(cfg, "GAP_EDGE_TOUCH_PX", 4))
    small_frac      = float(getattr(cfg, "GAP_SMALL_FRAC", 0.005))
    medium_frac     = float(getattr(cfg, "GAP_MEDIUM_FRAC", 0.05))
    context_expand  = int(getattr(cfg, "GAP_LARGE_CONTEXT_EXPAND_PX", 20))
    ink_thresh      = int(getattr(inpainting, "INPAINT_INK_THRESH", 140))
    band_px         = int(getattr(inpainting, "INPAINT_BAND_PX", 4))

    holes_interior, reports = _classify_holes(
        coverage,
        edge_touch_px=edge_touch_px,
        small_frac=small_frac,
        medium_frac=medium_frac,
    )
    save_mask(holes_interior, str(out_dir / "02_holes_interior.png"))

    counts = {k: 0 for k in ("edge", "small", "medium", "large")}
    largest_frac = 0.0
    for r in reports:
        counts[r["kind"]] += 1
        if r["kind"] != "edge":
            largest_frac = max(largest_frac, r["area_frac"])

    missing_fragment = any(r["kind"] == "large" for r in reports)

    # ── Build repair mask and dispatch ─────────────────────────────────
    repair = _build_repair_mask(
        canvas_rgb, coverage, holes_interior, reports,
        band_px=band_px, ink_thresh=ink_thresh,
        medium_context_px=context_expand,
    )
    save_mask(repair, str(out_dir / "03_repair_mask.png"))
    mask_px = int((repair > 0).sum())
    print(f"  Holes: edge={counts['edge']} small={counts['small']} "
          f"medium={counts['medium']} large={counts['large']}  "
          f"(largest={largest_frac*100:.2f}% of doc)")
    print(f"  Repair mask pixels: {mask_px:,}")

    # ── Dispatch ────────────────────────────────────────────────────────
    if not inpainting.is_available():
        print("  ! LaMa missing — returning canvas unchanged.")
        cleaned = canvas_rgb
        status = "SKIPPED_NO_MODEL"
        err = None
        backend = None
        duration_s = 0.0
    elif mask_px == 0:
        print("  -- No repair pixels; returning canvas as-is.")
        cleaned = canvas_rgb
        status = "OK_NO_OP"
        err = None
        backend = None
        duration_s = 0.0
    else:
        t0 = time.time()
        try:
            cleaned = inpainting.inpaint(canvas_rgb, repair,
                                          tile=True, refine=refine)
            status = "OK"
            err = None
        except Exception as exc:
            print(f"  ! LaMa failed: {exc}")
            cleaned = canvas_rgb
            status = "FAILED"
            err = str(exc)
        duration_s = round(time.time() - t0, 2)
        backend = (inpainting._predictor.kind
                   if inpainting._predictor is not None else None)

    save_image(cleaned, str(out_dir / "04_cleaned.png"))

    import torch
    meta = {
        "status":              status,
        "error":               err,
        "backend":             backend,
        "mask_pixels":         mask_px,
        "duration_s":          duration_s,
        "device":              "cuda" if torch.cuda.is_available() else "cpu",
        "refine":              bool(refine),
        "band_px":             band_px,
        "ink_threshold":       ink_thresh,
        "hole_counts":         counts,
        "largest_hole_frac":   round(largest_frac, 5),
        "missing_fragment":    bool(missing_fragment),
        "hole_reports":        reports,
    }
    with open(out_dir / "inpainting_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"canvas": cleaned, "meta": meta}


__all__ = ["inpaint_gaps"]
