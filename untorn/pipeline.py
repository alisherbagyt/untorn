"""
untorn.pipeline
===============
Orchestrates the full reconstruction pipeline:
  Phase 0: Preprocessing — downscale large images to fit in GPU VRAM
  Phase 1: Segmentation (SAM 2.1) — on working-resolution image
  Phase 2: Contour analysis — on working-resolution image
  Phase 3: Layout-agnostic MST global assembly
            (candidate enumeration → four-gate matching →
             seed + MST growth → bundle adjust → orphan rescue)
  Phase 4: Composition — upscale transforms, compose at FULL resolution
  Phase 5: Gap fill — classify holes, build repair mask, LaMa cleanup
"""

import os
import time
import json
import numpy as np
from pathlib import Path

from . import config as cfg
from . import edge_matcher
from .io_utils import load_image, save_image
from .preprocess import prepare_image, upscale_transforms, upscale_fragments
from .segmentation import segment_fragments
from .contours import analyze_fragments
from .assembly import reconstruct
from .composition import compose_final
from .gap_fill import inpaint_gaps
from .inpainting import is_available as lama_available


def run(input_path: str, output_path: str = None):
    """
    Full reconstruction pipeline.

    Args:
        input_path: Path to input image (.tif/.tiff/.jpg/.png)
        output_path: Path for output. Defaults to data/output/<stem>_reconstructed.png
    """
    cfg.ensure_dirs()

    input_path = Path(input_path)
    stem = input_path.stem

    if output_path is None:
        output_path = cfg.OUTPUT_DIR / f"{stem}_reconstructed.png"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = cfg.OUTPUT_DIR / output_path

    # Per-image debug directory
    debug_dir = cfg.DEBUG_DIR / stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    pipeline_meta = {
        "input": str(input_path),
        "output": str(output_path),
        "debug_dir": str(debug_dir),
    }

    print("=" * 65)
    print(f"  UNTORN - Torn Paper Reconstruction")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Debug:  {debug_dir}")
    print("=" * 65)

    # ── Load image ─────────────────────────────────────────────────────────
    print("\n[LOAD] Loading image ...")
    image_rgb = load_image(str(input_path))
    h, w = image_rgb.shape[:2]
    print(f"  Size: {w} x {h} ({w*h:,} pixels)")
    save_image(image_rgb, str(debug_dir / "00_input.png"))
    pipeline_meta["image_size"] = {"w": w, "h": h}

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 0: Preprocessing — downscale if needed
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 65)
    print("[PHASE 0] Preprocessing")
    print("-" * 65)
    t0 = time.time()

    prep = prepare_image(image_rgb, debug_dir)
    work_rgb = prep["work_rgb"]
    full_rgb = prep["full_rgb"]
    scale_factor = prep["scale_factor"]

    t1 = time.time()
    timings["preprocess"] = round(t1 - t0, 2)
    pipeline_meta["scale_factor"] = round(scale_factor, 4)
    pipeline_meta["working_size"] = {"w": prep["work_w"], "h": prep["work_h"]}
    print(f"  Phase 0 complete in {timings['preprocess']:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Segmentation (on working-resolution image)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 65)
    print("[PHASE 1] Segmenting fragments with SAM 2.1")
    print("-" * 65)
    t0 = time.time()

    fragments = segment_fragments(work_rgb, debug_dir)

    t1 = time.time()
    timings["segmentation"] = round(t1 - t0, 2)
    print(f"  Phase 1 complete: {len(fragments)} fragments in {timings['segmentation']:.1f}s")

    if len(fragments) < 2:
        print("\n  FAILED: Need at least 2 fragments. Check debug/segmentation/")
        print("    Possible causes:")
        print("    - Image has no clear dark background")
        print("    - Fragments too small or merged by SAM")
        pipeline_meta["status"] = "FAILED_SEGMENTATION"
        with open(debug_dir / "pipeline_meta.json", "w") as f:
            json.dump(pipeline_meta, f, indent=2)
        return None

    pipeline_meta["n_fragments"] = len(fragments)

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: Contour Analysis (working resolution)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 65)
    print("[PHASE 2] Analyzing contours and edge features")
    print("-" * 65)
    t0 = time.time()

    fragments = analyze_fragments(fragments, work_rgb, debug_dir)

    t1 = time.time()
    timings["contour_analysis"] = round(t1 - t0, 2)
    print(f"  Phase 2 complete in {timings['contour_analysis']:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 3: Hierarchical proximity-based reconstruction
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 65)
    print("[PHASE 3] Layout-agnostic MST global assembly")
    print("-" * 65)
    t0 = time.time()

    # Load the Siamese edge matcher (Phase 4 gate). Failure is non-fatal —
    # `edge_matcher.load()` logs and returns None when torch / the
    # checkpoint is missing, and `_match_edge_pair` checks `is_loaded()`
    # before calling so the pipeline degrades to the legacy 4-gate cascade.
    if getattr(cfg, "EDGE_MATCHER_ENABLED", False):
        edge_matcher.load()
    pipeline_meta["edge_matcher_loaded"] = bool(edge_matcher.is_loaded())

    try:
        transforms = reconstruct(fragments, work_rgb, debug_dir)
    finally:
        # Free GPU memory before LaMa (Phase 5) loads its own weights.
        edge_matcher.unload()

    t1 = time.time()
    timings["reconstruction"] = round(t1 - t0, 2)
    print(f"  Phase 3 complete in {timings['reconstruction']:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 4: Composition at FULL resolution
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 65)
    print("[PHASE 4] Composing final image at full resolution")
    print("-" * 65)
    t0 = time.time()

    if scale_factor > 1.0:
        print(f"  Upscaling transforms by {scale_factor:.2f}x ...")
        transforms = upscale_transforms(transforms, scale_factor)
        print(f"  Upscaling fragment masks to full resolution ...")
        fragments = upscale_fragments(fragments, full_rgb, scale_factor)
        compose_image = full_rgb
    else:
        compose_image = work_rgb

    comp = compose_final(compose_image, fragments, transforms, debug_dir)
    canvas_uncropped = comp["canvas"]
    coverage         = comp["coverage"]
    gap_mask         = comp["gap_mask"]
    crop_bbox        = comp["crop_bbox"]

    t1 = time.time()
    timings["composition"] = round(t1 - t0, 2)
    print(f"  Phase 4 complete in {timings['composition']:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 5: LaMa inpainting — seam/scar cleaning
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 65)
    print("[PHASE 5] Gap fill: classify holes, build repair mask, LaMa")
    print("-" * 65)
    t0 = time.time()

    refine = bool(int(os.environ.get("UNTORN_INPAINT_REFINE", "0")))
    if not lama_available():
        print("  ! LaMa checkpoint missing - Phase 5 will pass image through unchanged.")

    gap_result = inpaint_gaps(canvas_uncropped, coverage,
                              debug_dir, refine=refine)
    cleaned_uncropped = gap_result["canvas"]
    pipeline_meta["missing_fragment"] = bool(
        gap_result["meta"].get("missing_fragment", False))
    pipeline_meta["hole_counts"] = gap_result["meta"].get("hole_counts", {})

    # Apply the crop bbox that composition deferred to us.
    rx, ry, rw, rh = crop_bbox
    result = cleaned_uncropped[ry:ry+rh, rx:rx+rw]
    print(f"  Final image: {result.shape[1]} x {result.shape[0]}")

    t1 = time.time()
    timings["inpainting"] = round(t1 - t0, 2)
    print(f"  Phase 5 complete in {timings['inpainting']:.1f}s")

    # ── Save output ────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, str(output_path))
    print(f"\n{'=' * 65}")
    print(f"  DONE - Saved to {output_path}")

    total_time = sum(timings.values())
    timings["total"] = round(total_time, 2)
    pipeline_meta["timings"] = timings
    pipeline_meta["status"] = "SUCCESS"

    with open(debug_dir / "pipeline_meta.json", "w") as f:
        json.dump(pipeline_meta, f, indent=2)

    print(f"  Total time: {total_time:.1f}s")
    print(f"  Debug output: {debug_dir}")
    print(f"{'=' * 65}")

    return result
