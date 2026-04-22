"""
untorn.preprocess
=================
Phase 0: Prepare the input image for the pipeline.

Downscales large images to a working resolution that fits in GPU VRAM,
while preserving the full-resolution original for final composition.

The scale factor is tracked so that all translations computed on the
downscaled image can be mapped back to full-resolution coordinates.
"""

import numpy as np
import cv2
from pathlib import Path

from .io_utils import save_image


# Maximum dimension (longest side) for the working image.
# 1500px keeps SAM 2.1 comfortable on 4GB VRAM with points_per_side=32.
WORKING_MAX_DIM = 1500


def prepare_image(image_rgb: np.ndarray, debug_dir: Path,
                  max_dim: int = WORKING_MAX_DIM) -> dict:
    """
    Prepare working and full-resolution images.

    If the image's longest side exceeds max_dim, create a downscaled
    working copy.  Otherwise, use the original as-is (scale_factor=1.0).

    Returns:
        dict with keys:
            - full_rgb:      original full-resolution image (np.ndarray)
            - work_rgb:      working-resolution image (np.ndarray)
            - scale_factor:  full / work ratio (float, >= 1.0)
            - full_h, full_w, work_h, work_w: dimensions
    """
    prep_debug = debug_dir / "preprocess"
    prep_debug.mkdir(parents=True, exist_ok=True)

    full_h, full_w = image_rgb.shape[:2]
    longest = max(full_h, full_w)

    if longest <= max_dim:
        # No downscaling needed
        scale_factor = 1.0
        work_rgb = image_rgb
        print(f"  Image {full_w}x{full_h} fits within {max_dim}px — no downscaling")
    else:
        scale_factor = longest / max_dim
        work_w = int(round(full_w / scale_factor))
        work_h = int(round(full_h / scale_factor))
        work_rgb = cv2.resize(image_rgb, (work_w, work_h),
                              interpolation=cv2.INTER_AREA)
        print(f"  Downscaled {full_w}x{full_h} -> {work_w}x{work_h} "
              f"(scale factor {scale_factor:.2f}x)")

        save_image(work_rgb, str(prep_debug / "working_image.png"))

    work_h, work_w = work_rgb.shape[:2]

    result = {
        "full_rgb":     image_rgb,
        "work_rgb":     work_rgb,
        "scale_factor": scale_factor,
        "full_h": full_h,
        "full_w": full_w,
        "work_h": work_h,
        "work_w": work_w,
    }

    # Save metadata
    import json
    meta = {
        "full_size": f"{full_w}x{full_h}",
        "work_size": f"{work_w}x{work_h}",
        "scale_factor": round(scale_factor, 4),
        "downscaled": scale_factor > 1.0,
    }
    with open(prep_debug / "preprocess_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return result


def upscale_transforms(transforms: dict, scale_factor: float) -> dict:
    """
    Scale affine transforms from working resolution back to full resolution.

    Each transform is a 3x3 homogeneous affine matrix.  Rotation stays the
    same; translation components are multiplied by scale_factor.

    Args:
        transforms:   dict mapping fragment_index → 3x3 np.ndarray
        scale_factor: full_res / work_res ratio (>= 1.0)

    Returns:
        dict with same keys, translation components scaled.
    """
    if scale_factor <= 1.0:
        return transforms

    scaled = {}
    for k, M in transforms.items():
        M_new = M.copy()
        M_new[0, 2] *= scale_factor
        M_new[1, 2] *= scale_factor
        scaled[k] = M_new

    return scaled


def upscale_fragments(fragments: list[dict], full_rgb: np.ndarray,
                      scale_factor: float) -> list[dict]:
    """
    Re-derive fragment masks at full resolution by upscaling the working-res
    masks.  Contours, bboxes, centroids, and support points are all recomputed
    from the upscaled mask so they're pixel-accurate at full res.
    """
    if scale_factor <= 1.0:
        return fragments

    full_h, full_w = full_rgb.shape[:2]
    sf = scale_factor

    for frag in fragments:
        # Upscale mask with nearest-neighbor (keeps binary)
        mask_full = cv2.resize(frag["mask"], (full_w, full_h),
                               interpolation=cv2.INTER_NEAREST)
        frag["mask"] = mask_full

        # Recompute contour from full-res mask
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            frag["contour"] = max(contours, key=cv2.contourArea)
        frag["area"] = int(cv2.contourArea(frag["contour"]))

        x, y, bw, bh = cv2.boundingRect(frag["contour"])
        frag["bbox"] = (x, y, bw, bh)

        M = cv2.moments(frag["contour"])
        if M["m00"] > 0:
            frag["centroid"] = np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]])
        else:
            frag["centroid"] = np.array([x + bw/2, y + bh/2])

    return fragments
