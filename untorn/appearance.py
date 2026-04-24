"""
untorn.appearance
=================
DINOv2 dense-feature appearance gate.

Every fragment gets a (H_p, W_p, D) patch-token map over its bbox crop
(cached on `frag["dinov2"]`). For every proposed seam between two
fragments, we sample a few patches on each side of the seam in canvas
space, look up feature vectors in both fragments' crops via an
image-xy -> feature-grid affine, and cosine-average across matched pairs.

The model (ViT-S/14, ~22 M params) is frozen, loaded lazily once, and
reused across the pipeline.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

from . import config as cfg


# ---------------------------------------------------------------------------
# Extractor (lazy singleton)
# ---------------------------------------------------------------------------

_EXTRACTOR: "Optional[DINOv2Extractor]" = None


class DINOv2Extractor:
    """Thin wrapper around torch.hub's DINOv2 ViT-S/14 for dense features."""

    def __init__(self,
                 model_name: str = cfg.DINOV2_MODEL,
                 device: str = cfg.DINOV2_DEVICE,
                 input_size: int = cfg.DINOV2_INPUT_SIZE,
                 patch_size: int = cfg.DINOV2_PATCH_SIZE):
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.grid = self.input_size // self.patch_size  # 16 for 224/14

        # ImageNet normalisation that DINOv2 expects.
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = torch.hub.load(
                "facebookresearch/dinov2", model_name,
                source="github", verbose=False)
        self.model.eval().to(device)
        self._torch = torch
        # feature dim = embed_dim; available on the model.
        self.feat_dim = int(getattr(self.model, "embed_dim", 384))

    @staticmethod
    def get() -> "DINOv2Extractor":
        global _EXTRACTOR
        if _EXTRACTOR is None:
            _EXTRACTOR = DINOv2Extractor()
        return _EXTRACTOR

    @staticmethod
    def release() -> None:
        global _EXTRACTOR
        if _EXTRACTOR is None:
            return
        try:
            import torch
            del _EXTRACTOR.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _EXTRACTOR = None

    # -- Core: image crop -> dense feature grid ---------------------------

    def _preprocess(self, crop_rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """
        Resize `crop_rgb` (uint8 HxWx3) so its longer side is `input_size`,
        then centre-pad to input_size x input_size with mid-gray. Returns
        (chw_float_normalised, scale, pad_x, pad_y).
        """
        import cv2
        h, w = crop_rgb.shape[:2]
        scale = float(self.input_size) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(crop_rgb, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)

        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        # mid-gray pad so the model doesn't latch onto the border
        canvas = np.full((self.input_size, self.input_size, 3),
                         128, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        x = canvas.astype(np.float32) / 255.0
        x = (x - self._mean) / self._std
        x = np.transpose(x, (2, 0, 1))  # CHW
        return x, scale, pad_x, pad_y

    def extract(self, crop_rgb: np.ndarray) -> dict:
        """
        Dense patch-token features for one fragment crop.

        Returns a dict:
            features  : (grid, grid, D) float32, L2-normalised along D
            scale     : float, crop->input_size scale applied
            pad_x,y   : int, centre-padding offsets inside input_size canvas
            patch_size, grid: ints (for convenience at call sites)
        """
        chw, scale, pad_x, pad_y = self._preprocess(crop_rgb)
        t = self._torch.from_numpy(chw).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            out = self.model.forward_features(t)
        tokens = out["x_norm_patchtokens"]   # (1, grid*grid, D)
        tokens = tokens[0].detach().cpu().numpy().astype(np.float32)
        feats = tokens.reshape(self.grid, self.grid, -1)
        # L2-normalise along channel so cosine == dot.
        norm = np.linalg.norm(feats, axis=-1, keepdims=True)
        feats = feats / np.clip(norm, 1e-6, None)
        return {
            "features":  feats,
            "scale":     float(scale),
            "pad_x":     int(pad_x),
            "pad_y":     int(pad_y),
            "patch_size": int(self.patch_size),
            "grid":      int(self.grid),
        }


# ---------------------------------------------------------------------------
# Per-fragment attachment
# ---------------------------------------------------------------------------

def _fragment_crop(image_rgb: np.ndarray,
                   frag: dict,
                   pad_px: int = 8) -> tuple[np.ndarray, int, int]:
    """Tight-ish bbox crop with a few pixels of padding, clipped to image."""
    x, y, w, h = frag["bbox"]
    H, W = image_rgb.shape[:2]
    x0 = max(0, x - pad_px)
    y0 = max(0, y - pad_px)
    x1 = min(W, x + w + pad_px)
    y1 = min(H, y + h + pad_px)
    crop = image_rgb[y0:y1, x0:x1]
    return crop, x0, y0


def attach_dinov2_features_all(fragments: list[dict],
                               image_rgb: np.ndarray,
                               extractor: Optional[DINOv2Extractor] = None
                               ) -> list[dict]:
    """Compute and cache DINOv2 dense features on every fragment."""
    if not cfg.DINOV2_ENABLED or len(fragments) == 0:
        for frag in fragments:
            frag["dinov2"] = None
        return fragments
    if extractor is None:
        extractor = DINOv2Extractor.get()
    for frag in fragments:
        crop, x0, y0 = _fragment_crop(image_rgb, frag)
        info = extractor.extract(crop)
        info["crop_origin"] = (int(x0), int(y0))
        # Original crop size (pre-resize, post-pad-to-image-boundary).
        info["crop_hw"] = (int(crop.shape[0]), int(crop.shape[1]))
        frag["dinov2"] = info
    return fragments


# ---------------------------------------------------------------------------
# Image(x,y) -> feature-grid lookup
# ---------------------------------------------------------------------------

def _image_xy_to_feat_uv(info: dict, xy: np.ndarray) -> np.ndarray:
    """
    Map image-coord points (xy in working-image frame) into the fragment's
    feature-grid coordinates (fu, fv), continuous, bounded by grid-1.
    Returns (N, 2) float. Points falling outside the usable feature grid
    are returned with NaNs so callers can reject them.
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    x0, y0 = info["crop_origin"]
    s = info["scale"]
    px = info["patch_size"]
    pad_x = info["pad_x"]; pad_y = info["pad_y"]
    grid = info["grid"]

    # image -> crop -> input_size canvas
    cx = (xy[:, 0] - x0) * s + pad_x
    cy = (xy[:, 1] - y0) * s + pad_y
    # canvas px -> patch coords, then centre of patch is at i*px + px/2, so
    # fu = cx/px - 0.5 (so the feature at integer (u, v) lives at patch
    # centre). Clamp below to 0 so nearest-patch sampling still works.
    fu = cx / px - 0.5
    fv = cy / px - 0.5

    bad = (cx < 0) | (cy < 0) | (cx > grid * px) | (cy > grid * px)
    fu[bad] = np.nan
    fv[bad] = np.nan
    return np.stack([fu, fv], axis=-1)


def _sample_feature(info: dict, fuv: np.ndarray) -> np.ndarray:
    """Bilinear sample of the feature grid at a single (fu, fv). Returns
    a (D,) vector or None if out of grid."""
    fu, fv = fuv
    if not np.isfinite(fu) or not np.isfinite(fv):
        return None
    feats = info["features"]
    grid = info["grid"]
    fu = float(np.clip(fu, 0.0, grid - 1.0001))
    fv = float(np.clip(fv, 0.0, grid - 1.0001))
    u0 = int(np.floor(fu)); v0 = int(np.floor(fv))
    du = fu - u0;            dv = fv - v0
    f00 = feats[v0,     u0    ]
    f01 = feats[v0,     u0 + 1]
    f10 = feats[v0 + 1, u0    ]
    f11 = feats[v0 + 1, u0 + 1]
    f = ((1 - du) * (1 - dv) * f00 +
         du       * (1 - dv) * f01 +
         (1 - du) * dv       * f10 +
         du       * dv       * f11)
    n = np.linalg.norm(f)
    if n < 1e-6:
        return None
    return f / n


# ---------------------------------------------------------------------------
# Seam patch cosine
# ---------------------------------------------------------------------------

def _apply_affine(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    ones = np.ones((xy.shape[0], 1))
    return (np.concatenate([xy, ones], axis=1) @ M.T)[:, :2]


def seam_patch_cosine(frag_a: dict,
                      frag_b: dict,
                      M_a: np.ndarray,
                      M_b: np.ndarray,
                      seam_point: np.ndarray,
                      seam_normal: np.ndarray,
                      n_patches: int = cfg.DINOV2_SEAM_N_PATCHES,
                      patch_offset_px: float = cfg.DINOV2_SEAM_PATCH_OFFSET_PX,
                      sample_span_px: float = 60.0
                      ) -> tuple[float, int]:
    """
    Cosine similarity between fragments, sampled as `n_patches` pairs
    across the proposed seam.

    Both M_a and M_b are 3x3 affines from each fragment's frame into the
    shared canvas. `seam_point`/`seam_normal` live in the canvas frame.
    The normal points from A's side toward B's side.

    Returns (mean_cosine_in_[0,1], n_valid_pairs). If no valid pairs are
    found, returns (0.0, 0).
    """
    info_a = frag_a.get("dinov2")
    info_b = frag_b.get("dinov2")
    if info_a is None or info_b is None:
        return 0.0, 0

    n = np.asarray(seam_normal, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return 0.0, 0
    n = n / n_norm
    # tangent along the seam
    t = np.array([-n[1], n[0]], dtype=np.float64)

    sp = np.asarray(seam_point, dtype=np.float64)

    # Spread sample centres along the seam.
    if n_patches <= 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-sample_span_px * 0.5,
                               sample_span_px * 0.5, n_patches)

    Ma_inv = np.linalg.inv(np.asarray(M_a, dtype=np.float64))
    Mb_inv = np.linalg.inv(np.asarray(M_b, dtype=np.float64))

    cosines = []
    for s in offsets:
        along = sp + s * t
        pa_canvas = along - patch_offset_px * n   # A side
        pb_canvas = along + patch_offset_px * n   # B side

        # canvas -> each fragment's own image frame
        pa_img = _apply_affine(Ma_inv, pa_canvas)[0]
        pb_img = _apply_affine(Mb_inv, pb_canvas)[0]

        fa_uv = _image_xy_to_feat_uv(info_a, pa_img)[0]
        fb_uv = _image_xy_to_feat_uv(info_b, pb_img)[0]

        fa = _sample_feature(info_a, fa_uv)
        fb = _sample_feature(info_b, fb_uv)
        if fa is None or fb is None:
            continue
        cosines.append(float(np.dot(fa, fb)))

    if not cosines:
        return 0.0, 0
    mean_cos = float(np.mean(cosines))
    # Map [-1, 1] -> [0, 1] (cosine is mostly non-negative on paper crops
    # but clamp anyway so this is a valid score).
    return max(0.0, min(1.0, 0.5 * (mean_cos + 1.0))), len(cosines)
