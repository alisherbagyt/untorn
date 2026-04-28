"""
untorn.inpainting
=================
LaMa-powered hole filling. Phase 5 is orchestrated by ``untorn.gap_fill``;
this module provides the LaMa backend (JIT / simple_lama / saicinpainting)
plus a text-preserving scar-mask builder and tiled inference for big
canvases. The FastAPI backend's Assembly export route also calls
``inpaint`` directly.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config as cfg
from .io_utils import save_image, save_mask


# ── Paths ─────────────────────────────────────────────────────────────────────

LAMA_DIR        = cfg.PROJECT_ROOT / "lama"
LAMA_BIG_DIR    = LAMA_DIR / "big-lama"
LAMA_CONFIG     = LAMA_BIG_DIR / "config.yaml"
LAMA_CHECKPOINT = LAMA_BIG_DIR / "models" / "best.ckpt"

# Optional TorchScript build — no saicinpainting/pytorch-lightning/kornia needed.
# Place a JIT-compiled big-lama here or point LAMA_JIT_PATH to one. Download:
#   https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
LAMA_JIT_PATH = Path(os.environ.get("LAMA_JIT_PATH", LAMA_BIG_DIR / "big-lama.pt"))


# ── Tunables (can be overridden via env) ──────────────────────────────────────

INPAINT_BAND_PX      = 4       # half-width of the scar ring (total width = 2*band)
INPAINT_INK_THRESH   = 140     # pixels darker than this are considered ink
INPAINT_TILE_SIZE    = 512
INPAINT_TILE_OVERLAP = 64
INPAINT_MIN_MASK_PX  = 8       # drop components smaller than this


# ── Singleton state ───────────────────────────────────────────────────────────

_predictor = None   # object with .kind ("jit" | "simple" | "saic") + .predict(img, mask)
_model_lock = threading.Lock()


def is_available() -> bool:
    """True iff ANY supported LaMa backend looks loadable on this system."""
    if LAMA_JIT_PATH.exists():
        return True
    import importlib.util
    if importlib.util.find_spec("simple_lama_inpainting") is not None:
        return True
    return LAMA_CHECKPOINT.exists() and LAMA_CONFIG.exists()


def _ensure_lama_importable():
    """Make the lama repo importable and set TORCH_HOME for resnet_pl lookups."""
    os.environ.setdefault("TORCH_HOME", str(LAMA_DIR))
    lama_str = str(LAMA_DIR)
    if lama_str not in sys.path:
        sys.path.insert(0, lama_str)


class _JitPredictor:
    """TorchScript big-lama.pt — zero extra deps beyond torch."""
    kind = "jit"

    def __init__(self, device):
        import torch
        self.device = device
        self.model = torch.jit.load(str(LAMA_JIT_PATH), map_location=device)
        self.model.eval()

    def predict(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        img = image_rgb.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        m = (mask > 127).astype(np.float32)
        m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)

        h, w = img_t.shape[-2:]
        mod = 8
        pad_h = (mod - h % mod) % mod
        pad_w = (mod - w % mod) % mod
        img_p = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
        m_p = F.pad(m_t, (0, pad_w, 0, pad_h), mode="reflect")
        m_p = (m_p > 0).float()

        with torch.no_grad():
            out = self.model(img_p.to(self.device), m_p.to(self.device))
        out = out[0, :, :h, :w].detach().cpu().permute(1, 2, 0).numpy()
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)


class _SimpleLamaPredictor:
    """Uses the `simple_lama_inpainting` PyPI package — also TorchScript under the hood."""
    kind = "simple"

    def __init__(self, device):
        from simple_lama_inpainting import SimpleLama
        # SimpleLama picks device from LAMA_MODEL env var / defaults.
        # Point it at our local JIT file if we have one; else it downloads on first use.
        if LAMA_JIT_PATH.exists():
            os.environ.setdefault("LAMA_MODEL", str(LAMA_JIT_PATH))
        self.simple = SimpleLama(device=device)
        self.device = device

    def predict(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image
        pil_img = Image.fromarray(image_rgb, "RGB")
        pil_mask = Image.fromarray((mask > 127).astype(np.uint8) * 255, "L")
        out = self.simple(pil_img, pil_mask)  # PIL RGB
        return np.array(out)


class _SaicPredictor:
    """Original saicinpainting / best.ckpt path — needs the full lama env."""
    kind = "saic"

    def __init__(self, device):
        _ensure_lama_importable()
        import yaml
        from omegaconf import OmegaConf
        from saicinpainting.training.trainers import load_checkpoint

        with open(LAMA_CONFIG) as f:
            train_cfg = OmegaConf.create(yaml.safe_load(f))
        train_cfg.training_model.predict_only = True
        train_cfg.visualizer.kind = "noop"

        model = load_checkpoint(train_cfg, str(LAMA_CHECKPOINT),
                                strict=False, map_location="cpu")
        model.freeze()
        model.to(device)
        self.model = model
        self.device = device

    def predict(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch
        from saicinpainting.evaluation.data import pad_tensor_to_modulo

        img = image_rgb.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        m = (mask > 127).astype(np.float32)
        m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)

        h, w = img_t.shape[-2:]
        img_p = pad_tensor_to_modulo(img_t, 8)
        m_p = pad_tensor_to_modulo(m_t, 8)

        with torch.no_grad():
            batch = {"image": img_p.to(self.device), "mask": m_p.to(self.device)}
            batch["mask"] = (batch["mask"] > 0).float()
            batch = self.model(batch)
            out = batch["inpainted"][0].detach().cpu()
        out = out[:, :h, :w].permute(1, 2, 0).numpy()
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _load_model():
    """
    Load a LaMa predictor once per process, trying (in order):
      1. Local TorchScript at LAMA_JIT_PATH          (lightest — torch only)
      2. simple_lama_inpainting PyPI package         (downloads model on demand)
      3. saicinpainting + best.ckpt                  (legacy, heavy deps)
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    with _model_lock:
        if _predictor is not None:
            return _predictor

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        errors: list[str] = []

        # 1. Local TorchScript
        if LAMA_JIT_PATH.exists():
            try:
                _predictor = _JitPredictor(device)
                print(f"[LaMa] Loaded TorchScript model from {LAMA_JIT_PATH}")
                return _predictor
            except Exception as exc:
                errors.append(f"jit: {exc}")

        # 2. simple_lama_inpainting
        try:
            _predictor = _SimpleLamaPredictor(device)
            print("[LaMa] Loaded via simple_lama_inpainting package")
            return _predictor
        except Exception as exc:
            errors.append(f"simple_lama_inpainting: {exc}")

        # 3. saicinpainting (legacy)
        if LAMA_CHECKPOINT.exists() and LAMA_CONFIG.exists():
            try:
                _predictor = _SaicPredictor(device)
                print("[LaMa] Loaded via saicinpainting (best.ckpt)")
                return _predictor
            except Exception as exc:
                errors.append(f"saicinpainting: {exc}")

        raise RuntimeError(
            "LaMa: no loadable backend.\n"
            "Tried:\n  - " + "\n  - ".join(errors) + "\n"
            "Fix with ONE of:\n"
            f"  a) Drop a TorchScript big-lama at {LAMA_JIT_PATH}\n"
            "  b) pip install simple-lama-inpainting\n"
            "  c) pip install -r lama/requirements.txt (heavy, version-locked)"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Scar-mask construction
# ══════════════════════════════════════════════════════════════════════════════

def build_scar_mask(
    image_rgb: np.ndarray,
    coverage: np.ndarray,
    gap_mask: Optional[np.ndarray] = None,
    band_px: int = INPAINT_BAND_PX,
    ink_threshold: int = INPAINT_INK_THRESH,
) -> np.ndarray:
    """
    Build a binary mask marking the torn-edge seams between fragments while
    explicitly excluding ink (text) pixels.

    Strategy:
      1. Take a symmetric ring of width 2*band_px around every coverage edge.
      2. Union with the optional gap_mask (inside-hull but uncovered).
      3. Subtract a slightly-dilated ink mask (dark pixels) so LaMa never
         gets told to repaint text.
      4. Close small holes, drop speckle components smaller than MIN_MASK_PX.

    Args:
        image_rgb:    HxWx3 uint8 composite (what will be inpainted).
        coverage:     HxW uint8 — non-zero where any fragment is placed.
        gap_mask:     optional HxW uint8 — uncovered pixels inside the paper hull.
        band_px:      half-width of the scar ring in pixels.
        ink_threshold: luminance below which a pixel is treated as text ink.

    Returns:
        HxW uint8 mask with values in {0, 255}.
    """
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3
    assert coverage.ndim == 2
    h, w = coverage.shape
    cov = (coverage > 127).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(cov, kernel, iterations=band_px)
    eroded  = cv2.erode (cov, kernel, iterations=band_px)
    ring    = cv2.subtract(dilated, eroded)

    if gap_mask is not None:
        gm = (gap_mask > 127).astype(np.uint8) * 255
        ring = cv2.bitwise_or(ring, gm)

    # Text-preservation: exclude dark pixels with a small safety dilation
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ink = (gray < ink_threshold).astype(np.uint8) * 255
    ink = cv2.dilate(ink, kernel, iterations=1)
    scar = cv2.bitwise_and(ring, cv2.bitwise_not(ink))

    # Clean up
    scar = cv2.morphologyEx(scar, cv2.MORPH_CLOSE, kernel)

    # Drop tiny components
    if INPAINT_MIN_MASK_PX > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(scar, connectivity=8)
        keep = np.zeros_like(scar)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= INPAINT_MIN_MASK_PX:
                keep[labels == i] = 255
        scar = keep

    return scar


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════

def _predict_once(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Single-pass full-image inference via whichever backend loaded."""
    return _load_model().predict(image_rgb, mask)


def _predict_refined(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Refinement pass — only available when the saicinpainting backend loaded,
    since refine_predict lives in that codebase. Falls back to single-pass
    with a warning for JIT / simple_lama backends.
    """
    predictor = _load_model()
    if predictor.kind != "saic":
        print(f"  ! refine mode unavailable for backend '{predictor.kind}', "
              f"falling back to standard LaMa pass")
        return predictor.predict(image_rgb, mask)

    import torch
    from saicinpainting.evaluation.refinement import refine_predict

    img = image_rgb.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    m = (mask > 127).astype(np.float32)
    m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)

    h, w = img_t.shape[-2:]
    batch = {
        "image": img_t,
        "mask": m_t,
        "unpad_to_size": torch.tensor([[h], [w]]),
    }

    gpu_ids = "0," if predictor.device.type == "cuda" else ""
    out = refine_predict(
        batch, predictor.model,
        gpu_ids=gpu_ids,
        modulo=8, n_iters=15, lr=0.002,
        min_side=512, max_scales=3, px_budget=1_800_000,
    )
    out = out[0].permute(1, 2, 0).detach().cpu().numpy()
    out = out[:h, :w]
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def _predict_tiled(
    image_rgb: np.ndarray, mask: np.ndarray,
    tile: int = INPAINT_TILE_SIZE, overlap: int = INPAINT_TILE_OVERLAP,
) -> np.ndarray:
    """
    Tiled inference: only tiles containing mask pixels are run through LaMa.
    The rest of the image is returned untouched.

    Output pixels are composited via a feathered mask so non-mask regions
    match the input exactly and seams between tiles blend.
    """
    h, w = image_rgb.shape[:2]
    out = image_rgb.copy()

    # Feather: 1 at mask center, fades to 0 at mask boundary
    # We blend lama output into the original image only where the mask is.
    mask_bin = (mask > 127).astype(np.uint8)
    if mask_bin.sum() == 0:
        return out

    # Smooth alpha over the mask so the inpainted region blends softly into
    # neighbouring untouched pixels.
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3)
    feather = np.clip(dist / max(overlap // 2, 1), 0, 1).astype(np.float32)

    step = tile - overlap
    ys = list(range(0, max(1, h - overlap), step))
    xs = list(range(0, max(1, w - overlap), step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    ys = [max(0, y) for y in ys]
    xs = [max(0, x) for x in xs]

    for y in ys:
        for x in xs:
            y2 = min(h, y + tile)
            x2 = min(w, x + tile)
            y1 = max(0, y2 - tile)
            x1 = max(0, x2 - tile)

            tile_mask = mask[y1:y2, x1:x2]
            if int((tile_mask > 127).sum()) == 0:
                continue

            tile_img = out[y1:y2, x1:x2]
            predicted = _predict_once(tile_img, tile_mask)

            a = feather[y1:y2, x1:x2, None]
            blend = predicted.astype(np.float32) * a + tile_img.astype(np.float32) * (1 - a)
            out[y1:y2, x1:x2] = np.clip(blend, 0, 255).astype(np.uint8)

    return out


def inpaint(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    tile: bool = True,
    refine: bool = False,
) -> np.ndarray:
    """
    Public inpainting entry point.

    Args:
        image_rgb: HxWx3 uint8.
        mask:      HxW uint8 (0/255). Non-zero pixels will be repainted.
        tile:      if True, run tiled inference (skips tiles with no mask).
        refine:    if True, use LaMa's full refinement pipeline (slow).

    Returns:
        HxWx3 uint8. Non-mask pixels are preserved from the input.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    if int((mask > 127).sum()) == 0:
        return image_rgb.copy()

    if refine:
        predicted = _predict_refined(image_rgb, mask)
    elif tile:
        return _predict_tiled(image_rgb, mask)
    else:
        predicted = _predict_once(image_rgb, mask)

    # Composite: only replace pixels where mask is set.
    m = (mask > 127).astype(np.uint8)
    m3 = np.stack([m] * 3, axis=-1)
    return np.where(m3 > 0, predicted, image_rgb)


# Phase 5 is orchestrated by `untorn.gap_fill.inpaint_gaps`; the legacy
# `clean_final` entrypoint that lived here was removed in Step 1 cleanup.
