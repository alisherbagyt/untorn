"""
untorn.edge_matcher
===================
Inference-time adapter for the Siamese edge matcher (Phase 4 — Step 11/12).

This module is the bridge between the trained ``EdgeMatcher`` checkpoint
(``untorn.edge_matcher_model``) and the live matching pipeline. It does
three things:

1. **Lifecycle**: a lazy module-level singleton so ``pipeline.py`` can call
   ``edge_matcher.load()`` once before Phase 3 and ``edge_matcher.unload()``
   afterwards. The model holds ~520 K params (~2 MB) but pinning it on the
   GPU keeps inference per-pair at ~1 ms instead of paying setup each call.

2. **Strip extraction**: re-creates the EXACT (32, 256, 3) RGB strip format
   used by ``tools/build_edge_dataset.py`` so live edge pairs feed the
   network the same input distribution it trained on. Row 0 sits ON the
   torn edge; the 32-pixel axis goes inward into the fragment; the
   256-pixel axis runs along the boundary in arc-length order.

3. **Scoring**: ``score_edge_pair(image_rgb, edge_a, edge_b, orientation)``
   returns ``{"match_prob", "cosine", "pose_pred"}`` for a single edge
   pairing. ``matching._match_edge_pair`` calls this AFTER ICP refinement
   and rejects pairs whose ``match_prob < cfg.EDGE_MATCHER_MIN_SCORE``.

Graceful degradation
--------------------
* If torch is unavailable, ``load()`` logs a warning and returns ``None``.
* If the checkpoint file is missing, ``load()`` logs and returns ``None``.
* ``score_edge_pair`` returns ``None`` whenever the model isn't loaded —
  the caller then skips the gate, so the pipeline behaves identically to
  pre-Phase-4 when the model isn't available.

The strip extraction logic mirrors training:
    * For ``orientation == "complementary"`` (the only case the matcher
      keeps; ``MATCH_REJECT_DIRECT`` enforces this), edge B's polyline is
      reversed before sampling so column k of strip A corresponds to the
      same intended seam point as column k of strip B.
    * Inward direction at every sample is the local tangent rotated 90 deg
      and sign-corrected against the edge's stored outward_normal so the
      strip walks INTO the fragment regardless of arc-length direction.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from . import config as cfg


# Strip dimensions are fixed by the trained model.
_STRIP_W = 32         # rows perpendicular to the edge (going inward)
_STRIP_L = 256        # columns along the edge (arc-length)


# ─── Lazy singleton state ─────────────────────────────────────────────────

_LOAD_LOCK   = threading.Lock()
_MODEL       = None        # type: ignore[assignment]
_DEVICE      = None        # type: ignore[assignment]
_LOAD_FAILED = False       # sticky once load() decides the model isn't usable
_LOAD_REASON = ""          # human-readable reason for the failure


def is_loaded() -> bool:
    """Return True if a model is currently resident in memory."""
    return _MODEL is not None


def load(checkpoint_path: str | Path | None = None,
         device: str | None = None) -> object | None:
    """
    Load the Siamese edge-matcher checkpoint into memory (idempotent).

    Returns the model object on success, or ``None`` on any failure
    (missing torch, missing checkpoint, CUDA OOM, etc.). Errors are
    logged and remembered so subsequent calls are no-ops.
    """
    global _MODEL, _DEVICE, _LOAD_FAILED, _LOAD_REASON

    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None

    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL
        if _LOAD_FAILED:
            return None

        if not getattr(cfg, "EDGE_MATCHER_ENABLED", True):
            _LOAD_FAILED = True
            _LOAD_REASON = "disabled by cfg.EDGE_MATCHER_ENABLED"
            return None

        if checkpoint_path is None:
            checkpoint_path = getattr(cfg, "EDGE_MATCHER_CHECKPOINT",
                                       "models/edge_matcher.pt")
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (cfg.PROJECT_ROOT / ckpt_path).resolve()
        if not ckpt_path.exists():
            print(f"  ! Edge matcher checkpoint missing: {ckpt_path}")
            print(f"  ! Phase 4 Siamese gate disabled - pipeline will run "
                  f"with the legacy 4-gate cascade.")
            _LOAD_FAILED = True
            _LOAD_REASON = f"checkpoint missing: {ckpt_path}"
            return None

        try:
            import torch
        except ImportError as exc:
            print(f"  ! Edge matcher: torch unavailable ({exc}); skipping.")
            _LOAD_FAILED = True
            _LOAD_REASON = f"torch import failed: {exc}"
            return None

        from .edge_matcher_model import build_edge_matcher

        if device is None:
            device = getattr(cfg, "EDGE_MATCHER_DEVICE", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            print("  ! Edge matcher: CUDA requested but unavailable; using CPU.")
            device = "cpu"
        torch_device = torch.device(device)

        try:
            ckpt = torch.load(ckpt_path, map_location=torch_device,
                               weights_only=False)
            model = build_edge_matcher().to(torch_device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
        except Exception as exc:
            print(f"  ! Edge matcher: failed to load checkpoint ({exc}); "
                  f"skipping.")
            _LOAD_FAILED = True
            _LOAD_REASON = f"checkpoint load failed: {exc}"
            return None

        _MODEL = model
        _DEVICE = torch_device
        ep = ckpt.get("epoch")
        vm = ckpt.get("val_metrics") or {}
        auc = vm.get("auc") if isinstance(vm, dict) else None
        print(f"  + Edge matcher loaded from {ckpt_path.name}  "
              f"(epoch={ep}, val_auc={auc}, device={device})")
        return _MODEL


def unload() -> None:
    """Release the model and free GPU memory. Safe to call when not loaded."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        return
    try:
        import torch
        del _MODEL
        if _DEVICE is not None and _DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
    _MODEL = None
    _DEVICE = None


# ─── Strip extraction (mirrors tools/build_edge_dataset.py) ────────────────

def _resample_arc_length(pts: np.ndarray, n_samples: int) -> np.ndarray | None:
    """Resample a polyline to ``n_samples`` equally-spaced arc-length points.

    Returns (n_samples, 2) float32 or None if the polyline has zero length.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return None
    diffs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    total = float(seg_len.sum())
    if total <= 0.0:
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.linspace(0.0, total, n_samples)
    out = np.empty((n_samples, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(targets):
        while j < len(seg_len) - 1 and cum[j + 1] < t:
            j += 1
        denom = float(seg_len[j]) if seg_len[j] > 1e-9 else 1.0
        alpha = (t - cum[j]) / denom if seg_len[j] > 1e-9 else 0.0
        out[i] = pts[j] + alpha * (pts[j + 1] - pts[j])
    return out


def _per_sample_inward(curve: np.ndarray,
                       edge_outward_normal: np.ndarray) -> np.ndarray:
    """Per-sample unit vector pointing INTO the fragment.

    Equivalent to training's ``_outward_normals`` flipped: take the local
    tangent rotated 90 deg and sign-correct against the edge's stored
    outward normal. We use the global outward_normal as a sign anchor so we
    don't need the mask at inference time.
    """
    diffs = np.diff(curve, axis=0, append=curve[-1:])
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    tangents = diffs / norms                              # (N, 2)
    # Rotate +90 deg → candidate normal.
    candidate = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    # Inward = candidate when candidate · outward_normal < 0.
    on = np.asarray(edge_outward_normal, dtype=np.float32).reshape(2,)
    on_norm = float(np.linalg.norm(on))
    if on_norm < 1e-9:
        # No reliable anchor — return raw candidate; rare and the network
        # is robust to ±sign flips on a few samples.
        return candidate.astype(np.float32)
    on = on / on_norm
    dot = candidate @ on
    # Where dot > 0 the candidate points outward → flip to get inward.
    inward = np.where(dot[:, None] > 0.0, -candidate, candidate)
    return inward.astype(np.float32)


def _sample_strip(image_rgb: np.ndarray,
                  curve: np.ndarray,
                  inward_dir: np.ndarray,
                  strip_w: int) -> np.ndarray:
    """Bilinear-sample a (strip_w, len(curve), 3) uint8 strip."""
    n_samples = curve.shape[0]
    steps = np.arange(strip_w, dtype=np.float32)[:, None, None]   # (W,1,1)
    base = curve[None, :, :]                                       # (1,N,2)
    direction = inward_dir[None, :, :]                             # (1,N,2)
    coords = base + steps * direction                              # (W,N,2)

    map_x = coords[..., 0].astype(np.float32)
    map_y = coords[..., 1].astype(np.float32)
    return cv2.remap(image_rgb, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0))


def extract_strip_pair(image_rgb: np.ndarray,
                        edge_a: dict, edge_b: dict,
                        orientation: str,
                        strip_w: int = _STRIP_W,
                        strip_l: int = _STRIP_L
                        ) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build the (strip_a, strip_b) input pair for the Siamese network.

    Both strips are sampled from the live ``image_rgb`` (the working-resolution
    composite) at the live torn-edge polylines. For ``"complementary"``
    orientation we reverse edge B's traversal so column k of strip A and
    column k of strip B correspond to the same intended seam point — exactly
    how the training dataset was built.

    Returns ``(strip_a, strip_b)`` each ``(strip_w, strip_l, 3)`` uint8, or
    ``None`` if either edge is too degenerate to resample.
    """
    pts_a = np.asarray(edge_a["pts"], dtype=np.float64)
    pts_b = np.asarray(edge_b["pts"], dtype=np.float64)

    curve_a = _resample_arc_length(pts_a, strip_l)
    curve_b = _resample_arc_length(pts_b, strip_l)
    if curve_a is None or curve_b is None:
        return None

    # Complementary tear surfaces are traversed in opposite directions when
    # placed face-to-face. Reversing curve_b here aligns column-k indexing
    # with the training-time convention.
    if orientation == "complementary":
        curve_b = curve_b[::-1].copy()

    inward_a = _per_sample_inward(curve_a, edge_a["outward_normal"])
    inward_b = _per_sample_inward(curve_b, edge_b["outward_normal"])

    strip_a = _sample_strip(image_rgb, curve_a, inward_a, strip_w)
    strip_b = _sample_strip(image_rgb, curve_b, inward_b, strip_w)
    return strip_a, strip_b


# ─── Scoring entry point used by matching._match_edge_pair ─────────────────

def score_edge_pair(image_rgb: np.ndarray,
                    edge_a: dict, edge_b: dict,
                    orientation: str) -> dict | None:
    """
    Run the Siamese network on a single edge pair.

    Returns a dict with keys ``match_prob`` (float), ``cosine`` (float),
    ``pose_pred`` ((3,) float array, normalized (Δθ, Δdx, Δdy)).
    Returns ``None`` if the model isn't loaded or strip extraction failed.
    """
    if _MODEL is None:
        return None

    pair = extract_strip_pair(image_rgb, edge_a, edge_b, orientation)
    if pair is None:
        return None
    strip_a, strip_b = pair

    import torch

    # (H, W, 3) uint8 → (1, 3, H, W) float32 in [0, 1] — matches training.
    ta = torch.from_numpy(np.ascontiguousarray(strip_a)) \
        .permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(_DEVICE)
    tb = torch.from_numpy(np.ascontiguousarray(strip_b)) \
        .permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(_DEVICE)

    with torch.no_grad():
        out = _MODEL(ta, tb)

    return {
        "match_prob": float(out.match_prob[0].item()),
        "cosine":     float((out.embed_a * out.embed_b).sum(dim=1)[0].item()),
        "pose_pred":  out.pose_pred[0].detach().cpu().numpy().astype(np.float64),
    }


# ─── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Run-as-script: synthesize an edge pair and verify the strip pipeline.

    No network checkpoint is required; the strip extraction stands on its own.
    """
    rng = np.random.default_rng(0)

    H, W = 200, 400
    img = (rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    # Two horizontal edges 10 px apart, going left-to-right.
    pts_a = np.stack([np.linspace(50, 350, 80),
                       np.full(80, 100.0)], axis=1)
    pts_b = np.stack([np.linspace(50, 350, 80),
                       np.full(80, 110.0)], axis=1)
    edge_a = {"pts": pts_a, "outward_normal": np.array([0.0, -1.0])}   # up
    edge_b = {"pts": pts_b, "outward_normal": np.array([0.0,  1.0])}   # down

    pair = extract_strip_pair(img, edge_a, edge_b, "complementary")
    assert pair is not None, "extract_strip_pair returned None on a clean pair"
    sa, sb = pair
    assert sa.shape == (32, 256, 3), f"strip_a wrong shape: {sa.shape}"
    assert sb.shape == (32, 256, 3), f"strip_b wrong shape: {sb.shape}"
    assert sa.dtype == np.uint8 and sb.dtype == np.uint8
    print(f"strip_a shape={sa.shape} dtype={sa.dtype}  "
          f"mean={sa.mean():.1f}")
    print(f"strip_b shape={sb.shape} dtype={sb.dtype}  "
          f"mean={sb.mean():.1f}")

    # Try loading. With no checkpoint at the default path the test still
    # passes silently — the goal here is to exercise the import + lifecycle.
    m = load()
    if m is None:
        print(f"load() returned None (expected if checkpoint missing): "
              f"{_LOAD_REASON}")
    else:
        print(f"load() returned a model on {_DEVICE}.  "
              f"is_loaded()={is_loaded()}")
        scored = score_edge_pair(img, edge_a, edge_b, "complementary")
        print(f"score_edge_pair -> {scored}")
        unload()
        print(f"after unload() is_loaded={is_loaded()}")
    print("edge_matcher self-test OK")
