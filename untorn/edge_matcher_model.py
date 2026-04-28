"""Siamese edge-matcher CNN — Step 6 of Phase 2.

Two-tower shared-weight CNN that ingests pairs of 32×256 RGB edge strips and
emits:
    * a binary match probability via cosine similarity of L2-normalized
      embeddings (passed through a learnable temperature + sigmoid)
    * a 3-DOF relative-pose regression (Δθ, Δdx, Δdy) on positive pairs only

Architecture (≈ 1.2M parameters, fits comfortably in 4 GB VRAM):

    Input  (B, 3, 32, 256)
        ↓
    Tower:
        Conv 3→32, BN, ReLU, Pool (2×2)            (B, 32, 16, 128)
        Conv 32→64, BN, ReLU, Pool (2×2)           (B, 64,  8,  64)
        Conv 64→128, BN, ReLU, Pool (2×2)          (B,128,  4,  32)
        Conv 128→256, BN, ReLU, AdaptiveAvgPool    (B,256,  1,   1)
        Linear 256→256, L2-normalize               (B,256)
        ↓
    Match head: cosine(emb_a, emb_b) → ×τ → sigmoid → P(match)
    Pose head : Linear(2·256→128)→ReLU→Linear(128→3)  (Δθ, Δdx, Δdy)

Channel-major NCHW order is used throughout (standard PyTorch). The dataset
loader is responsible for converting (32, 256, 3) uint8 strips to (3, 32, 256)
float32 in the [0, 1] range.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["EdgeMatcher", "EdgeMatcherOutput", "build_edge_matcher",
            "edge_matcher_loss"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EdgeMatcherConfig:
    in_channels: int = 3
    embed_dim: int = 256
    pose_hidden_dim: int = 128
    init_temperature: float = 10.0       # logit_scale for cosine sim
    bn_momentum: float = 0.1


# ---------------------------------------------------------------------------
# Tower (shared-weight feature extractor)
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv → BN → ReLU (no pool) — pool applied separately to keep code tidy."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class EdgeTower(nn.Module):
    """Shared-weight tower turning a 32×256 strip into a 256-D embedding."""

    def __init__(self, cfg: EdgeMatcherConfig):
        super().__init__()
        c0 = cfg.in_channels
        # Four downsampling blocks: 32×256 → 16×128 → 8×64 → 4×32 → 1×1.
        self.b1 = _ConvBlock(c0, 32)
        self.b2 = _ConvBlock(32, 64)
        self.b3 = _ConvBlock(64, 128)
        self.b4 = _ConvBlock(128, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 32, 256)
        x = F.max_pool2d(self.b1(x), 2)    # (B, 32, 16, 128)
        x = F.max_pool2d(self.b2(x), 2)    # (B, 64,  8,  64)
        x = F.max_pool2d(self.b3(x), 2)    # (B,128,  4,  32)
        x = self.b4(x)                     # (B,256,  4,  32)
        x = self.gap(x).flatten(1)         # (B, 256)
        x = self.proj(x)                   # (B, embed_dim)
        return F.normalize(x, dim=1)       # L2-normalized


# ---------------------------------------------------------------------------
# Full Siamese model
# ---------------------------------------------------------------------------

@dataclass
class EdgeMatcherOutput:
    match_logit: torch.Tensor    # (B,)  raw logit (before sigmoid)
    match_prob: torch.Tensor     # (B,)  sigmoid of match_logit
    pose_pred: torch.Tensor      # (B, 3) (Δθ, Δdx, Δdy) in normalized units
    embed_a: torch.Tensor        # (B, D)
    embed_b: torch.Tensor        # (B, D)


class EdgeMatcher(nn.Module):
    """Siamese network: shared tower + cosine match head + pose regression."""

    def __init__(self, cfg: EdgeMatcherConfig):
        super().__init__()
        self.cfg = cfg
        self.tower = EdgeTower(cfg)
        # logit_scale (a.k.a. temperature) — learnable, initialized to a value
        # that gives reasonable initial probabilities; bounded for stability.
        self.logit_scale = nn.Parameter(
            torch.tensor(float(cfg.init_temperature)).log()
        )
        self.pose_head = nn.Sequential(
            nn.Linear(2 * cfg.embed_dim, cfg.pose_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.pose_hidden_dim, 3),
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Clamped exp(logit_scale)."""
        return self.logit_scale.exp().clamp(max=100.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of strips → embeddings. Public for offline use."""
        return self.tower(x)

    def forward(self, strip_a: torch.Tensor, strip_b: torch.Tensor
                 ) -> EdgeMatcherOutput:
        ea = self.tower(strip_a)
        eb = self.tower(strip_b)
        cos = (ea * eb).sum(dim=1)               # (B,) since both L2-normed
        logit = cos * self.temperature
        prob = torch.sigmoid(logit)

        pose_pred = self.pose_head(torch.cat([ea, eb], dim=1))

        return EdgeMatcherOutput(
            match_logit=logit, match_prob=prob, pose_pred=pose_pred,
            embed_a=ea, embed_b=eb,
        )

    def predict(self, strip_a: torch.Tensor, strip_b: torch.Tensor
                 ) -> dict:
        """Convenience inference helper. Returns CPU floats per pair."""
        self.eval()
        with torch.no_grad():
            out = self.forward(strip_a, strip_b)
        return {
            "match_prob": out.match_prob.cpu().numpy(),
            "pose_pred": out.pose_pred.cpu().numpy(),
            "cosine": (out.embed_a * out.embed_b).sum(dim=1).cpu().numpy(),
        }


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

def edge_matcher_loss(out: EdgeMatcherOutput,
                       label: torch.Tensor,
                       pose_target: torch.Tensor,
                       pose_weight: float = 0.1) -> dict:
    """Combined BCE + masked-L1 pose loss.

    Args:
        out: model output.
        label: (B,) float in {0, 1} — match label.
        pose_target: (B, 3) — relative (Δθ, Δdx, Δdy). Only positives are
            used; negatives are masked out.
        pose_weight: scalar weight on pose loss.

    Returns dict with `loss`, `bce`, `pose_l1` for logging.
    """
    bce = F.binary_cross_entropy_with_logits(out.match_logit, label.float())

    pos_mask = label > 0.5
    if pos_mask.any():
        pose_l1 = F.l1_loss(out.pose_pred[pos_mask], pose_target[pos_mask])
    else:
        pose_l1 = torch.zeros((), device=out.match_logit.device)

    loss = bce + pose_weight * pose_l1
    return {"loss": loss, "bce": bce, "pose_l1": pose_l1}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_edge_matcher(**overrides) -> EdgeMatcher:
    """Construct an EdgeMatcher with the default config plus any overrides."""
    cfg = EdgeMatcherConfig(**overrides)
    return EdgeMatcher(cfg)


# ---------------------------------------------------------------------------
# Self-test (run this file directly to verify shapes & param count)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_edge_matcher()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EdgeMatcher built. Parameters: {n_params:,}")

    B = 4
    strip_a = torch.randn(B, 3, 32, 256)
    strip_b = torch.randn(B, 3, 32, 256)
    out = model(strip_a, strip_b)
    print("match_prob:", tuple(out.match_prob.shape), out.match_prob.dtype)
    print("pose_pred :", tuple(out.pose_pred.shape))
    print("embed_a   :", tuple(out.embed_a.shape))
    print("temperature:", float(model.temperature))

    label = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    pose = torch.zeros(B, 3)
    losses = edge_matcher_loss(out, label, pose)
    print("losses:", {k: float(v) for k, v in losses.items()})
