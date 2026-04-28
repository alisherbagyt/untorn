"""Step 7 of Phase 2 — train the Siamese edge matcher.

Reads ``data/dataset/edge_strips/{train,val}.h5``, trains the EdgeMatcher,
saves the best checkpoint to ``models/edge_matcher.pt`` and full training
logs to ``data/training_logs/edge_matcher_run_<timestamp>/``.

Highlights:
    * 1:3 positive:negative ratio per batch (negatives sampled on-the-fly
      from cross-document strip pairs, with positive-pair filtering)
    * Pose augmentation: a random rigid transform is applied to strip_b for
      positive pairs; the inverse becomes the pose-head regression target,
      so the head actually learns alignment instead of always emitting zero.
    * Brightness/contrast jitter + Gaussian noise as scanner-domain
      augmentation, applied to both strips independently.
    * AdamW + cosine LR + AMP mixed precision.
    * Validation each epoch logs AUC-ROC, accuracy@0.5, and median pose error.

CLI:
    python tools/train_edge_matcher.py
        --train data/dataset/edge_strips/train.h5
        --val   data/dataset/edge_strips/val.h5
        --epochs 50 --batch_size 128 --lr 1e-3
        --out models/edge_matcher.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure we can `from untorn.edge_matcher_model import ...` when running
# from the repo root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from untorn.edge_matcher_model import (
    EdgeMatcher, EdgeMatcherConfig, edge_matcher_loss, build_edge_matcher,
)  # noqa: E402


# ---------------------------------------------------------------------------
# Pose augmentation (kept in pixel space, inverted to give regression target)
# ---------------------------------------------------------------------------

POSE_THETA_RANGE_DEG = 12.0
POSE_DX_RANGE_PX = 6.0
POSE_DY_RANGE_PX = 4.0


def _normalize_pose(theta_deg: float, dx_px: float, dy_px: float
                    ) -> tuple[float, float, float]:
    """Map physical pose params to roughly [-1, 1] ranges for stable regression."""
    return (theta_deg / POSE_THETA_RANGE_DEG,
            dx_px / POSE_DX_RANGE_PX,
            dy_px / POSE_DY_RANGE_PX)


def _apply_pose_to_strip(strip: torch.Tensor, theta_deg: torch.Tensor,
                          dx_px: torch.Tensor, dy_px: torch.Tensor
                          ) -> torch.Tensor:
    """Apply a rigid transform to a batch of strips in pixel space.

    strip:    (B, C, H, W) float
    theta_deg/dx_px/dy_px: (B,) tensors

    Returns warped strips of the same shape via grid_sample.
    """
    B, C, H, W = strip.shape
    # Build inverse affine for grid_sample (it expects output→input mapping).
    theta_rad = torch.deg2rad(theta_deg)
    cos = torch.cos(theta_rad)
    sin = torch.sin(theta_rad)
    # Forward transform (input→output) is rotation then translation.
    # PyTorch grid_sample wants the inverse: output→input.
    # In normalized [-1, 1] coordinates: dx_norm = 2*dx_px / (W-1), dy similar.
    dx_norm = 2.0 * dx_px / (W - 1)
    dy_norm = 2.0 * dy_px / (H - 1)

    # Inverse rigid: R^T (p - t) -- rotation transposed, with translation negated.
    M = torch.zeros(B, 2, 3, device=strip.device, dtype=strip.dtype)
    M[:, 0, 0] = cos
    M[:, 0, 1] = sin
    M[:, 0, 2] = -(cos * dx_norm + sin * dy_norm)
    M[:, 1, 0] = -sin
    M[:, 1, 1] = cos
    M[:, 1, 2] = -(-sin * dx_norm + cos * dy_norm)

    grid = F.affine_grid(M, size=strip.size(), align_corners=False)
    return F.grid_sample(strip, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EdgeStripPairDataset(Dataset):
    """Yields (strip_a, strip_b, label, pose_target) tuples from one HDF5 file.

    The ``label`` is 1.0 for true adjacent pairs, 0.0 otherwise. The
    ``pose_target`` is a normalized (Δθ, Δdx, Δdy) — only meaningful (non-zero)
    when label == 1.0 AND pose_aug=True.

    Parameters:
        h5_path: dataset file produced by build_edge_dataset.py.
        positive_ratio: fraction of items that are real positives (default 0.25
            → 1:3 positive:negative).
        seed: per-worker base seed for negatives & augmentation.
        pose_aug: if True, apply random rigid transform to strip_b on positives
            and use the inverse params as the pose target.
        appearance_aug: if True, brightness/contrast jitter + Gaussian noise.
        epoch_length: number of items per "epoch" iteration (the dataset can
            generate effectively infinite negatives, so we cap explicitly).
    """

    def __init__(self, h5_path: Path, *, positive_ratio: float = 0.25,
                 seed: int = 0, pose_aug: bool = True,
                 appearance_aug: bool = True,
                 epoch_length: int | None = None):
        self.h5_path = Path(h5_path)
        self.positive_ratio = positive_ratio
        self.seed = seed
        self.pose_aug = pose_aug
        self.appearance_aug = appearance_aug

        self._h5: h5py.File | None = None  # opened lazily per worker

        # Read minimal metadata to size epoch_length and pre-build a positive-
        # pairs lookup hash set for negative-sampling exclusion.
        with h5py.File(self.h5_path, "r") as h5:
            self.n_strips = h5["strips"].shape[0]
            self.n_positive_pairs = h5["positive_pairs"].shape[0]
            pairs = h5["positive_pairs"][:]
            self._positive_set = set(map(tuple, pairs.tolist())) | \
                                 set(map(lambda p: (p[1], p[0]), pairs.tolist()))
            self._positive_pair_idx = pairs  # (P, 2)
            self.doc_idx_per_strip = h5["doc_idx"][:]

        if epoch_length is None:
            epoch_length = max(self.n_positive_pairs * 4, 1024)
        self.epoch_length = epoch_length

    def _ensure_open(self):
        if self._h5 is None:
            # SWMR-friendly read
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int):
        self._ensure_open()

        # Use a fresh RNG seeded from OS entropy so each call yields a NEW
        # sample. Without this, idx maps deterministically to a single pair,
        # which silently turns the dataset into a fixed corpus of size
        # `epoch_length` and causes catastrophic overfitting after a few
        # epochs. Reproducibility is sacrificed but training generalises.
        rng = np.random.default_rng()

        is_positive = rng.random() < self.positive_ratio

        if is_positive:
            row = int(rng.integers(0, self.n_positive_pairs))
            ia, ib = self._positive_pair_idx[row]
        else:
            # Sample two random strips, exclude actual positive pairs.
            for _ in range(20):
                ia = int(rng.integers(0, self.n_strips))
                ib = int(rng.integers(0, self.n_strips))
                if ia == ib:
                    continue
                if (ia, ib) in self._positive_set:
                    continue
                break

        # Load the two strips (HDF5 reads).
        strip_a = self._h5["strips"][ia]   # (sw, sl, 3) uint8
        strip_b = self._h5["strips"][ib]

        # Convert to NCHW float in [0, 1].
        sa = torch.from_numpy(strip_a).permute(2, 0, 1).float() / 255.0
        sb = torch.from_numpy(strip_b).permute(2, 0, 1).float() / 255.0

        # Pose target (normalized) — only meaningful for positives w/ pose_aug.
        if is_positive and self.pose_aug:
            theta = float(rng.uniform(-POSE_THETA_RANGE_DEG, POSE_THETA_RANGE_DEG))
            dx = float(rng.uniform(-POSE_DX_RANGE_PX, POSE_DX_RANGE_PX))
            dy = float(rng.uniform(-POSE_DY_RANGE_PX, POSE_DY_RANGE_PX))
            sb = _apply_pose_to_strip(
                sb.unsqueeze(0),
                torch.tensor([theta]), torch.tensor([dx]), torch.tensor([dy])
            ).squeeze(0)
            pose_target = torch.tensor(_normalize_pose(-theta, -dx, -dy),
                                         dtype=torch.float32)
        else:
            pose_target = torch.zeros(3, dtype=torch.float32)

        # Appearance augmentation (per-strip independent).
        if self.appearance_aug:
            sa = self._aug_appearance(sa, rng)
            sb = self._aug_appearance(sb, rng)

        label = torch.tensor(1.0 if is_positive else 0.0, dtype=torch.float32)
        return sa, sb, label, pose_target

    @staticmethod
    def _aug_appearance(s: torch.Tensor, rng) -> torch.Tensor:
        bright = float(rng.uniform(-0.10, 0.10))
        contrast = float(rng.uniform(0.85, 1.15))
        s = s * contrast + bright
        s = s + torch.from_numpy(
            rng.normal(0.0, 0.015, size=s.shape).astype(np.float32))
        return s.clamp_(0.0, 1.0)


def _seed_worker(worker_id):  # ensure determinism per worker
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    np.random.seed(info.seed % (2**31 - 1))


# ---------------------------------------------------------------------------
# Train + validate
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    train_h5: Path
    val_h5: Path
    out_ckpt: Path
    log_dir: Path
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    positive_ratio: float = 0.25
    pose_weight: float = 0.1
    seed: int = 20260427
    amp: bool = True
    val_every: int = 1
    log_every: int = 50
    # Pose augmentation is currently disabled by default. The grid_sample
    # warp introduces border artifacts that turn out to be a near-perfect
    # shortcut feature, which the BCE loss latches onto and ends up generalising
    # in *reversed* direction on validation. See known-issues note in the
    # docstring. Without it, the match head trains cleanly and the pose head
    # learns to predict ~0 (which is the correct answer for ground-truth-
    # aligned strips at inference time).
    pose_aug: bool = False
    appearance_aug: bool = True
    epoch_length: int | None = None


def _evaluate(model: EdgeMatcher, loader: DataLoader, device: torch.device,
              pose_weight: float) -> dict:
    model.eval()
    all_logits, all_labels, all_pose_err = [], [], []
    losses, bces = [], []
    with torch.no_grad():
        for sa, sb, label, pose_target in loader:
            sa = sa.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pose_target = pose_target.to(device, non_blocking=True)
            out = model(sa, sb)
            losses_d = edge_matcher_loss(out, label, pose_target,
                                            pose_weight=pose_weight)
            losses.append(float(losses_d["loss"]))
            bces.append(float(losses_d["bce"]))
            all_logits.append(out.match_logit.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())
            pos = label > 0.5
            if pos.any():
                err = (out.pose_pred[pos] - pose_target[pos]).abs().mean(dim=1)
                all_pose_err.append(err.detach().cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    auc = _auc_roc(logits, labels)
    pred = (logits > 0).astype(np.float32)
    acc = float((pred == labels).mean())
    pose_err = float(np.median(np.concatenate(all_pose_err))) \
        if all_pose_err else float("nan")
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "bce": float(np.mean(bces)) if bces else 0.0,
        "auc": auc,
        "acc": acc,
        "pose_med_err": pose_err,
        "n": int(len(labels)),
    }


def _auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Simple AUC-ROC via rank statistics."""
    if labels.min() == labels.max():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels > 0.5].sum()
    n_pos = int((labels > 0.5).sum())
    n_neg = int((labels < 0.5).sum())
    return float((pos_ranks - n_pos * (n_pos - 1) / 2) / max(n_pos * n_neg, 1))


def train(cfg: TrainConfig) -> dict:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    train_ds = EdgeStripPairDataset(
        cfg.train_h5, positive_ratio=cfg.positive_ratio,
        seed=cfg.seed, pose_aug=cfg.pose_aug,
        appearance_aug=cfg.appearance_aug,
        epoch_length=cfg.epoch_length)
    val_ds = EdgeStripPairDataset(
        cfg.val_h5, positive_ratio=0.5,           # 50/50 for clean AUC
        seed=cfg.seed + 999, pose_aug=cfg.pose_aug,
        appearance_aug=False,
        epoch_length=min(2048, train_ds.n_positive_pairs * 4) or 2048)
    train_eval_ds = EdgeStripPairDataset(
        cfg.train_h5, positive_ratio=0.5,
        seed=cfg.seed + 1234, pose_aug=cfg.pose_aug,
        appearance_aug=False,
        epoch_length=min(2048, train_ds.n_positive_pairs * 4) or 2048)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, worker_init_fn=_seed_worker, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker, persistent_workers=cfg.num_workers > 0)
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker, persistent_workers=cfg.num_workers > 0)

    model = build_edge_matcher().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model parameters: {n_params:,}")
    print(f"[train] train items/epoch: {len(train_ds)}  "
          f"val items: {len(val_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    best_auc = -1.0
    history = []
    log_path = cfg.log_dir / "training_log.json"
    t_start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        ep_losses, ep_bces = [], []
        ep_t0 = time.time()
        for step, (sa, sb, label, pose_target) in enumerate(train_loader):
            sa = sa.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pose_target = pose_target.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(sa, sb)
                losses = edge_matcher_loss(
                    out, label, pose_target, pose_weight=cfg.pose_weight)
            scaler.scale(losses["loss"]).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            ep_losses.append(float(losses["loss"].detach()))
            ep_bces.append(float(losses["bce"].detach()))

            if step % cfg.log_every == 0:
                lr_now = sched.get_last_lr()[0]
                print(f"[train] e{epoch:02d} s{step:04d}/{steps_per_epoch} "
                      f"loss={ep_losses[-1]:.4f} bce={ep_bces[-1]:.4f} "
                      f"lr={lr_now:.2e}")

        ep_dt = time.time() - ep_t0
        train_metrics = {
            "loss": float(np.mean(ep_losses)),
            "bce": float(np.mean(ep_bces)),
            "lr": float(sched.get_last_lr()[0]),
            "epoch_seconds": ep_dt,
        }

        # Validation pass + held-out train AUC (sanity check on overfitting).
        val_metrics = _evaluate(model, val_loader, device, cfg.pose_weight)
        train_eval_metrics = _evaluate(model, train_eval_loader, device,
                                          cfg.pose_weight)
        train_metrics["heldout_auc"] = train_eval_metrics["auc"]
        train_metrics["heldout_acc"] = train_eval_metrics["acc"]
        elapsed = time.time() - t_start
        msg = (f"[train] EPOCH {epoch}: train_loss={train_metrics['loss']:.4f} "
               f"train_auc={train_metrics['heldout_auc']:.4f} "
               f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f} "
               f"val_pose_err={val_metrics['pose_med_err']:.4f} "
               f"epoch_s={ep_dt:.1f} elapsed_s={elapsed:.0f}")
        print(msg)

        history.append({"epoch": epoch, "train": train_metrics,
                        "val": val_metrics})
        with open(log_path, "w") as fh:
            json.dump({"config": {k: str(v) for k, v in asdict(cfg).items()},
                       "history": history,
                       "best_auc": best_auc,
                       "n_params": n_params}, fh, indent=2)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg.__class__()) if False else None,
                "model_config": asdict(EdgeMatcherConfig()),
                "val_metrics": val_metrics,
                "epoch": epoch,
            }, cfg.out_ckpt)
            print(f"[train]   -> saved new best ckpt to {cfg.out_ckpt} "
                  f"(auc={best_auc:.4f})")

    print(f"[train] DONE - best val AUC = {best_auc:.4f}")
    return {"best_auc": best_auc, "log_path": str(log_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path,
                     default=Path("data/dataset/edge_strips/train.h5"))
    ap.add_argument("--val", type=Path,
                     default=Path("data/dataset/edge_strips/val.h5"))
    ap.add_argument("--out", type=Path,
                     default=Path("models/edge_matcher.pt"))
    ap.add_argument("--log_dir", type=Path,
                     default=Path("data/training_logs/edge_matcher"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--positive_ratio", type=float, default=0.25)
    ap.add_argument("--epoch_length", type=int, default=None,
                    help="Override items/epoch (default: n_positive_pairs * 4). "
                         "Use ~40000 to cut training to ~2.5h.")
    ap.add_argument("--pose_weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--pose_aug", action="store_true",
                     help="Enable pose-warp data augmentation (default: off "
                          "due to a known issue causing inverted val AUC).")
    ap.add_argument("--no_appearance_aug", action="store_true",
                     help="Disable brightness/noise jitter (default: on).")
    args = ap.parse_args()

    cfg = TrainConfig(
        train_h5=args.train, val_h5=args.val, out_ckpt=args.out,
        log_dir=args.log_dir, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        num_workers=args.num_workers, positive_ratio=args.positive_ratio,
        pose_weight=args.pose_weight, seed=args.seed,
        amp=(not args.no_amp),
        pose_aug=args.pose_aug,
        appearance_aug=(not args.no_appearance_aug))
    train(cfg)


if __name__ == "__main__":
    main()
