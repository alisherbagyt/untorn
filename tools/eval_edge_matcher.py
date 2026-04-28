"""Step 8 of Phase 2 — evaluate the trained Siamese edge matcher.

Loads ``models/edge_matcher.pt`` and runs it on ``val.h5``, then compares
against several deterministic baselines that operate on the same stored
edge-strip pairs:

    * pixel_ncc       — flat NCC of raw RGB strip pixels
    * row_corr        — mean per-row Pearson correlation
    * lab_mean_l1     — L1 distance of per-strip mean Lab colour
    * sw_curvature    — Smith-Waterman on curvature strings derived from the
                         stored boundary curves (closest analogue of the
                         existing pipeline's matching gate)

Note: the *full* SW+ICP pipeline gate also includes Procrustes alignment, ICP
refinement, and the SDT physical-feasibility check. Those gates require
fragment-level context (full masks, contours, signed-distance fields) that
this per-pair evaluation can't reproduce. The fair end-to-end comparison
between Siamese and the full SW+ICP pipeline is reserved for Phase 5
(``benchmark_synthetic.py``). Here we benchmark only the curvature-matching
component, which is the most direct analogue of the Siamese match score.

Metrics reported per scorer:
    AUC-ROC
    accuracy @ 0.5 threshold (where applicable)
    precision @ recall=0.9
    F1 max
    median pose-regression L1 error (Siamese only, on positives)

CLI:
    python tools/eval_edge_matcher.py
        --val   data/dataset/edge_strips/val.h5
        --ckpt  models/edge_matcher.pt
        --out   data/eval_results/edge_matcher_eval.json
        --max_pairs 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from untorn.edge_matcher_model import build_edge_matcher  # noqa: E402

# Pull SW + curvature helpers from the existing pipeline
try:
    from untorn import contours as _contours_mod
    from untorn.matching import smith_waterman_real
    HAVE_SW = True
except Exception as exc:
    HAVE_SW = False
    SW_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Pair loading: build (strip_a, strip_b, label) at index level
# ---------------------------------------------------------------------------

def _load_split(h5_path: Path) -> dict:
    h5 = h5py.File(h5_path, "r")
    return {
        "h5": h5,
        "strips": h5["strips"],
        "doc_idx": h5["doc_idx"][:],
        "fragment_idx": h5["fragment_idx"][:],
        "partner_idx": h5["partner_fragment_idx"][:],
        "boundary_arc_length": h5["boundary_arc_length"][:],
        "positive_pairs": h5["positive_pairs"][:],
        "doc_ids": [s.decode("utf-8") if isinstance(s, bytes) else s
                    for s in h5["doc_ids"][:]],
        "n_strips": h5["strips"].shape[0],
    }


def _build_eval_pairs(split: dict, *, max_pairs: int, rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (pair_indices [N,2], labels [N]). 50/50 split between true
    positives and randomly mined negatives that are NOT positives."""
    n_strips = split["n_strips"]
    pos_pairs = split["positive_pairs"]
    n_pos_total = len(pos_pairs)
    if n_pos_total == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32)

    n_each = min(max_pairs // 2, n_pos_total)
    pos_subset = pos_pairs[rng.choice(n_pos_total, size=n_each, replace=False)]

    pos_set = set(map(tuple, pos_pairs.tolist()))
    pos_set |= set((b, a) for a, b in pos_pairs.tolist())

    neg_pairs = []
    while len(neg_pairs) < n_each:
        ia = int(rng.integers(0, n_strips))
        ib = int(rng.integers(0, n_strips))
        if ia == ib or (ia, ib) in pos_set:
            continue
        neg_pairs.append((ia, ib))
    neg_subset = np.array(neg_pairs, dtype=np.int32)

    pairs = np.vstack([pos_subset, neg_subset]).astype(np.int32)
    labels = np.concatenate([
        np.ones(len(pos_subset), dtype=np.int32),
        np.zeros(len(neg_subset), dtype=np.int32),
    ])
    return pairs, labels


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.min() == labels.max():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels > 0.5].sum()
    n_pos = int((labels > 0.5).sum())
    n_neg = int((labels < 0.5).sum())
    return float((pos_ranks - n_pos * (n_pos - 1) / 2) / max(n_pos * n_neg, 1))


def _precision_at_recall(scores: np.ndarray, labels: np.ndarray,
                          recall: float = 0.9) -> tuple[float, float]:
    """Return (precision, threshold). Higher score = more positive."""
    order = np.argsort(-scores)
    sorted_lab = labels[order]
    sorted_sc = scores[order]
    tp = np.cumsum(sorted_lab == 1)
    fp = np.cumsum(sorted_lab == 0)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan"), float("nan")
    rec = tp / max(n_pos, 1)
    prec = tp / np.maximum(tp + fp, 1)
    ok = rec >= recall
    if not ok.any():
        return float("nan"), float("nan")
    idx = int(np.argmax(ok))   # first index where recall >= target
    return float(prec[idx]), float(sorted_sc[idx])


def _f1_max(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores)
    sorted_lab = labels[order]
    tp = np.cumsum(sorted_lab == 1)
    fp = np.cumsum(sorted_lab == 0)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan"), float("nan")
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(n_pos, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(scores[order][idx])


def _accuracy(scores: np.ndarray, labels: np.ndarray, thresh: float = 0.5
              ) -> float:
    return float(((scores > thresh).astype(np.int32) == labels).mean())


def _summarize(name: str, scores: np.ndarray, labels: np.ndarray,
               *, threshold_for_acc: float | None = None) -> dict:
    auc = _auc_roc(scores, labels)
    prec_r90, thr = _precision_at_recall(scores, labels, 0.9)
    f1m, f1_thr = _f1_max(scores, labels)
    out = {
        "scorer": name,
        "n": int(len(labels)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "auc": auc,
        "precision_at_recall_90": prec_r90,
        "threshold_at_recall_90": thr,
        "f1_max": f1m,
        "f1_threshold": f1_thr,
    }
    if threshold_for_acc is not None:
        out["accuracy_at_default_threshold"] = _accuracy(
            scores, labels, threshold_for_acc)
    return out


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def _siamese_scores(model, pairs: np.ndarray, split: dict,
                     device: torch.device, batch_size: int = 256
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (match_probs, pose_l1_per_pair). pose_l1 is NaN for negatives
    by default (we have no GT augmentation here at eval — the strips ARE the
    aligned pair, so a perfectly-trained pose head would predict ~0)."""
    strips = split["strips"]
    probs = np.empty(len(pairs), dtype=np.float32)
    pose_l1 = np.full(len(pairs), np.nan, dtype=np.float32)
    model.eval()

    def gather_strips(indices: np.ndarray) -> np.ndarray:
        # h5py requires strictly increasing indices; use unique + inverse.
        uniq, inv = np.unique(indices, return_inverse=True)
        return np.asarray(strips[uniq])[inv]

    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            sa = gather_strips(batch_pairs[:, 0])
            sb = gather_strips(batch_pairs[:, 1])

            ta = torch.from_numpy(sa).permute(0, 3, 1, 2).float().to(device) / 255.0
            tb = torch.from_numpy(sb).permute(0, 3, 1, 2).float().to(device) / 255.0
            out = model(ta, tb)
            probs[i:i + batch_size] = out.match_prob.cpu().numpy()
            pose_l1[i:i + batch_size] = out.pose_pred.abs().mean(dim=1).cpu().numpy()
    return probs, pose_l1


def _pixel_ncc_scores(pairs: np.ndarray, split: dict) -> np.ndarray:
    """Flat normalised cross-correlation between strip_a and strip_b."""
    strips = split["strips"]
    out = np.empty(len(pairs), dtype=np.float32)
    for k, (ia, ib) in enumerate(pairs):
        a = strips[ia].astype(np.float32).ravel()
        b = strips[ib].astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        out[k] = float(a @ b / denom) if denom > 1e-9 else 0.0
    return out


def _row_correlation_scores(pairs: np.ndarray, split: dict) -> np.ndarray:
    """Mean per-row Pearson correlation across the 32 strip rows."""
    strips = split["strips"]
    out = np.empty(len(pairs), dtype=np.float32)
    for k, (ia, ib) in enumerate(pairs):
        a = strips[ia].astype(np.float32).reshape(strips.shape[1], -1)
        b = strips[ib].astype(np.float32).reshape(strips.shape[1], -1)
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        num = (a * b).sum(axis=1)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        denom = np.maximum(denom, 1e-9)
        out[k] = float((num / denom).mean())
    return out


def _lab_mean_l1_neg(pairs: np.ndarray, split: dict) -> np.ndarray:
    """Negative L1 distance of per-strip mean Lab colour. Negated so that
    higher = more similar (matching the convention in other scorers)."""
    import cv2
    strips = split["strips"]
    out = np.empty(len(pairs), dtype=np.float32)
    for k, (ia, ib) in enumerate(pairs):
        la = cv2.cvtColor(strips[ia], cv2.COLOR_RGB2LAB).astype(np.float32)
        lb = cv2.cvtColor(strips[ib], cv2.COLOR_RGB2LAB).astype(np.float32)
        out[k] = -float(np.abs(la.mean(axis=(0, 1)) - lb.mean(axis=(0, 1))).mean())
    return out


def _sw_curvature_scores(pairs: np.ndarray, split: dict) -> np.ndarray | None:
    """SW alignment on curvature strings derived from boundary geometry.

    Each strip's boundary curve is reconstructed from row 0 of the strip — but
    that's just sampled pixels, not coordinates. We don't store curve_xy in
    HDF5 (that would inflate file size); instead, we derive a 1D shape
    descriptor from the strip's row-0 grayscale gradient pattern and run SW on
    that. This is a *proxy* for the curvature-matching gate, not an exact
    reproduction of it.
    """
    if not HAVE_SW:
        print(f"[eval] SW unavailable: {SW_IMPORT_ERROR}")
        return None
    import cv2
    strips = split["strips"]
    out = np.empty(len(pairs), dtype=np.float32)

    def boundary_signature(strip: np.ndarray) -> np.ndarray:
        # Row 0 sits ON the torn edge. Use its column-wise grayscale gradient
        # as a 1D shape descriptor.
        gray = cv2.cvtColor(strip[0:1], cv2.COLOR_RGB2GRAY).ravel().astype(np.float32)
        sig = np.diff(gray)
        # Smooth slightly for noise robustness.
        kernel = np.ones(5, dtype=np.float32) / 5
        return np.convolve(sig, kernel, mode="same")

    for k, (ia, ib) in enumerate(pairs):
        sig_a = boundary_signature(strips[ia])
        # For B, the boundary is traversed in the opposite direction in the
        # canvas frame compared to A — flipping yields the matching shape.
        sig_b = boundary_signature(strips[ib])[::-1]
        score, _, _ = smith_waterman_real(sig_a, sig_b)
        out[k] = float(score)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path,
                     default=Path("data/dataset/edge_strips/val.h5"))
    ap.add_argument("--ckpt", type=Path,
                     default=Path("models/edge_matcher.pt"))
    ap.add_argument("--out", type=Path,
                     default=Path("data/eval_results/edge_matcher_eval.json"))
    ap.add_argument("--max_pairs", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260427)
    ap.add_argument("--no_sw", action="store_true",
                     help="Skip SW curvature baseline (it's the slowest).")
    args = ap.parse_args()

    if not args.val.exists():
        raise SystemExit(f"val.h5 not found: {args.val}")
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    rng = np.random.default_rng(args.seed)
    print(f"[eval] loading {args.val}")
    split = _load_split(args.val)
    print(f"[eval] strips={split['n_strips']}  "
          f"positive_pairs={len(split['positive_pairs'])}")

    pairs, labels = _build_eval_pairs(split, max_pairs=args.max_pairs, rng=rng)
    print(f"[eval] eval pairs: {len(pairs)} (pos={(labels==1).sum()}, "
          f"neg={(labels==0).sum()})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")
    print(f"[eval] loading checkpoint {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_edge_matcher().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[eval] checkpoint epoch={ckpt.get('epoch')}  "
          f"val_metrics={ckpt.get('val_metrics')}")

    summaries: list[dict] = []
    pose_summary: dict | None = None
    timings: dict = {}

    # 1. Siamese
    t0 = time.time()
    s_probs, s_pose_l1 = _siamese_scores(model, pairs, split, device)
    timings["siamese_seconds"] = time.time() - t0
    summaries.append(_summarize("siamese", s_probs, labels,
                                 threshold_for_acc=0.5))
    pos_mask = labels == 1
    if pos_mask.any():
        pose_summary = {
            "median_pose_l1": float(np.median(s_pose_l1[pos_mask])),
            "mean_pose_l1": float(np.mean(s_pose_l1[pos_mask])),
            "p90_pose_l1": float(np.percentile(s_pose_l1[pos_mask], 90)),
        }

    # 2. pixel NCC baseline
    t0 = time.time()
    p_ncc = _pixel_ncc_scores(pairs, split)
    timings["pixel_ncc_seconds"] = time.time() - t0
    summaries.append(_summarize("pixel_ncc", p_ncc, labels))

    # 3. row-wise correlation baseline
    t0 = time.time()
    rc = _row_correlation_scores(pairs, split)
    timings["row_corr_seconds"] = time.time() - t0
    summaries.append(_summarize("row_corr", rc, labels))

    # 4. Lab mean colour baseline
    t0 = time.time()
    lab = _lab_mean_l1_neg(pairs, split)
    timings["lab_mean_l1_seconds"] = time.time() - t0
    summaries.append(_summarize("lab_mean_l1", lab, labels))

    # 5. SW curvature proxy
    if not args.no_sw:
        t0 = time.time()
        sw = _sw_curvature_scores(pairs, split)
        timings["sw_curvature_seconds"] = time.time() - t0
        if sw is not None:
            summaries.append(_summarize("sw_curvature", sw, labels))

    # Print summary table
    print()
    print(f"{'scorer':<16} {'AUC':>7} {'P@R=0.9':>9} {'F1max':>7}")
    print("-" * 42)
    for s in summaries:
        print(f"{s['scorer']:<16} {s['auc']:>7.4f} "
              f"{s['precision_at_recall_90']:>9.4f} "
              f"{s['f1_max']:>7.4f}")
    if pose_summary:
        print(f"\nSiamese pose head L1 (positives):  median={pose_summary['median_pose_l1']:.4f}"
              f"  mean={pose_summary['mean_pose_l1']:.4f}"
              f"  p90={pose_summary['p90_pose_l1']:.4f}")

    # Save full results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "checkpoint": str(args.ckpt),
            "val_h5": str(args.val),
            "n_eval_pairs": int(len(pairs)),
            "summaries": summaries,
            "pose_summary": pose_summary,
            "timings_seconds": timings,
            "checkpoint_epoch": ckpt.get("epoch"),
        }, fh, indent=2)
    print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
