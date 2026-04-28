"""Step 5 of Phase 2 — edge-strip training dataset builder.

For every adjacent fragment pair recorded in the synthetic dataset, this
extracts two 32×256 RGB strips from the *composite* image:
    * strip A — sampled along the shared torn boundary, oriented inward into
      fragment A (the 32-pixel axis goes 0→inside-A; the 256-pixel axis runs
      along the boundary in arc-length order)
    * strip B — same boundary, opposite side, going inward into fragment B

Positive training pairs are built as (strip_a, strip_b). Negative pairs are
NOT pre-stored — they're sampled at training time by picking strips whose
``partner_strip_index`` is different (i.e. they don't share a true boundary).

The strips are sampled directly from the composite at the canvas-frame
location of the boundary, using bilinear interpolation. Sampling is
chirality-aware: column k of strip A and column k of strip B correspond to
the SAME physical point on the original-frame boundary curve (so the model
can learn ink/fiber continuity column-by-column).

HDF5 schema (one file per split — train.h5 / val.h5):
    strips                : (N, 32, 256, 3) uint8
    doc_idx               : (N,)            int32        index into doc_ids
    fragment_idx          : (N,)            int32        within-doc fragment
    partner_fragment_idx  : (N,)            int32
    boundary_arc_length   : (N,)            float32      original-frame px
    positive_pairs        : (P, 2)          int32        (strip_a, strip_b) pairs
    doc_ids               : (D,)            S<L>         per-doc identifier strings

CLI:
    python tools/build_edge_dataset.py
        --index data/dataset/index.json
        --out_dir data/dataset/edge_strips
        --strip_w 32 --strip_l 256
        --max_docs 50      # optional, for smoke testing

Per-fragment-pair the script may emit zero strips if the shared boundary is
too short, or one positive pair (and the two strips that form it) otherwise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_STRIP_WIDTH = 32        # pixels perpendicular (into fragment)
DEFAULT_STRIP_LENGTH = 256      # samples along the boundary curve
DEFAULT_MIN_BOUNDARY_PX = 64    # skip pairs whose shared boundary is shorter
DEFAULT_BOUNDARY_PROXIMITY_PX = 1.5
DEFAULT_DTYPE = np.uint8


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_full_mask(image_shape_hw: tuple[int, int],
                      bbox: tuple[int, int, int, int],
                      crop_alpha: np.ndarray) -> np.ndarray:
    """Place a (bh, bw) crop alpha mask back into a full-image binary mask."""
    H, W = image_shape_hw
    bx, by, bw, bh = bbox
    out = np.zeros((H, W), dtype=np.uint8)
    out[by:by + bh, bx:bx + bw] = (crop_alpha > 0).astype(np.uint8)
    return out


def _ordered_boundary_points(mask_a: np.ndarray, mask_b: np.ndarray,
                              proximity_px: float) -> np.ndarray | None:
    """Return the longest contiguous boundary curve between A and B, ordered
    along A's contour direction. Returns (M, 2) float32 array of (x, y) points
    in the original-image frame, or None if no usable boundary exists."""
    contours_a, _ = cv2.findContours(mask_a, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    if not contours_a:
        return None
    contour = max(contours_a, key=cv2.contourArea).reshape(-1, 2)
    if contour.shape[0] < 4:
        return None

    # Distance transform of B-complement: dt(x) = distance to nearest B-pixel.
    inv_b = (1 - mask_b).astype(np.uint8)
    dt = cv2.distanceTransform(inv_b, cv2.DIST_L2, 3)

    # Mark contour-A points whose nearest B-pixel is within proximity_px.
    near_b = dt[contour[:, 1], contour[:, 0]] <= proximity_px

    if not near_b.any():
        return None

    # Contour is a cyclic sequence — find the longest contiguous run of True.
    n = len(near_b)
    best_start, best_len = 0, 0
    cur_start, cur_len = -1, 0
    # Roll once to handle wrap-around: scan twice through the array.
    extended = np.concatenate([near_b, near_b])
    cur_start = -1
    cur_len = 0
    for i in range(2 * n):
        if extended[i]:
            if cur_start < 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_start = -1
            cur_len = 0
    if best_len < 4:
        return None
    best_len = min(best_len, n)  # don't double-count
    indices = [(best_start + k) % n for k in range(best_len)]
    return contour[indices].astype(np.float32)


def _resample_curve(points: np.ndarray, n_samples: int
                     ) -> tuple[np.ndarray, float]:
    """Resample a polyline to n equally-spaced samples by arc length.

    Returns (resampled (n,2), total_arc_length)."""
    diffs = np.diff(points, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    if seg_len.sum() <= 0:
        return None, 0.0
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    targets = np.linspace(0.0, total, n_samples)
    out = np.empty((n_samples, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(targets):
        while j < len(seg_len) - 1 and cum[j + 1] < t:
            j += 1
        if seg_len[j] > 1e-9:
            alpha = (t - cum[j]) / seg_len[j]
        else:
            alpha = 0.0
        out[i] = points[j] + alpha * (points[j + 1] - points[j])
    return out, total


def _outward_normals(curve_orig: np.ndarray, mask_a: np.ndarray,
                      mask_b: np.ndarray, lookahead_px: float = 4.0
                      ) -> np.ndarray:
    """Per-sample outward normal (unit) pointing AWAY from A toward B.

    Heuristic: for each curve point, take a small step in each candidate
    perpendicular direction; the one whose endpoint lies in B's mask wins.
    """
    H, W = mask_a.shape
    diffs = np.diff(curve_orig, axis=0, append=curve_orig[-1:])
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    tangents = diffs / norms
    # Rotate tangent +90° → candidate normal.
    n1 = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # Probe candidate +n1 vs -n1.
    probe_pos = curve_orig + lookahead_px * n1
    probe_neg = curve_orig - lookahead_px * n1
    pp = np.clip(probe_pos.round().astype(int),
                  [0, 0], [W - 1, H - 1])
    pn = np.clip(probe_neg.round().astype(int),
                  [0, 0], [W - 1, H - 1])
    pos_in_b = mask_b[pp[:, 1], pp[:, 0]] > 0
    neg_in_b = mask_b[pn[:, 1], pn[:, 0]] > 0

    out = n1.copy()
    flip = (~pos_in_b) & neg_in_b
    out[flip] = -n1[flip]
    # If neither side is in B (rare), keep n1 as-is — the model will learn it.
    return out.astype(np.float32)


def _sample_strip(image: np.ndarray,
                   curve_canvas: np.ndarray,
                   inward_dir_canvas: np.ndarray,
                   strip_width: int) -> np.ndarray:
    """Bilinear-sample a strip from `image` along the curve.

    Args:
        image: (H, W, 3) uint8 RGB.
        curve_canvas: (256, 2) sample positions (x, y) in canvas frame.
        inward_dir_canvas: (256, 2) unit vectors pointing into the fragment.
        strip_width: number of perpendicular pixels to sample (≥ 1).

    Returns: (strip_width, 256, 3) uint8 strip. Row 0 sits ON the boundary;
    row strip_width - 1 is the deepest interior sample.
    """
    n_samples = curve_canvas.shape[0]
    # Build a (strip_width, n_samples, 2) coordinate grid.
    steps = np.arange(strip_width, dtype=np.float32)[:, None, None]   # (W,1,1)
    base = curve_canvas[None, :, :]                                    # (1,N,2)
    direction = inward_dir_canvas[None, :, :]                          # (1,N,2)
    coords = base + steps * direction                                  # (W,N,2)

    map_x = coords[..., 0].astype(np.float32)
    map_y = coords[..., 1].astype(np.float32)
    out = cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(0, 0, 0))
    return out  # (strip_width, n_samples, 3) uint8


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

@dataclass
class StripExtraction:
    strip: np.ndarray              # (sw, sl, 3) uint8
    fragment_idx: int
    partner_fragment_idx: int
    boundary_arc_len_px: float


def _process_pair(composite: np.ndarray, image_shape_hw: tuple[int, int],
                   ga: dict, gb: dict, ca: dict, cb: dict, *,
                   strip_w: int, strip_l: int,
                   min_boundary_px: float,
                   proximity_px: float
                   ) -> tuple[StripExtraction, StripExtraction] | None:
    """Process one adjacent pair (A, B). Returns (strip_a_record, strip_b_record)
    or None if the pair was skipped."""
    bbox_a = tuple(ga["bbox_in_original"])
    bbox_b = tuple(gb["bbox_in_original"])

    # Re-derive masks from the cropped RGBA alpha channel.
    rgba_a = np.asarray(Image.open(ca["rgba_full_path"]).convert("RGBA"))
    rgba_b = np.asarray(Image.open(cb["rgba_full_path"]).convert("RGBA"))
    mask_a = _build_full_mask(image_shape_hw, bbox_a, rgba_a[..., 3])
    mask_b = _build_full_mask(image_shape_hw, bbox_b, rgba_b[..., 3])

    curve = _ordered_boundary_points(mask_a, mask_b, proximity_px)
    if curve is None:
        return None
    resampled, arc_len = _resample_curve(curve, strip_l)
    if resampled is None or arc_len < min_boundary_px:
        return None

    out_normal = _outward_normals(resampled, mask_a, mask_b)

    # Place transforms map crop coords → canvas coords.
    M_orig_to_canvas_a = np.linalg.inv(np.array(ca["M_to_original"], dtype=np.float64))
    M_orig_to_canvas_b = np.linalg.inv(np.array(cb["M_to_original"], dtype=np.float64))

    # Map curve points + tangent directions from original → canvas frame.
    pts_h = np.concatenate([resampled, np.ones((resampled.shape[0], 1),
                                                  dtype=np.float32)], axis=1).T  # (3,N)
    canvas_a = (M_orig_to_canvas_a @ pts_h)[:2].T.astype(np.float32)
    canvas_b = (M_orig_to_canvas_b @ pts_h)[:2].T.astype(np.float32)

    # Inward direction in original frame: A's inward = -outward, B's = +outward.
    inward_a_orig = -out_normal
    inward_b_orig = +out_normal

    # Rotate (only the rotation part of the affine, no translation).
    Ra = M_orig_to_canvas_a[:2, :2]
    Rb = M_orig_to_canvas_b[:2, :2]
    inward_a_canvas = (Ra @ inward_a_orig.T).T.astype(np.float32)
    inward_b_canvas = (Rb @ inward_b_orig.T).T.astype(np.float32)
    # Re-normalize (place transforms are pure rigid → already unit, but
    # numerical drift can creep in).
    for arr in (inward_a_canvas, inward_b_canvas):
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        arr /= norms

    strip_a = _sample_strip(composite, canvas_a, inward_a_canvas, strip_w)
    strip_b = _sample_strip(composite, canvas_b, inward_b_canvas, strip_w)

    rec_a = StripExtraction(
        strip=strip_a,
        fragment_idx=int(ga["index"]),
        partner_fragment_idx=int(gb["index"]),
        boundary_arc_len_px=float(arc_len),
    )
    rec_b = StripExtraction(
        strip=strip_b,
        fragment_idx=int(gb["index"]),
        partner_fragment_idx=int(ga["index"]),
        boundary_arc_len_px=float(arc_len),
    )
    return rec_a, rec_b


def _iter_doc_pairs(doc_dir: Path, gt: dict, cmp_meta: dict) -> Iterable[
        tuple[dict, dict, dict, dict]]:
    """Yield (gt_a, gt_b, cmp_a, cmp_b) for every placed adjacency a < b."""
    placements = {p["fragment_index"]: p for p in cmp_meta["placements"]}

    # Resolve fragment paths once, attach to placement dict.
    for p in placements.values():
        p["rgba_full_path"] = doc_dir / p["rgba_source"]

    placed = set(placements.keys())
    fragments_by_idx = {f["index"]: f for f in gt["fragments"]}

    for ga in gt["fragments"]:
        idx_a = ga["index"]
        if idx_a not in placed:
            continue
        for idx_b in ga["neighbors"]:
            if idx_b <= idx_a:
                continue
            if idx_b not in placed:
                continue
            gb = fragments_by_idx[idx_b]
            yield ga, gb, placements[idx_a], placements[idx_b]


# ---------------------------------------------------------------------------
# Build one split into HDF5
# ---------------------------------------------------------------------------

def build_split(index_entries: list[dict], synthetic_root: Path,
                 out_path: Path, *, strip_w: int, strip_l: int,
                 min_boundary_px: float, proximity_px: float,
                 max_docs: int | None) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if max_docs:
        index_entries = index_entries[:max_docs]

    n_strips_so_far = 0
    n_pairs_so_far = 0
    skipped_short_boundary = 0
    skipped_no_boundary = 0
    failed_docs = 0

    # Use HDF5 with pre-extended chunked datasets.
    chunk_n = 1024
    with h5py.File(out_path, "w") as h5:
        strips_ds = h5.create_dataset(
            "strips", shape=(0, strip_w, strip_l, 3), maxshape=(None, strip_w, strip_l, 3),
            dtype=DEFAULT_DTYPE, chunks=(min(chunk_n, 64), strip_w, strip_l, 3),
            compression="gzip", compression_opts=4,
        )
        doc_idx_ds = h5.create_dataset(
            "doc_idx", shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(chunk_n,))
        frag_idx_ds = h5.create_dataset(
            "fragment_idx", shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(chunk_n,))
        partner_idx_ds = h5.create_dataset(
            "partner_fragment_idx", shape=(0,), maxshape=(None,),
            dtype=np.int32, chunks=(chunk_n,))
        arc_len_ds = h5.create_dataset(
            "boundary_arc_length", shape=(0,), maxshape=(None,),
            dtype=np.float32, chunks=(chunk_n,))
        pairs_ds = h5.create_dataset(
            "positive_pairs", shape=(0, 2), maxshape=(None, 2),
            dtype=np.int32, chunks=(min(chunk_n, 1024), 2))

        doc_ids: list[str] = []
        for di, entry in enumerate(index_entries):
            doc_id = entry["doc_id"]
            doc_dir = synthetic_root / doc_id
            try:
                with open(doc_dir / "ground_truth.json") as fh:
                    gt = json.load(fh)
                with open(doc_dir / "composite_transforms.json") as fh:
                    cmp_meta = json.load(fh)
                composite = np.asarray(Image.open(doc_dir / "composite.png").convert("RGB"))
            except Exception as exc:
                print(f"[edges] skip {doc_id}: {exc}")
                failed_docs += 1
                continue

            doc_ids.append(doc_id)
            doc_h5_idx = len(doc_ids) - 1
            H, W = gt["image_shape"][:2]

            this_doc_pairs = 0
            for ga, gb, ca, cb in _iter_doc_pairs(doc_dir, gt, cmp_meta):
                pair = _process_pair(
                    composite, (H, W), ga, gb, ca, cb,
                    strip_w=strip_w, strip_l=strip_l,
                    min_boundary_px=min_boundary_px,
                    proximity_px=proximity_px,
                )
                if pair is None:
                    skipped_short_boundary += 1
                    continue
                ra, rb = pair

                # Append two new strips.
                idx_a_global = n_strips_so_far
                idx_b_global = n_strips_so_far + 1
                strips_ds.resize(n_strips_so_far + 2, axis=0)
                strips_ds[idx_a_global] = ra.strip
                strips_ds[idx_b_global] = rb.strip
                doc_idx_ds.resize(n_strips_so_far + 2, axis=0)
                doc_idx_ds[idx_a_global] = doc_h5_idx
                doc_idx_ds[idx_b_global] = doc_h5_idx
                frag_idx_ds.resize(n_strips_so_far + 2, axis=0)
                frag_idx_ds[idx_a_global] = ra.fragment_idx
                frag_idx_ds[idx_b_global] = rb.fragment_idx
                partner_idx_ds.resize(n_strips_so_far + 2, axis=0)
                partner_idx_ds[idx_a_global] = ra.partner_fragment_idx
                partner_idx_ds[idx_b_global] = rb.partner_fragment_idx
                arc_len_ds.resize(n_strips_so_far + 2, axis=0)
                arc_len_ds[idx_a_global] = ra.boundary_arc_len_px
                arc_len_ds[idx_b_global] = rb.boundary_arc_len_px
                n_strips_so_far += 2

                pairs_ds.resize(n_pairs_so_far + 1, axis=0)
                pairs_ds[n_pairs_so_far] = (idx_a_global, idx_b_global)
                n_pairs_so_far += 1
                this_doc_pairs += 1

            if (di + 1) % 50 == 0 or di == 0:
                print(f"[edges] [{di + 1}/{len(index_entries)}] {doc_id}: "
                      f"+{this_doc_pairs} pairs (running total {n_pairs_so_far})")

        # Write doc_ids string dataset.
        doc_ids_bytes = np.array(doc_ids, dtype=h5py.string_dtype("utf-8"))
        h5.create_dataset("doc_ids", data=doc_ids_bytes)

        # Persist build params for reproducibility.
        h5.attrs["strip_width"] = strip_w
        h5.attrs["strip_length"] = strip_l
        h5.attrs["min_boundary_px"] = min_boundary_px
        h5.attrs["proximity_px"] = proximity_px

    summary = {
        "out_path": str(out_path),
        "n_strips": n_strips_so_far,
        "n_positive_pairs": n_pairs_so_far,
        "n_docs": len(doc_ids) if 'doc_ids' in locals() else 0,
        "skipped_short_boundary": skipped_short_boundary,
        "skipped_no_boundary": skipped_no_boundary,
        "failed_docs": failed_docs,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path,
                     default=Path("data/dataset/index.json"))
    ap.add_argument("--out_dir", type=Path,
                     default=Path("data/dataset/edge_strips"))
    ap.add_argument("--strip_w", type=int, default=DEFAULT_STRIP_WIDTH)
    ap.add_argument("--strip_l", type=int, default=DEFAULT_STRIP_LENGTH)
    ap.add_argument("--min_boundary_px", type=float,
                     default=DEFAULT_MIN_BOUNDARY_PX)
    ap.add_argument("--proximity_px", type=float,
                     default=DEFAULT_BOUNDARY_PROXIMITY_PX)
    ap.add_argument("--max_docs", type=int, default=None,
                     help="Cap on docs per split (smoke testing).")
    args = ap.parse_args()

    if not args.index.exists():
        raise SystemExit(f"index not found: {args.index}")
    with open(args.index) as fh:
        master = json.load(fh)

    synthetic_root = Path(master["synthetic_root"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        entries = master[split]
        if not entries:
            print(f"[edges] skipping {split}: empty")
            continue
        out_path = args.out_dir / f"{split}.h5"
        print(f"[edges] === building {split}.h5  ({len(entries)} docs) ===")
        summary = build_split(
            entries, synthetic_root, out_path,
            strip_w=args.strip_w, strip_l=args.strip_l,
            min_boundary_px=args.min_boundary_px,
            proximity_px=args.proximity_px,
            max_docs=args.max_docs,
        )
        print(f"[edges] {split}: {summary}")
        with open(args.out_dir / f"{split}_summary.json", "w",
                   encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
