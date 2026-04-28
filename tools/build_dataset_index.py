"""Step 4 of Phase 1 — build the master dataset index.

Walks ``data/dataset/synthetic/`` (one subdir per document) and produces
``data/dataset/index.json`` — a flat list of training examples consumable by
the Phase 2 / Phase 3 / Phase 5 scripts.

Each entry contains everything a downstream loader needs to find:
  * the original image  (Y target)
  * the composite       (X input)
  * the per-fragment metadata + transforms (ground truth)

Splitting strategy
------------------
By default, every document gets a deterministic hash-based split using
``--val_frac`` (default 0.2). This means about 20% of all documents become
the validation set, regardless of which parquet they came from. The split is
reproducible across runs because it's a hash of the doc_id.

If you want to instead respect the train/val labels recorded by
``extract_publaynet.py`` (i.e. use the parquets' own split), pass
``--respect_publaynet_split``.

CLI:
    python tools/build_dataset_index.py
        --synthetic_root data/dataset/synthetic
        --out data/dataset/index.json
        --val_frac 0.2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FILES = (
    "original.png",
    "composite.png",
    "ground_truth.json",
    "composite_transforms.json",
)


def _is_complete(doc_dir: Path) -> bool:
    return all((doc_dir / f).exists() for f in REQUIRED_FILES)


def _load_publaynet_split(index_path: Path) -> dict[str, str]:
    """Return {doc_id_str: 'train'|'val'} from the publaynet image index."""
    if not index_path.exists():
        return {}
    with open(index_path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    out: dict[str, str] = {}
    for row in rows:
        out[str(row["id"])] = row.get("split", "train")
    return out


def _hash_split(doc_id: str, val_frac: float) -> str:
    h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / float(0xFFFFFFFF)
    return "val" if bucket < val_frac else "train"


def _per_doc_summary(doc_dir: Path) -> dict | None:
    """Read the JSON metadata for one document and reduce it to compact entry."""
    try:
        with open(doc_dir / "ground_truth.json", "r", encoding="utf-8") as fh:
            gt = json.load(fh)
        with open(doc_dir / "composite_transforms.json", "r",
                  encoding="utf-8") as fh:
            cmp_meta = json.load(fh)
    except Exception:
        return None

    placements = cmp_meta.get("placements", [])
    placed_ids = sorted(p["fragment_index"] for p in placements)

    # Build a fragment-index → adjacency list view that's restricted to
    # fragments actually placed on the composite (so downstream loaders can
    # build positive training pairs without re-checking).
    placed_set = set(placed_ids)
    adjacency: dict[int, list[int]] = {}
    for frag in gt["fragments"]:
        idx = frag["index"]
        if idx not in placed_set:
            continue
        adjacency[idx] = [n for n in frag["neighbors"] if n in placed_set]

    return {
        "doc_id": gt["doc_id"],
        "image_shape": gt["image_shape"],          # [H, W, 3]
        "canvas_size": cmp_meta["canvas_size"],     # [W, H]
        "n_fragments_total": len(gt["fragments"]),
        "n_fragments_placed": len(placed_ids),
        "placed_fragment_indices": placed_ids,
        "skipped_fragment_indices": cmp_meta.get(
            "skipped_fragment_indices", []),
        "adjacency_placed": adjacency,             # idx -> [neighbor_idx, ...]
        "n_positive_pairs": sum(
            1 for idx, nbrs in adjacency.items() for n in nbrs if n > idx),
        "files": {
            "original": "original.png",
            "composite": "composite.png",
            "ground_truth": "ground_truth.json",
            "composite_transforms": "composite_transforms.json",
            "fragments_dir": "fragments",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_root",
                     default=Path("data/dataset/synthetic"), type=Path)
    ap.add_argument("--publaynet_index",
                     default=Path("data/dataset/PubLayNet/images/index.json"),
                     type=Path)
    ap.add_argument("--out", default=Path("data/dataset/index.json"), type=Path)
    ap.add_argument("--val_frac", type=float, default=0.2,
                     help="Fraction of documents assigned to validation via "
                          "deterministic hash split.")
    ap.add_argument("--respect_publaynet_split", action="store_true",
                     help="Use the train/val labels from "
                          "publaynet/images/index.json instead of hash split.")
    args = ap.parse_args()

    if not args.synthetic_root.exists():
        raise SystemExit(f"synthetic_root not found: {args.synthetic_root}")

    publaynet_split = _load_publaynet_split(args.publaynet_index)
    if args.respect_publaynet_split:
        print(f"[index] using publaynet split: {len(publaynet_split)} entries")
    else:
        print(f"[index] using hash-based split: val_frac={args.val_frac}")

    docs = sorted(p for p in args.synthetic_root.iterdir() if p.is_dir())
    train: list[dict] = []
    val: list[dict] = []
    incomplete = 0
    no_pairs = 0

    for doc_dir in docs:
        if not _is_complete(doc_dir):
            incomplete += 1
            continue
        summary = _per_doc_summary(doc_dir)
        if summary is None:
            incomplete += 1
            continue

        doc_id = str(summary["doc_id"])
        if args.respect_publaynet_split:
            split = publaynet_split.get(doc_id) or _hash_split(doc_id, args.val_frac)
        else:
            split = _hash_split(doc_id, args.val_frac)

        # Path is relative to the index file's parent for portability.
        rel_root = doc_dir.relative_to(args.out.parent.resolve()
                                       if args.out.is_absolute()
                                       else args.synthetic_root.parent.resolve()
                                       if False
                                       else args.synthetic_root.parent)
        summary["root_relative"] = rel_root.as_posix()
        summary["split"] = split

        if summary["n_positive_pairs"] == 0:
            no_pairs += 1

        if split == "val":
            val.append(summary)
        else:
            train.append(summary)

    out = {
        "version": 1,
        "synthetic_root": str(args.synthetic_root.as_posix()),
        "n_train": len(train),
        "n_val": len(val),
        "n_incomplete": incomplete,
        "n_docs_without_positive_pairs": no_pairs,
        "train": train,
        "val": val,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"[index] train={len(train)}  val={len(val)}  "
          f"incomplete={incomplete}  no_pairs={no_pairs}")
    print(f"[index] wrote {args.out}")


if __name__ == "__main__":
    main()
