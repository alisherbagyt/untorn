"""Step 1 of Phase 1 — extract PubLayNet images from parquet files.

Reads a parquet file, decodes the `image.bytes` column (JPEG), saves each image
as a PNG to disk, and writes an `index.json` mapping document id -> filename.

Usage:
    python tools/extract_publaynet.py
        --parquet data/dataset/PubLayNet/parquet/train-00001-of-00208-...parquet
        --out_dir data/dataset/PubLayNet/images
        --limit 500            # extract only first 500 (omit for all)
        --split train          # tag (just for the index)

Output layout:
    {out_dir}/
        {id}.png               (one per row)
        index.json             (appended-to / merged across runs)

The index is shared across all parquets so multiple runs accumulate. Each entry:
    {"id": int, "filename": "<id>.png", "split": "train"|"val", "source_parquet": "<basename>"}
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def extract_parquet(parquet_path: Path, out_dir: Path, *, limit: int | None,
                    split: str) -> list[dict]:
    """Decode every image in `parquet_path` and write it as a PNG.

    Returns a list of index entries appended this run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    pqfile = pq.ParquetFile(str(parquet_path))
    total = pqfile.metadata.num_rows
    target = min(total, limit) if limit else total
    print(f"[extract] {parquet_path.name}: {total} rows, extracting {target}")

    seen = 0
    for batch in pqfile.iter_batches(batch_size=64, columns=["image", "id"]):
        ids = batch.column("id").to_pylist()
        images = batch.column("image").to_pylist()
        for doc_id, img_struct in zip(ids, images):
            if limit and seen >= limit:
                break
            try:
                pil = Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB")
            except Exception as exc:  # pragma: no cover
                print(f"[extract] WARN: skipping id={doc_id} ({exc})")
                continue
            out_path = out_dir / f"{doc_id}.png"
            pil.save(out_path, format="PNG", optimize=False, compress_level=1)
            entries.append({
                "id": int(doc_id),
                "filename": out_path.name,
                "split": split,
                "source_parquet": parquet_path.name,
            })
            seen += 1
            if seen % 50 == 0:
                print(f"[extract]   ... {seen}/{target}")
        if limit and seen >= limit:
            break
    print(f"[extract] {parquet_path.name}: wrote {seen} PNGs")
    return entries


def merge_index(out_dir: Path, new_entries: list[dict]) -> None:
    """Merge new entries into out_dir/index.json (de-duplicated by id)."""
    index_path = out_dir / "index.json"
    existing: dict[int, dict] = {}
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as fh:
            for entry in json.load(fh):
                existing[entry["id"]] = entry
    for entry in new_entries:
        existing[entry["id"]] = entry
    merged = sorted(existing.values(), key=lambda e: e["id"])
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"[extract] index.json updated: {len(merged)} total entries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path,
                    help="Path to one parquet file")
    ap.add_argument("--out_dir", default=Path("data/dataset/PubLayNet/images"),
                    type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max rows to extract (omit for all rows in parquet)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    args = ap.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"Parquet not found: {args.parquet}")

    entries = extract_parquet(args.parquet, args.out_dir,
                              limit=args.limit, split=args.split)
    merge_index(args.out_dir, entries)


if __name__ == "__main__":
    main()
