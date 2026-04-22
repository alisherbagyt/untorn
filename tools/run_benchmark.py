"""
tools/run_benchmark.py
======================
Generate synthetic torn-paper cases, run the UNTORN pipeline on each,
score against ground truth, and print an aggregate report.

Usage:

    python tools/run_benchmark.py \
        --source data/input/t6.tif \
        --out data/benchmark/run_01 \
        --n-cases 3 \
        --n-fragments 8

    # Generate-only (no GPU / pipeline):
    python tools/run_benchmark.py --source ... --out ... --generate-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when invoked directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows console encoding quirks when Python prints non-ASCII
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from untorn.benchmark.run_suite import run_suite
from untorn.benchmark.synth_doc import synth_document
import cv2


def main() -> int:
    ap = argparse.ArgumentParser(description="UNTORN benchmark runner")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="Clean source document image")
    src.add_argument("--synth-doc", action="store_true",
                     help="Generate a synthetic clean document as source")
    ap.add_argument("--synth-doc-size", default="900x1200",
                    help="WxH for --synth-doc (default 900x1200)")
    ap.add_argument("--out", required=True,
                    help="Output directory (cases + report land here)")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--n-fragments", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--roughen", type=float, default=6.0,
                    help="Tear roughness amplitude (pixels)")
    ap.add_argument("--max-rot", type=float, default=25.0,
                    help="Max per-fragment rotation (degrees)")
    ap.add_argument("--generate-only", action="store_true",
                    help="Skip running the pipeline; just generate inputs")
    args = ap.parse_args()

    # If the user asked for a synthesised document, produce one and point
    # the suite at it on disk (the suite API takes a path).
    source_path = args.source
    if args.synth_doc:
        w, h = map(int, args.synth_doc_size.lower().split("x"))
        doc = synth_document(width=w, height=h, seed=args.seed)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        source_path = str(out_dir / "synth_source.png")
        cv2.imwrite(source_path, cv2.cvtColor(doc, cv2.COLOR_RGB2BGR))
        print(f"  Synthesised clean document at {source_path}")

    run_suite(
        source_image_path=source_path,
        output_dir=args.out,
        n_cases=args.n_cases,
        n_fragments=args.n_fragments,
        seed=args.seed,
        roughen_amplitude=args.roughen,
        max_rotation_deg=args.max_rot,
        skip_pipeline=args.generate_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
