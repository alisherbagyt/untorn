"""
tools/test_packer.py
====================
Verify the synthetic-case packer places fragments without overlap.

For each pair of fragments, check that centre distance >= inscribed
circle sum (half-diagonals plus the configured gap). Fails loud if any
pair would overlap on the scan.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from untorn.benchmark.generate import generate_case, TearGeneratorConfig
from untorn.benchmark.synth_doc import synth_document


def _half_diag(bbox_wh, gap):
    w, h = bbox_wh
    return math.hypot(w, h) / 2 + gap


def _check(truth, gap):
    bboxes = [f["bbox_in_source"][2:] for f in truth["fragments"]]
    centres = [f["target_center_on_canvas"] for f in truth["fragments"]]
    bad = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            r_i = _half_diag(bboxes[i], gap)
            r_j = _half_diag(bboxes[j], gap)
            d = math.hypot(centres[i][0] - centres[j][0],
                           centres[i][1] - centres[j][1])
            if d < r_i + r_j:
                bad.append((i, j, d, r_i + r_j))
    return bad


def main() -> int:
    out = Path(os.environ["TEMP"]) / "untorn_pack_test"
    out.mkdir(exist_ok=True)
    src = synth_document(width=900, height=1200, seed=42)
    failures = 0
    for seed in (7, 17, 42, 99, 123, 2026):
        cfg = TearGeneratorConfig(n_fragments=6, seed=seed)
        truth = generate_case(src, out, f"pack_{seed}", cfg)
        bad = _check(truth, cfg.min_gap_px)
        ok = "PASS" if not bad else "FAIL"
        cw, ch = truth["canvas_size"]
        print(f"[{ok}] seed={seed:<4d} canvas={cw}x{ch:<5d}  "
              f"{len(truth['fragments'])} frags  {len(bad)} overlaps")
        if bad:
            failures += 1
            for i, j, d, r in bad:
                print(f"    frag {i}<>{j}: centre-dist={d:.1f} < r_sum={r:.1f}")
    if failures:
        print(f"\n{failures} seeds produced overlapping placements")
        return 1
    print("\nAll seeds produced strictly non-overlapping placements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
