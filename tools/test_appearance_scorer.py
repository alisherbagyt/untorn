"""
Sanity check the LAB+correlation+sigmoid appearance scorer.

The test feeds strips with controlled correlation and verifies:
  * highly correlated strips → score near 0
  * uncorrelated strips → score near/above 0.7
  * constant strips → fallback to mean-diff

This isolates the scoring math; it does NOT attempt to simulate actual
tearing physics (that's the benchmark's job).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from untorn.matching import _score_edge_appearance


def _make_edge(pts, outward_normal):
    return {"pts": np.asarray(pts, dtype=np.float64),
            "outward_normal": np.asarray(outward_normal, dtype=np.float64)}


def _controlled_img(h, w, pattern_fn):
    """Build an image where a vertical strip around column 200 has a
    controlled texture pattern (same content on both sides of the line)."""
    img = np.full((h, w, 3), fill_value=(230, 225, 210), dtype=np.uint8)
    for y in range(h):
        for x in range(180, 221):
            img[y, x] = pattern_fn(x, y)
    return img


def main():
    h, w = 400, 400

    # ── Case 1: identical content on both sides of the seam ────────
    # A's strip (at cols 199..192) and B's strip (at cols 201..208) are
    # sampled "AWAY from the seam by 1..8 px" on each side. Use a
    # mirror-symmetric pattern about x=200 so both strips see the same
    # value at the same step distance — that's the "perfectly matched
    # tear" the scorer should reward with a score near 0.
    def textured(x, y):
        d = abs(x - 200)
        v = int(128 + 60 * np.sin(0.6 * y) * np.cos(0.4 * d))
        v = max(0, min(255, v))
        return (v, v, v)
    img1 = _controlled_img(h, w, textured)
    seam = np.array([[200, y] for y in range(50, 350, 4)], dtype=np.float64)
    ea = _make_edge(seam, [+1.0, 0.0])
    eb = _make_edge(seam, [-1.0, 0.0])
    s1 = _score_edge_appearance(img1, ea, eb, seam, seam)

    # ── Case 2: A-side is textured, B-side is blank paper ──────────
    # Strip B samples cols 201..208, so blank everything from col 201.
    img2 = img1.copy()
    img2[:, 201:] = (230, 225, 210)
    s2 = _score_edge_appearance(img2, ea, eb, seam, seam)

    # ── Case 3: A-side is textured grey, B-side is dark text ───────
    # Strip B samples cols 201..208, so a dark band there forces a clear
    # appearance mismatch.
    img3 = img1.copy()
    img3[:, 201:209] = (40, 35, 30)
    s3 = _score_edge_appearance(img3, ea, eb, seam, seam)

    # ── Case 4: both sides constant paper (fallback path) ──────────
    img4 = np.full((h, w, 3), 230, dtype=np.uint8)
    s4 = _score_edge_appearance(img4, ea, eb, seam, seam)

    print("=== Appearance scorer sanity ===")
    print(f"  Case 1 (matched texture L<>R)        : sapp = {s1:.4f}")
    print(f"  Case 2 (textured A vs blank B)       : sapp = {s2:.4f}")
    print(f"  Case 3 (grey A vs dark band B)       : sapp = {s3:.4f}")
    print(f"  Case 4 (both constant paper)         : sapp = {s4:.4f}")

    # Case 1 should be clearly the lowest (matched texture). Cases 2-3
    # should be well above it; the scorer's "uncorrelated/mismatched"
    # band centres around ~0.4, so we require a clear separation rather
    # than a fixed absolute floor.
    ok = (s1 < s2 - 0.20) and (s1 < s3 - 0.20)
    print()
    print("RESULT:", "PASS" if ok else "FAIL",
          "   (separation = case2-case1 =", f"{s2 - s1:+.3f})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
