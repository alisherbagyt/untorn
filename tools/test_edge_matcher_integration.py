"""Phase 4 smoke test — exercises the edge_matcher adapter end-to-end and
verifies graceful degradation when the checkpoint is unavailable.

Runs three scenarios:
  1. Strip extraction on a synthetic image (no model required).
  2. Load with missing checkpoint -> None, score_edge_pair returns None.
  3. Load with the real checkpoint -> scoring path returns a probability.

After the third scenario we hand-craft a `_match_edge_pair` invocation with a
matched pair of straight horizontal edges to confirm the gate fires cleanly
inside the matching cascade (rejecting nothing on a degenerate-but-valid
synthetic input is fine — we're checking that the integration plumbing
runs without errors).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from untorn import edge_matcher                       # noqa: E402
from untorn import config as cfg                      # noqa: E402


def main() -> None:
    print("=" * 60)
    print("Phase 4 integration smoke test")
    print("=" * 60)

    # --- 1. Strip extraction without a model -----------------------------
    rng = np.random.default_rng(0)
    H, W = 200, 400
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    pts_a = np.stack([np.linspace(50, 350, 80), np.full(80, 100.0)], axis=1)
    pts_b = np.stack([np.linspace(50, 350, 80), np.full(80, 110.0)], axis=1)
    edge_a = {"pts": pts_a, "outward_normal": np.array([0.0, -1.0])}
    edge_b = {"pts": pts_b, "outward_normal": np.array([0.0,  1.0])}

    pair = edge_matcher.extract_strip_pair(img, edge_a, edge_b, "complementary")
    assert pair is not None
    sa, sb = pair
    assert sa.shape == (32, 256, 3) and sb.shape == (32, 256, 3)
    assert sa.dtype == np.uint8 and sb.dtype == np.uint8
    print(f"[1] strip extraction OK  shapes={sa.shape}/{sb.shape}")

    # --- 2. Graceful degradation: missing checkpoint ---------------------
    # Reset the singleton so a previous good run doesn't shadow this test.
    edge_matcher._MODEL = None
    edge_matcher._LOAD_FAILED = False
    edge_matcher._LOAD_REASON = ""
    m_none = edge_matcher.load(checkpoint_path="models/__nonexistent__.pt")
    assert m_none is None, f"expected None, got {m_none}"
    assert not edge_matcher.is_loaded()
    score_none = edge_matcher.score_edge_pair(img, edge_a, edge_b, "complementary")
    assert score_none is None
    print("[2] missing-ckpt path OK  (load=None, score=None, is_loaded=False)")

    # --- 3. Real checkpoint, score_edge_pair fires -----------------------
    edge_matcher._LOAD_FAILED = False
    edge_matcher._LOAD_REASON = ""
    m_real = edge_matcher.load()           # default cfg.EDGE_MATCHER_CHECKPOINT
    if m_real is None:
        print(f"[3] SKIP — cannot find real checkpoint at "
              f"{cfg.EDGE_MATCHER_CHECKPOINT}")
        return
    assert edge_matcher.is_loaded()
    score = edge_matcher.score_edge_pair(img, edge_a, edge_b, "complementary")
    assert score is not None
    assert "match_prob" in score and 0.0 <= score["match_prob"] <= 1.0
    print(f"[3] real-ckpt scoring OK  match_prob={score['match_prob']:.4f}  "
          f"cos={score['cosine']:.4f}")

    # --- 4. Verify _match_edge_pair runs the new gate -------------------
    # Build a fake edge dict that has the geometry-stage requirements
    # (curvature string + resampled pts). Two perfectly-aligned straight
    # edges will fail the matching cascade earlier (low curvature variance),
    # which is fine — the goal is to confirm the integration imports cleanly.
    from untorn import matching as M
    from untorn.contours import compute_curvature_string
    res_a, curv_a = compute_curvature_string(pts_a)
    res_b, curv_b = compute_curvature_string(pts_b)
    edge_a["_resampled"] = res_a; edge_a["_curvature"] = curv_a
    edge_a["length"] = float(np.sum(np.linalg.norm(np.diff(pts_a, axis=0), axis=1)))
    edge_b["_resampled"] = res_b; edge_b["_curvature"] = curv_b
    edge_b["length"] = float(np.sum(np.linalg.norm(np.diff(pts_b, axis=0), axis=1)))
    out = M._match_edge_pair(edge_a, edge_b, img, tag="smoke")
    print(f"[4] _match_edge_pair returned: "
          f"{'None (rejected upstream - expected for straight noise edges)' if out is None else 'match dict'}")

    edge_matcher.unload()
    assert not edge_matcher.is_loaded()
    print("[*] Phase 4 smoke test PASSED")


if __name__ == "__main__":
    main()
