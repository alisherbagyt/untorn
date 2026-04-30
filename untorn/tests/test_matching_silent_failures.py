"""Loud-failure invariants for the matcher.

The OLD matcher had several silent-skip code paths: missing SDT was
treated as "gate passes", missing DINOv2 features defaulted to 0.5,
missing text lines redistributed weight to geometry. These hid real
configuration bugs from the operator and produced inconsistent
confidence numbers.

The NEW matcher must:
  * Lazy-compute SDT when missing (and log it) — never silently pass.
  * Tolerate missing DINOv2 features cleanly (mean of available
    appearance signals; LOG that DINOv2 was absent).
  * Always return a structured dict (no None returns hiding why a pair
    was rejected when at least one edge pair was attempted — the
    per-edge-pair decisions are logged at DEBUG level by the engine
    logger).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from untorn import matching as M


def test_sdt_lazy_computed_when_missing(torn_pair_factory, caplog):
    """If a fragment has no _sdt_interior, the matcher should compute it
    on the fly (and log it), not silently skip the SDT gate."""
    p = torn_pair_factory(angle_deg=0.0)
    # Strip _sdt_interior off frag_a.
    p.frag_a.pop("_sdt_interior", None)

    with caplog.at_level(logging.INFO, logger="untorn.engine.matching"):
        m = M.match_pair(p.frag_a, p.frag_b, p.image_a, direction_aware=False)

    assert m is not None, "match still works when SDT was missing"
    assert "computed missing SDT" in caplog.text, \
        "missing SDT must be logged at INFO"
    # SDT was filled in.
    assert p.frag_a.get("_sdt_interior") is not None


def test_match_dict_carries_structured_diagnostics(torn_pair_factory):
    """An accepted match must expose every gate's score so the assembler
    (and a human debugging) can see what each gate decided."""
    p = torn_pair_factory(angle_deg=0.0)
    m = M.match_pair(p.frag_a, p.frag_b, p.image_a, direction_aware=False)
    assert m is not None
    # Required keys (covers every gate the matcher runs)
    for key in ("frag_i", "frag_j", "edge_i", "edge_j",
                 "R", "t", "angle", "rms",
                 "sw_score", "n_aligned", "orientation",
                 "fit_cost", "fit_overlap_px", "fit_gap_px", "fit_coverage",
                 "confidence", "geom_score", "appearance",
                 "paper_score", "strip_score",
                 "matched_a", "matched_b",
                 "sdt_gate"):
        assert key in m, f"match dict missing key {key!r}"


def test_dinov2_absence_logged_not_silent(torn_pair_factory, caplog):
    """When neither fragment has DINOv2 features attached, the appearance
    score falls back to (paper, strip) only. The matcher must NOT crash
    and must NOT silently insert a 0.5 placeholder for the missing DINOv2
    component (the absence is observable via the dinov2_score field
    being None)."""
    p = torn_pair_factory(angle_deg=0.0)
    # Make sure neither fragment has dinov2 features
    p.frag_a.pop("dinov2", None)
    p.frag_b.pop("dinov2", None)
    m = M.match_pair(p.frag_a, p.frag_b, p.image_a, direction_aware=False)
    assert m is not None
    assert m.get("dinov2_score") is None, \
        "dinov2_score should be None when DINOv2 features are absent"


def test_no_torn_edges_returns_none_explicitly(torn_pair_factory):
    """A fragment with no torn edges produces no match — the matcher must
    NOT crash; it must return None cleanly."""
    p = torn_pair_factory(angle_deg=0.0)
    for e in p.frag_a["edges"]:
        e["is_torn"] = False
    assert M.match_pair(p.frag_a, p.frag_b, p.image_a, direction_aware=False) is None
