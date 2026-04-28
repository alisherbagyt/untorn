"""
untorn.fragment_profile
=======================
Phase 2.5 - per-fragment individual analysis.

Before pair-matching, each fragment is profiled in isolation: every torn
edge gets a quality grade, an inward color/ink fingerprint, a curvature
distinctiveness score, and a list of which other torn edges it shares a
support point with (a 'corner-mate' on the same fragment). Fragments are
classified by document role - corner / boundary / interior - based on
whether they own factory (straight) edges.

The profile is the data the assembler USES to pick which pairs to score
first, which seam is the highest-quality one to seed the MST from, and
which fragments are likely to anchor the document corners.

Outputs (per job, written by assembly.reconstruct via debug_dir):
    fragment_profiles.json  - everything below, JSON-serialised

Per-fragment record:
{
    "id":                  int,
    "centroid":            [x, y],
    "bbox":                [x, y, w, h],
    "area_px":             int,
    "perimeter_px":        float,
    "n_torn_edges":        int,
    "n_factory_edges":     int,
    "role":                "corner" | "boundary" | "interior",
    "ink_density":         float,            # fraction of mask pixels that are ink
    "paper_lab":           [L, a, b] | null, # cached fingerprint
    "n_text_lines":        int,
    "text_angle_deg":      float | null,     # median local baseline angle
    "torn_edges": [
        {
            "edge_idx":           int,
            "length_px":          float,
            "is_torn":            true,
            "curvature_std":      float,    # signal strength
            "curvature_range":    float,
            "anchor_strength":    float,    # 0..1 score - higher = more matchable
            "ink_strip_density":  float,    # ink near this edge inside the fragment
            "paper_strip_lab":    [L,a,b],
            "midpoint":           [x, y],
            "outward_normal":     [nx, ny],
            "endpoint_a":         [x, y],   # support point at the start
            "endpoint_b":         [x, y],   # support point at the end
            "shared_corners":     [int]     # edge indices that share an endpoint
        }, ...
    ]
}

"anchor_strength" is the function the matcher uses to decide which edges
deserve top billing in candidate enumeration: long, curvy, ink-rich,
distinctively-coloured edges land near 1.0; short, straight, smooth-
texture edges sit near 0.0 and get matched only as a last resort.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Anchor strength - per-torn-edge "how matchable is this edge?"
# ---------------------------------------------------------------------------

def _curvature_features(edge: dict) -> tuple[float, float]:
    """Return (std, range) of the curvature feature string. Zero if missing."""
    curv = edge.get("_curvature")
    if curv is None or len(curv) < 4:
        return 0.0, 0.0
    arr = np.asarray(curv, dtype=np.float64)
    return float(np.std(arr)), float(np.ptp(arr))


def _strip_ink_and_paper(image_rgb: np.ndarray,
                         mask: np.ndarray,
                         edge_pts: np.ndarray,
                         outward_normal: np.ndarray,
                         depth_px: int = 12,
                         ink_grayscale_max: int = 140) -> tuple[float, tuple[float, float, float] | None]:
    """
    Sample a strip of pixels inside the fragment along the inward direction
    of an edge. Returns (ink_density, paper_lab_mean) - the fraction of strip
    pixels darker than the ink threshold (i.e. text), and the mean LAB of the
    NON-ink pixels (i.e. paper). LAB is None if there aren't enough samples.
    """
    if edge_pts is None or len(edge_pts) < 2:
        return 0.0, None
    h, w = mask.shape[:2]
    inward = -outward_normal
    n = float(np.linalg.norm(inward))
    if n < 1e-6:
        return 0.0, None
    inward = inward / n

    rows = []
    n_samples = min(len(edge_pts), 256)
    step_idx = max(1, len(edge_pts) // n_samples)
    for k in range(0, len(edge_pts), step_idx):
        pt = edge_pts[k]
        for d in range(1, depth_px + 1):
            sp = pt + d * inward
            xi = int(np.clip(round(sp[0]), 0, w - 1))
            yi = int(np.clip(round(sp[1]), 0, h - 1))
            if mask[yi, xi] > 127:
                rows.append(image_rgb[yi, xi])

    if len(rows) < 6:
        return 0.0, None
    arr = np.asarray(rows, dtype=np.uint8)
    gray = cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(-1)
    ink_mask = gray < ink_grayscale_max
    ink_density = float(ink_mask.mean())
    paper = arr[~ink_mask]
    if paper.size < 9:
        return ink_density, None
    paper_lab = cv2.cvtColor(paper.reshape(-1, 1, 3),
                              cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return ink_density, (
        float(paper_lab[:, 0].mean()),
        float(paper_lab[:, 1].mean()),
        float(paper_lab[:, 2].mean()),
    )


def _anchor_strength(length_px: float,
                     curv_std: float,
                     curv_range: float,
                     ink_density: float,
                     reference_length_px: float) -> float:
    """
    Combine length, curvature distinctiveness and ink presence into a
    single 0..1 score. Long curvy ink-rich edges -> close to 1.0. Short
    straight blank edges -> close to 0.0.
    """
    len_score = min(1.0, length_px / max(reference_length_px, 1.0))
    curv_score = min(1.0, curv_std / 0.35) * 0.6 + min(1.0, curv_range / 1.6) * 0.4
    ink_score = min(1.0, ink_density / 0.2)
    # Weighted blend: length is the strongest predictor of matchability,
    # then curvature signal, then ink (mostly a tie-breaker).
    return float(0.5 * len_score + 0.35 * curv_score + 0.15 * ink_score)


# ---------------------------------------------------------------------------
# Cross-edge endpoints - which torn edges meet at the same support point
# ---------------------------------------------------------------------------

def _shared_corners(edges: list[dict]) -> list[list[int]]:
    """
    For each edge, list the indices of OTHER torn edges that share a
    support point with it. Two adjacent torn edges meeting at the same
    corner often continue around a single tear - we want the matcher to
    consider them as a unit.
    """
    n = len(edges)
    result: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        if not edges[i].get("is_torn"):
            continue
        si = int(edges[i].get("start_sp", -1))
        ei = int(edges[i].get("end_sp", -1))
        for j in range(n):
            if j == i or not edges[j].get("is_torn"):
                continue
            sj = int(edges[j].get("start_sp", -1))
            ej = int(edges[j].get("end_sp", -1))
            if si in (sj, ej) or ei in (sj, ej):
                result[i].append(j)
    return result


# ---------------------------------------------------------------------------
# Document role classification
# ---------------------------------------------------------------------------

def _classify_role(edges: list[dict]) -> str:
    """Corner / boundary / interior based on factory-edge count + adjacency."""
    factory = [e for e in edges if not e.get("is_torn", False)]
    n_factory = len(factory)
    if n_factory == 0:
        return "interior"
    if n_factory == 1:
        return "boundary"
    # Two or more factory edges - if any two of them share a support point,
    # this fragment owns a document corner.
    factory_sps = []
    for e in factory:
        factory_sps.append((int(e.get("start_sp", -1)),
                            int(e.get("end_sp", -1))))
    for i in range(len(factory_sps)):
        for j in range(i + 1, len(factory_sps)):
            si1, ei1 = factory_sps[i]
            si2, ei2 = factory_sps[j]
            if {si1, ei1} & {si2, ei2}:
                return "corner"
    return "boundary"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_profile(frag: dict, image_rgb: np.ndarray,
                  reference_length_px: float | None = None) -> dict:
    """Build a single fragment's profile."""
    edges = frag.get("edges", []) or []
    mask = frag["mask"]

    # Choose a "reference length" to normalise per-edge length scores.
    if reference_length_px is None:
        torn_lens = [float(e.get("length", 0.0))
                     for e in edges if e.get("is_torn", False)]
        reference_length_px = max(torn_lens) if torn_lens else 100.0

    n_torn = sum(1 for e in edges if e.get("is_torn", False))
    n_factory = len(edges) - n_torn
    role = _classify_role(edges)
    shared = _shared_corners(edges)

    # Whole-fragment ink density (cheap, useful as a sanity-check).
    if image_rgb is not None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        ink_pixels = ((mask > 127) & (gray < 140))
        denom = int((mask > 127).sum())
        ink_density = float(ink_pixels.sum() / max(denom, 1))
    else:
        ink_density = 0.0

    text_lines = frag.get("text_lines") or []
    if text_lines:
        angles = [float(t.get("angle", 0.0)) for t in text_lines]
        text_angle_deg = float(math.degrees(np.median(angles)))
    else:
        text_angle_deg = None

    paper_lab = frag.get("paper_lab")
    if paper_lab is not None:
        paper_lab_list = [float(v) for v in np.asarray(paper_lab).reshape(-1)]
    else:
        paper_lab_list = None

    torn_records = []
    for k, e in enumerate(edges):
        if not e.get("is_torn", False):
            continue
        c_std, c_rng = _curvature_features(e)
        ink_strip, paper_strip_lab = _strip_ink_and_paper(
            image_rgb, mask,
            np.asarray(e.get("pts", []), dtype=np.float64),
            np.asarray(e.get("outward_normal", (0.0, 0.0)), dtype=np.float64))
        anchor = _anchor_strength(
            length_px=float(e.get("length", 0.0)),
            curv_std=c_std, curv_range=c_rng,
            ink_density=ink_strip,
            reference_length_px=reference_length_px,
        )
        pts = np.asarray(e.get("pts", []), dtype=np.float64)
        endpoint_a = pts[0].tolist() if len(pts) else [0.0, 0.0]
        endpoint_b = pts[-1].tolist() if len(pts) else [0.0, 0.0]
        torn_records.append({
            "edge_idx":          int(k),
            "length_px":         round(float(e.get("length", 0.0)), 2),
            "is_torn":           True,
            "curvature_std":     round(c_std, 4),
            "curvature_range":   round(c_rng, 4),
            "anchor_strength":   round(anchor, 4),
            "ink_strip_density": round(float(ink_strip), 4),
            "paper_strip_lab":   ([round(c, 2) for c in paper_strip_lab]
                                  if paper_strip_lab is not None else None),
            "midpoint":          [round(float(v), 2) for v in
                                   np.asarray(e.get("midpoint", (0.0, 0.0)))],
            "outward_normal":    [round(float(v), 4) for v in
                                   np.asarray(e.get("outward_normal", (0.0, 0.0)))],
            "endpoint_a":        [round(float(v), 2) for v in endpoint_a],
            "endpoint_b":        [round(float(v), 2) for v in endpoint_b],
            "shared_corners":    list(shared[k]),
        })

    # Sort torn edges within the profile by anchor strength descending so
    # the JSON reads "the strongest edges first".
    torn_records.sort(key=lambda r: -r["anchor_strength"])

    perimeter = float(sum(float(e.get("length", 0.0)) for e in edges))
    bbox = list(int(v) for v in frag.get("bbox", (0, 0, 0, 0)))

    return {
        "id":               int(frag.get("id", 0)),
        "centroid":         [round(float(v), 2) for v in
                              np.asarray(frag.get("centroid", (0.0, 0.0)))],
        "bbox":             bbox,
        "area_px":          int((mask > 127).sum()),
        "perimeter_px":     round(perimeter, 1),
        "n_torn_edges":     int(n_torn),
        "n_factory_edges":  int(n_factory),
        "role":             role,
        "ink_density":      round(ink_density, 4),
        "paper_lab":        paper_lab_list,
        "n_text_lines":     int(len(text_lines)),
        "text_angle_deg":   (round(float(text_angle_deg), 2)
                              if text_angle_deg is not None else None),
        "torn_edges":       torn_records,
    }


def build_fragment_profiles(fragments: list[dict],
                             image_rgb: np.ndarray) -> list[dict]:
    """Run build_profile on every fragment, share a global reference length."""
    # Use the 75th-percentile torn-edge length across the whole scene as the
    # length reference, so anchor_strength is comparable between fragments.
    all_torn_lens = []
    for f in fragments:
        for e in f.get("edges", []) or []:
            if e.get("is_torn", False):
                all_torn_lens.append(float(e.get("length", 0.0)))
    if all_torn_lens:
        ref = float(np.quantile(np.asarray(all_torn_lens), 0.75))
    else:
        ref = 100.0
    if ref <= 0:
        ref = 100.0
    return [build_profile(f, image_rgb, reference_length_px=ref)
            for f in fragments]


def save_profiles(profiles: list[dict], debug_dir: Path) -> None:
    """Write the profiles to <debug_dir>/fragment_profiles.json."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / "fragment_profiles.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(profiles, fh, indent=2)


def print_profile_summary(profiles: list[dict]) -> None:
    """One-line per-fragment summary printed to stdout for the operator."""
    print(f"  -- Fragment profiles ({len(profiles)} fragments) --")
    for p in profiles:
        torn = p["torn_edges"]
        anchors = " ".join(
            f"e{r['edge_idx']}={r['anchor_strength']:.2f}"
            for r in torn[:3])
        print(f"     frag {p['id']:>2}: role={p['role']:<8} "
              f"torn={p['n_torn_edges']} fac={p['n_factory_edges']} "
              f"ink={p['ink_density']:.2f} "
              f"top_torn=[{anchors}]")


__all__ = [
    "build_profile",
    "build_fragment_profiles",
    "save_profiles",
    "print_profile_summary",
]
