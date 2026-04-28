"""
untorn.edge_rank
================
Per-edge partner ranking and mutual-rank scoring.

The legacy assembler enumerates pair candidates (i, j) and ranks them by
a single global confidence score. That makes the MST greedy on an
'absolute' axis: the highest-confidence pair wins. The catch is that the
SAME torn edge can be the top partner for two different fragments. The
greedy MST then anchors the edge to its first taker, even if the other
fragment is its better mutual match.

This module reframes the ranking around the EDGE, not the pair:

    For every torn edge of every fragment, build a sorted list of its
    candidate partner edges across other fragments. We then compute a
    mutual-rank score per pair: how high does each side rank the other?

A pair where edge A is edge B's #1 choice AND edge B is edge A's #1
choice is a 'mutual #1' - this is the strongest signal we have that
the seam is real, far stronger than absolute confidence on its own.

The MST seeder uses this to anchor on a pair that both sides agree is
their best partner; the MST grower uses it to validate proposed
attachments. Stale or wrong absolute-best matches that lose mutual
agreement get demoted.

Public API
----------
rank_edges_per_fragment(fragments, candidates, cache)
    -> dict[(frag_id, edge_idx), list[PartnerRanking]]

build_pair_mutual_scores(rankings, cache, fragments)
    -> dict[(min_id, max_id), {mutual_score, rank_a, rank_b, fit_cost,
                                confidence, edge_a_idx, edge_b_idx}]

save_rankings(rankings, mutual, debug_dir)

ranked_seed_candidates(mutual, profiles, top_n=10)
    -> list[seed_candidate dict]   - sorted, mutual-best first
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Internal: pull match-result fields out of a cached _MatchCache entry,
# normalising the edge orientation to the (frag_a, frag_b) order used here.
# ---------------------------------------------------------------------------

def _match_in_orientation(cache, fragments: list[dict],
                          a: int, b: int):
    """
    Return the cached match between fragments a and b expressed in the
    (a, b) orientation - i.e. result['frag_i'] == fragments[a]['id'].
    Returns None when no match is cached or the cached value is None.
    """
    if a == b:
        return None
    lo, hi = (a, b) if a < b else (b, a)
    cached = cache.cache.get((lo, hi))
    if cached is None:
        return None
    if cached["frag_i"] == fragments[a]["id"]:
        return cached
    # Inverse orientation - mirror R, t, edge indices.
    R_inv = cached["R"].T
    t_inv = -R_inv @ cached["translation"].reshape(2)
    out = dict(cached)
    out["R"] = R_inv
    out["translation"] = t_inv
    out["frag_i"] = fragments[a]["id"]
    out["frag_j"] = fragments[b]["id"]
    out["edge_i"], out["edge_j"] = cached["edge_j"], cached["edge_i"]
    if "matched_a" in cached and "matched_b" in cached:
        out["matched_a"], out["matched_b"] = cached["matched_b"], cached["matched_a"]
    return out


# ---------------------------------------------------------------------------
# Per-edge partner ranking
# ---------------------------------------------------------------------------

def rank_edges_per_fragment(fragments: list[dict],
                             candidates: list[tuple[int, int]],
                             cache,
                             top_k: int = 8
                             ) -> dict[tuple[int, int], list[dict]]:
    """
    Build, for each (frag_idx, edge_idx) torn-edge identifier, a sorted
    list of its top-K candidate partners across other fragments.

    The cache supplies the BEST edge pairing for each fragment pair (the
    matcher's match_pair() returns the lowest-fit_cost edge pairing, not a
    full edge x edge grid). That's enough to rank per-edge: each fragment
    pair contributes one ranked entry to one source-edge's list.

    Returned format:
        rankings[(fa_idx, ea_idx)] = [
            {
                "rank":           int,                 # 1-based
                "partner_frag":   int (idx in fragments[]),
                "partner_edge":   int,
                "fit_cost":       float,
                "confidence":     float,
                "match_key":      [min_id, max_id],
            }, ...
        ]

    Lower fit_cost is better. Edges with no entries don't appear in the dict.
    """
    pairs_by_edge: dict[tuple[int, int],
                          list[tuple[float, float, int, int, tuple[int, int]]]] = {}
    for (i, j) in candidates:
        m = cache.match(i, j)
        if m is None:
            continue
        ea = int(m.get("edge_i", -1))
        eb = int(m.get("edge_j", -1))
        if ea < 0 or eb < 0:
            continue
        cost = float(m.get("fit_cost", float("inf")))
        conf = float(m.get("confidence", 0.0))
        key = (min(i, j), max(i, j))
        # Edge A's perspective on partner.
        pairs_by_edge.setdefault((i, ea), []).append((cost, -conf, j, eb, key))
        # Edge B's perspective on partner.
        pairs_by_edge.setdefault((j, eb), []).append((cost, -conf, i, ea, key))

    rankings: dict[tuple[int, int], list[dict]] = {}
    for (frag_idx, edge_idx), entries in pairs_by_edge.items():
        # Rank by ascending cost, break ties by descending confidence.
        entries.sort()
        ranked: list[dict] = []
        for rk, (cost, neg_conf, partner_frag, partner_edge, key) in enumerate(
                entries[:top_k], start=1):
            ranked.append({
                "rank":         int(rk),
                "partner_frag": int(partner_frag),
                "partner_edge": int(partner_edge),
                "fit_cost":     round(float(cost), 3),
                "confidence":   round(float(-neg_conf), 4),
                "match_key":    [int(key[0]), int(key[1])],
            })
        rankings[(frag_idx, edge_idx)] = ranked
    return rankings


# ---------------------------------------------------------------------------
# Mutual-rank scoring per fragment pair
# ---------------------------------------------------------------------------

def _rank_lookup(rankings: dict[tuple[int, int], list[dict]],
                 frag_idx: int, edge_idx: int,
                 partner_frag: int, partner_edge: int) -> int | None:
    """Return the 1-based rank of (partner_frag, partner_edge) in
    rankings[(frag_idx, edge_idx)]; None if not present."""
    lst = rankings.get((frag_idx, edge_idx))
    if lst is None:
        return None
    for r in lst:
        if (r["partner_frag"] == partner_frag and
                r["partner_edge"] == partner_edge):
            return int(r["rank"])
    return None


def build_pair_mutual_scores(rankings: dict[tuple[int, int], list[dict]],
                              cache,
                              fragments: list[dict]
                              ) -> dict[tuple[int, int], dict]:
    """
    For every cached pair, compute a mutual-rank score. The score weights
    a pair higher when both sides rank each other near the top.

        mutual_score = 1/(1 + rank_a) + 1/(1 + rank_b)        # best == 2
                       + 0.25 * confidence                    # tiebreaker

    A 'mutual #1' pair scores ~1.0 + 1.0 + 0.25 = 2.25, easily separated
    from a pair where one side ranks the other 5th and gets ~0.17.

    Pairs only show up here if both directions have a non-None match.
    """
    mutual: dict[tuple[int, int], dict] = {}
    for (lo, hi), m in cache.cache.items():
        if m is None:
            continue
        ea = int(m.get("edge_i", -1))
        eb = int(m.get("edge_j", -1))
        if ea < 0 or eb < 0:
            continue
        # Recover (lo, hi) -> (a_idx, b_idx) mapping for the stored match.
        if m["frag_i"] == fragments[lo]["id"]:
            a_idx, b_idx = lo, hi
            edge_a, edge_b = ea, eb
        else:
            a_idx, b_idx = hi, lo
            edge_a, edge_b = ea, eb
        rank_a_in_b = _rank_lookup(rankings, a_idx, edge_a, b_idx, edge_b)
        rank_b_in_a = _rank_lookup(rankings, b_idx, edge_b, a_idx, edge_a)
        if rank_a_in_b is None or rank_b_in_a is None:
            continue
        cost = float(m.get("fit_cost", float("inf")))
        conf = float(m.get("confidence", 0.0))
        score = (1.0 / (1.0 + rank_a_in_b)
                 + 1.0 / (1.0 + rank_b_in_a)
                 + 0.25 * conf)
        mutual[(lo, hi)] = {
            "frag_a":      int(a_idx),
            "frag_b":      int(b_idx),
            "edge_a":      int(edge_a),
            "edge_b":      int(edge_b),
            "rank_a_in_b": int(rank_a_in_b),
            "rank_b_in_a": int(rank_b_in_a),
            "fit_cost":    round(cost, 3),
            "confidence":  round(conf, 4),
            "mutual_score": round(float(score), 4),
        }
    return mutual


# ---------------------------------------------------------------------------
# Seed-candidate ranking (mutual-best pairs at the top)
# ---------------------------------------------------------------------------

def ranked_seed_candidates(mutual: dict[tuple[int, int], dict],
                            profiles: list[dict] | None = None,
                            top_n: int = 12) -> list[dict]:
    """
    Sort the mutual-score table into a seed-candidate list. The MST seeder
    walks this list top-down and tries each in turn; the first one whose
    attach passes the overlap gate becomes the seed.

    A small bonus is awarded when both fragments are 'interior' role -
    those sit deep inside the document and are the safest seeds (corners
    can be ambiguous because their factory edges look alike).
    """
    role_by_id = {}
    if profiles is not None:
        role_by_id = {int(p["id"]): p["role"] for p in profiles}
    items: list[dict] = []
    for (_lo, _hi), v in mutual.items():
        bonus = 0.0
        ra = role_by_id.get(int(v["frag_a"]))
        rb = role_by_id.get(int(v["frag_b"]))
        if ra == "interior" and rb == "interior":
            bonus += 0.05
        elif ra in ("boundary", "corner") and rb in ("boundary", "corner"):
            bonus -= 0.03
        item = dict(v)
        item["seed_score"] = round(float(v["mutual_score"]) + bonus, 4)
        items.append(item)
    items.sort(key=lambda r: -r["seed_score"])
    return items[:top_n]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _key_str(k: tuple[int, int]) -> str:
    return f"{k[0]}_e{k[1]}"


def save_rankings(rankings: dict[tuple[int, int], list[dict]],
                  mutual: dict[tuple[int, int], dict],
                  seeds: list[dict],
                  debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    rk_serial = {_key_str(k): v for k, v in rankings.items()}
    mu_serial = {f"{lo}_{hi}": v for (lo, hi), v in mutual.items()}
    with open(debug_dir / "partner_rankings.json", "w", encoding="utf-8") as fh:
        json.dump({
            "per_edge":    rk_serial,
            "pair_mutual": mu_serial,
            "seed_candidates": seeds,
        }, fh, indent=2)


def print_seed_summary(seeds: list[dict], n: int = 5) -> None:
    """Operator-friendly summary of the top seed candidates."""
    if not seeds:
        print("  -- No mutual-best seeds available - falling back to absolute confidence.")
        return
    print(f"  -- Top {min(n, len(seeds))} seed candidates (mutual-rank):")
    for i, s in enumerate(seeds[:n], start=1):
        print(f"       #{i} frag {s['frag_a']}.e{s['edge_a']} <-> "
              f"frag {s['frag_b']}.e{s['edge_b']}  "
              f"score={s['seed_score']:.2f}  "
              f"ranks=({s['rank_a_in_b']},{s['rank_b_in_a']})  "
              f"conf={s['confidence']:.2f}  cost={s['fit_cost']:.1f}")


__all__ = [
    "rank_edges_per_fragment",
    "build_pair_mutual_scores",
    "ranked_seed_candidates",
    "save_rankings",
    "print_seed_summary",
]
