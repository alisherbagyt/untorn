"""
untorn.config
=============
Central configuration: paths, SAM 2.1 settings, reconstruction parameters.

Post-overhaul layout (Stage 8):
  * Paths / SAM / background detection / fragment limits     -- unchanged
  * Text-line + boundary + DINOv2 feature extraction          -- new
  * Four-gate matching config + confidence weights            -- new
  * Assembly (layout-agnostic MST growth)                     -- new
  * Composition (supersampled warp + LAB harmonise)           -- new
  * Gap fill (hole classification thresholds)                 -- new
  * Legacy reconstruction knobs still consumed by matching.py
    (Smith-Waterman, SDT gate, ICP, FIT_*, BA_*, etc.)        -- kept

Dead knobs removed in Stage 8 (all confirmed unreferenced in *.py):
    NEIGHBOR_K, NEIGHBOR_MAX_EDGE_DIST_PX, HIGH_CONFIDENCE_LOCK,
    PERIMETER_MIN_CONFIDENCE, INTERIOR_MIN_CONFIDENCE,
    PERIMETER_MIN_FACTORY_PX, DOC_MIN_ASPECT, DOC_MAX_ASPECT,
    MATCH_THRESHOLD, MATCH_CLUSTER_MAX_RMS, ADJ_CONTOUR_DILATE_PX,
    DT_SEARCH_RANGE_MAX / DT_COARSE_STEP / DT_FINE_STEP /
    DT_FINEST_STEP / DT_BOUNDARY_SAMPLE / DT_OVERLAP_PENALTY,
    RANSAC_ENABLED / RANSAC_CYCLE_GATE_PX / RANSAC_VOTE_BONUS /
    RANSAC_DROP_MIN_CON / RANSAC_DROP_RATIO,
    CLUSTER_JITTER_ITERS / CLUSTER_JITTER_TOL_PX /
    CLUSTER_JITTER_ITERS_FINAL / CLUSTER_JITTER_TOL_PX_FINAL /
    CLUSTER_JITTER_WIDE_DIST_PX,
    BA_ENABLED, ORPHAN_RESCUE_ENABLED, INPAINT_RADIUS, RECON_MAX_STALLS.
"""

import os
import sys
from pathlib import Path

# ─── Project directories ───────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_DIR    = DATA_DIR / "input"
OUTPUT_DIR   = DATA_DIR / "output"
DEBUG_DIR    = DATA_DIR / "debug"

# ─── Working resolution ────────────────────────────────────────────────────
# Phase 0 downscales any input whose longer side exceeds this. 1500 px keeps
# SAM 2.1 comfortable on 4 GB VRAM at points_per_side=32. Lift to 2000 if
# you have ≥ 6 GB and need more detail; the rest of the pipeline scales.
WORKING_MAX_DIM = 1500

# ─── SAM 2.1 ───────────────────────────────────────────────────────────────

SAM2_REPO_DIR   = str(PROJECT_ROOT / "sam2")
SAM2_CHECKPOINT = str(PROJECT_ROOT / "sam2" / "sam2.1_hiera_small.pt")
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Fix import shadowing: the repo root must be on sys.path
if SAM2_REPO_DIR not in sys.path:
    sys.path.insert(0, SAM2_REPO_DIR)

# ─── SAM AMG parameters ────────────────────────────────────────────────────
# Tuned for large paper fragments; appearance filtering now handles the rest.

SAM2_POINTS_PER_SIDE  = 32     # dense grid for thorough coverage
SAM2_PRED_IOU_THRESH  = 0.80   # slightly relaxed — appearance filter is the gate
SAM2_STABILITY_THRESH = 0.88   # slightly relaxed — same reason
SAM2_MIN_MASK_AREA    = 500    # absolute pixel floor (pre-filter noise)
SAM2_CROP_N_LAYERS    = 1      # one crop layer for multi-scale (catches tiny fragments)
SAM2_CROP_N_POINTS    = 1      # downscale factor for crop points

# ─── Background auto-detection ────────────────────────────────────────────

# Fraction of image width/height to sample from each corner
BG_CORNER_SAMPLE_FRAC = 0.05

# Minimum LAB ΔE distance from background for a mask to be paper candidate.
# LAB ΔE: ~2.3 is "just noticeable", ~10 is clearly different color.
# 18 comfortably separates white paper from dark grey, wood from white paper,
# and brown paper from white/pink scanner bed.
# Raise if background is very close in color to paper (rare).
BG_DIST_LAB_THRESH = 18.0

# ─── Fragment size thresholds (relative to image area) ────────────────────
# Using fractions instead of absolute pixels — DPI-agnostic.

# Minimum fragment area as fraction of total image pixels.
# 0.0003 = 0.03% of a 10 Mpx image ≈ 3000 px  (catches tiny crumbs in Image 3)
MIN_FRAGMENT_AREA_FRAC = 0.0003

# Maximum fragment area as fraction of total image pixels.
# 0.90 = anything covering 90%+ of the image is the true background mask.
# Large telegram fragments (Image 4) cover ~35% each — well within this bound.
MAX_FRAGMENT_AREA_FRAC = 0.90

# ─── Hole filling ──────────────────────────────────────────────────────────

# Fraction of a background-colored mask that must lie inside a paper mask
# for it to be treated as a physical hole and filled in.
HOLE_CONTAINMENT_THRESH = 0.85

# ─── General fragment limits ───────────────────────────────────────────────

MAX_FRAGMENTS = 40   # raised from 20 — Image 1 has ~30 fragments

# ─── Text-line detection ──────────────────────────────────────────────────
# Per-fragment baseline detection uses a rotation sweep + projection
# profile peak-finder on the ink-only mask. Baselines feed two things
# downstream: (a) a matching gate that rewards text continuity across a
# proposed seam, and (b) a global rotation prior so the assembled cluster
# ends up with horizontal text.

TEXT_LINE_ANGLE_SEARCH_DEG = 30.0    # +- angular search window for baseline tilt
TEXT_LINE_ANGLE_STEP_DEG   = 2.0     # angular step during sweep
TEXT_LINE_MIN_INK_FRAC     = 0.02    # row must have >=2% ink pixels to be a peak
TEXT_LINE_INK_GRAYSCALE_MAX = 140    # pixels darker than this count as ink
TEXT_LINE_MAX_Y_DISC_PX    = 3.0     # vertical discontinuity cap (continuity gate)
TEXT_LINE_MAX_ANGLE_DISC_DEG = 8.0   # angular discontinuity cap
TEXT_LINE_SEAM_RADIUS_PX   = 20.0    # how close a line must run to the seam

# ─── Sub-pixel boundary refinement ─────────────────────────────────────────
# SAM 2.1 masks are integer-pixel, so raw boundary points have +-1 px
# jitter that dominates curvature for gentle tears. We snap each boundary
# point to the local gradient-magnitude maximum along its inward normal,
# within a small band. Gives genuinely sub-pixel contours.

BOUNDARY_REFINE_ENABLED    = True
BOUNDARY_GRADIENT_BAND_PX  = 5       # search +-this many px along normal
BOUNDARY_SMOOTH_SIGMA      = 1.0     # Gaussian smoothing on image before gradient
BOUNDARY_STEP_PX           = 0.5     # sub-pixel sampling step along normal

# ─── DINOv2 dense-feature appearance gate ──────────────────────────────────
# Small Vision Transformer (ViT-S/14, ~22M params, ~84 MB checkpoint, ~0.8 GB
# VRAM when active) used purely as a frozen feature extractor. For every
# fragment we cache a (H_p, W_p, D) dense patch-token map over its bbox
# crop; for every proposed seam we sample a few patches on each side of the
# seam in canvas space, map them back into the two fragments' crops, and
# cosine-similarity the feature vectors. Same paper + same ink regime →
# cosine ~0.7+; different paper / different fragment source → ~0.3.

DINOV2_ENABLED             = True
DINOV2_MODEL               = "dinov2_vits14"   # ~90 MB when loaded
DINOV2_PATCH_SIZE          = 14
DINOV2_INPUT_SIZE          = 224               # 16 x 16 token grid
DINOV2_DEVICE              = "cuda"            # falls back to CPU if unavailable
DINOV2_SEAM_N_PATCHES      = 8                 # samples per side of the seam
DINOV2_SEAM_PATCH_OFFSET_PX = 10.0             # how far from seam to sample

# ─── Matching gates (layout-agnostic, 4-cascade) ────────────────────────────
# The overhauled matcher runs four cascaded gates per candidate edge pair.
# (A) SW curvature pre-filter, (B) Procrustes + ICP + SDT geometry, (C)
# DINOv2 + strip-NCC appearance, (D) text-line continuity. Confidence is
# a weighted sum of the gate scores. Weights must sum close to 1.0 so the
# score stays interpretable in [0, 1].

MATCH_ICP_DRIFT_DEG        = 15.0    # relaxed from 5 deg; multi-seed Procrustes
                                     # + SDT gate catches bad basins instead.
MATCH_PROCRUSTES_SEEDS     = 3       # try start/middle/end sub-arc windows
MATCH_APPEARANCE_COS_MIN   = 0.55    # DINOv2 seam cosine floor (gate C)
MATCH_TEXT_LINE_MIN_CONT   = 0.30    # text-line continuity floor (gate D)
MATCH_TEXT_LINE_MIN_EXPECT = 2       # below this "expected" count, skip gate D
MATCH_PAPER_COLOR_DELTA_MAX = 12.0   # LAB ΔE prefilter between fragments

# Confidence weights (sum should be ~1.0)
CONF_W_GEOMETRY    = 0.30    # (1 - stotal / CONFIDENCE_STOTAL_SPAN)
CONF_W_APPEARANCE  = 0.20    # DINOv2 seam cosine, mapped to [0, 1]
CONF_W_STRIP_NCC   = 0.20    # existing 8-px strip NCC
CONF_W_TEXT_LINE   = 0.20    # text-line continuity (0 when not applicable)
CONF_W_PAPER_COLOR = 0.10    # 1 - LAB ΔE / MATCH_PAPER_COLOR_DELTA_MAX

# Geometry-score normalisation. confidence(geometry) = clip(1 - stotal /
# CONFIDENCE_STOTAL_SPAN, 0, 1). stotal lies in [0, ~3.5]; excellent
# matches sit at ~0.3-0.8. Span of 2.5 maps that range onto [0.68, 0.88].
CONFIDENCE_STOTAL_SPAN      = 2.5

# ─── Assembly — layout-agnostic MST-growth orchestration ───────────────────
# The new `untorn.assembly.reconstruct` replaces the corner-seeded
# hierarchical orchestrator. It enumerates all torn-edge pairs, scores
# them through the four-gate matcher, seeds the MST from the highest-
# confidence pair, and grows outward by repeatedly attaching the highest
# -confidence match whose one side is already placed. Conflicts (multiple
# free fragments competing for the same anchor) are resolved by
# bipartite max-weight matching, and every few placements we fit a
# global text-line rotation to keep text horizontal.

ASSEMBLY_HIGH_CONFIDENCE_LOCK   = 0.92   # auto-lock (no overlap re-check if seen)
ASSEMBLY_MIN_CONFIDENCE         = 0.45   # threshold to enter the MST queue
                                          # (relaxed from 0.55 - the four-gate
                                          # cascade already drops bad pairs;
                                          # too-strict floor was causing 10/12
                                          # fragments to be left unplaced).
ASSEMBLY_ORPHAN_MIN_CONFIDENCE  = 0.40   # relaxed bar for the standard rescue pass
# NOTE: ASSEMBLY_GLOBAL_ROT_FIX_EVERY, ASSEMBLY_GLOBAL_ROT_MIN_LINES,
# ASSEMBLY_GLOBAL_ROT_MIN_DEG and ASSEMBLY_TEXT_ROTATION_MIN_DEG were removed
# in Step 3 along with the cosmetic post-placement rotation passes. The
# matcher now folds per-fragment text orientation into Procrustes seeding
# (see matching._match_edge_pair "text_prior_used"); no global rotation
# correction is needed because per-fragment alignment is correct by
# construction.

# Localized edge-connection filtering
ASSEMBLY_EDGE_PROXIMITY_PX      = 200.0  # local boundary-distance gate for pair viability
ASSEMBLY_MIN_EDGE_LENGTH_PX     = 40.0   # reject very short torn-edge pairings
ASSEMBLY_MAX_EDGE_LENGTH_RATIO  = 2.5    # drop candidates whose edges differ by >2.5x
ASSEMBLY_EDGE_LENGTH_RATIO_MAX  = ASSEMBLY_MAX_EDGE_LENGTH_RATIO  # backward-compatible alias
ASSEMBLY_MAX_CANDIDATE_PAIRS    = 4000   # safety cap on scored pairs
ASSEMBLY_MAX_STEPS              = 1024   # safety cap on MST growth steps

# Ensure every fragment has at least this many candidate partners scored.
# The per-fragment prefilter (paper-LAB + edge length + grid filter) is too
# aggressive on real-world degraded scans and can leave a fragment with
# zero candidates - in which case it never enters the MST. When the total
# survivor count is below this density floor, assembly falls back to "all
# torn-edge pairs" so every fragment gets a fair shot at the matcher.
ASSEMBLY_MIN_CANDIDATES_PER_FRAG = 3.0

# Mutual-rank seed selection: an edge pair where both sides rank each other
# in the top K is treated as a "mutually best" match - the strongest signal
# we have that the seam is real. The MST seeder prefers these over
# absolute-confidence top picks; the MST grower rewards them with a
# +0.20 confidence bonus during conflict resolution.
ASSEMBLY_MUTUAL_TOP_K           = 3

# Aggressive orphan rescue: after the standard rescue pass, force a match
# attempt between every (orphan, anchor) and every (orphan, orphan) pair
# - even those the prefilter dropped - and accept the lowest-cost fit.
# This is the "leave no fragment behind" pass and is what catches
# fragments whose true partner sat below the candidate prefilter.
ASSEMBLY_AGGRESSIVE_ORPHAN_RESCUE = True
ORPHAN_RESCUE_MIN_CONFIDENCE      = 0.30   # bar for the aggressive pass
ORPHAN_RESCUE_MAX_COST            = 1200.0 # per-pair fit_cost ceiling

# Cluster reconciliation: the aggressive orphan rescue can leave the
# canvas split into multiple disconnected clusters (each anchored at its
# own scan-position identity). The reconciliation phase identifies them
# via the merge_log and tries to bridge any pair of clusters using the
# lowest-cost cached cross-cluster match. The full second cluster is
# then SHIFTED rigidly so the bridge is satisfied.
ASSEMBLY_CLUSTER_RECONCILE        = False
CLUSTER_MERGE_MAX_COST            = 1500.0 # per-bridge fit_cost ceiling
CLUSTER_MERGE_MAX_SHIFT_PX        = 4000.0 # absurd-shift safety guard
# When two clusters merge they share a seam, so some seam-zone mask
# overlap is expected. The MST overlap threshold (RECON_OVERLAP_THRESH)
# is too strict here - we relax it by this factor for cluster bridges.
# 6.0 -> ~50% allowed; the bridge fit_cost filter is the primary gate
# at the cluster-reconciliation stage, not pixel overlap.
CLUSTER_MERGE_OVERLAP_RELAX       = 6.0

# Connection-line ("seam") gates applied at attach-time. The matcher can
# return a (R, t) whose SW correspondences locally agree but whose full-
# length seam either gaps wide, overlaps too much, or only covers a small
# sub-arc of the actual edge. These gates kill those before they pollute
# the MST. Loose enough that real (sub-pixel) drifts stay accepted; tight
# enough that "right curvature, wrong edge" near-misses are rejected.
SEAM_MAX_GAP_PX                 = 3.0    # mean nearest-neighbour gap across
                                          # the seam. Tightened from 6.0 in
                                          # Step 4 because the new seam_solver
                                          # gets seams sub-pixel-tight; a
                                          # 6 px residual now means a real
                                          # mismatch, not just ICP slack.
SEAM_MAX_OVERLAP_FRAC           = 0.18   # fraction of edge points overlapping
SEAM_MIN_COVERAGE               = 0.55   # arc-length coverage of both edges

# ─── Grid / binary fast-filter (Phase 3, Step 9) ──────────────────────────
# Optional pre-screening layer that runs before the SW + Procrustes + ICP
# matcher and the Siamese gate. For each fragment we extract a 20-px band
# inside every torn edge, Otsu-binarize, tile into 16x16 blocks, compute a
# uniform-LBP histogram per block, and run a 2-point rigid-transform
# RANSAC over block correspondences to score pair plausibility. The top-K
# survivors per fragment are intersected with the existing edge-length +
# paper-color prefilter — only pairs that survive BOTH gates reach the
# expensive matcher.
#
# When disabled the assembly path falls back to the legacy paper-color
# prefilter only, so the filter is safe to toggle off if a regression
# is suspected.
GRID_FILTER_ENABLED         = True
GRID_FILTER_TOP_K           = 8     # top-K partner indices kept per fragment
GRID_FILTER_BAND_DEPTH_PX   = 20    # band depth (px) inward from torn edge
GRID_FILTER_BLOCK_SIZE      = 16    # tile size; blocks are block_size x block_size
GRID_FILTER_LBP_P           = 8     # LBP neighbours
GRID_FILTER_LBP_R           = 1     # LBP radius
GRID_FILTER_TOP_BLOCK_NN    = 3     # nearest-neighbour width per query block
GRID_FILTER_RANSAC_ITERS    = 64    # 2-point RANSAC iterations per pair
GRID_FILTER_RANSAC_TOL_PX   = 8.0   # inlier tolerance in pixels
GRID_FILTER_MIN_INLIERS     = 3     # minimum spatially-consistent block matches

# ─── Siamese edge matcher (Phase 4, Steps 11-12) ──────────────────────────
# A trained CNN (EdgeMatcher, ~520 K params) added as the FIFTH gate in the
# matching cascade. It runs after Procrustes + ICP refinement and BEFORE the
# SDT physical gate, scoring whether the two aligned torn edge strips
# visually match. The model is loaded once at pipeline startup (see
# untorn.edge_matcher.load) and queried in untorn.matching._match_edge_pair.
#
# Why 0.985 instead of the plan's 0.55: the trained checkpoint sits at
# temperature ~10 (logit_scale init=log(10)), so positive examples cluster
# very near probability 1.0 and negatives near 0.0. Eval (val.h5, 8 K pairs)
# gave AUC=0.924, with the F1-max threshold at 0.9886 and the
# precision@recall=0.9 threshold at 0.9936. 0.985 sits between the two,
# slightly toward the F1 side so we don't reject too many true matches.
#
# The full 5-gate cascade (SW + Procrustes + ICP + Siamese + SDT) gives the
# Siamese gate a job: catch "right-curvature wrong-edge" false positives
# that survive geometry but visually don't belong together. Real matches
# should sail through; the gate is the cheapest learned check we have to
# kill that final class of geometric near-misses.
#
# Set EDGE_MATCHER_ENABLED = False or move the checkpoint out of the way to
# fall back to the legacy 4-gate cascade with no other code changes.
EDGE_MATCHER_ENABLED        = True
EDGE_MATCHER_CHECKPOINT     = "models/edge_matcher.pt"
EDGE_MATCHER_DEVICE         = "cuda"     # falls back to CPU if unavailable
# Relaxed from the train-time-derived 0.985 to 0.55. The high threshold was
# tuned on the validation distribution of the training data; on real-world
# scans it discards most true matches (visual jitter, paper colour drift,
# different ink). 0.55 still catches obvious negatives (the train-time
# AUC-ROC at 0.5 was 0.93) without nuking the MST. Bump back up if false
# positives become the dominant failure mode again.
EDGE_MATCHER_MIN_SCORE      = 0.55
EDGE_MATCHER_POSE_WEIGHT    = 0.0        # optional pose-blend weight (unused
                                          # at inference today; reserved for
                                          # future ICP warm-start work).

# ─── Composition (Phase 4) — polygon-clip / Step 6 ─────────────────────────
# The composition runs a SUPERSAMPLED warp of each fragment's mask, RGB
# crop and *interior SDT* onto a shared canvas. At every canvas pixel
# the WINNER is the fragment with the LARGEST interior-edge-distance
# (i.e. the pixel that lies "deepest" inside one of the fragments —
# overlapping seams resolve to the fragment whose boundary is farther
# away). One pixel = one fragment, no feathered alpha blending.
#
# Small seam-zone gaps (uncovered pixels within ``COMP_SEAM_FILL_MAX_PX``
# of a placed fragment) are filled by a Voronoi-of-fragments lookup —
# each gap pixel takes the colour of its nearest covered pixel. Larger
# uncovered regions are forwarded to ``gap_fill`` as a true-hole mask.
#
# This replaces the prior feathered-alpha policy, which hid (rather than
# eliminated) seam gaps, and the smaller-on-top z-order, which let tiny
# crumbs paint over the centres of large fragments.
COMP_SUPERSAMPLE                = 2       # 2x warp canvas, downsampled with INTER_AREA
COMP_LAB_HARMONISE_ENABLED      = True
COMP_INK_THRESH                 = 140     # L below this is treated as ink
COMP_SEAM_FILL_MAX_PX           = 6.0     # uncovered pixels within this many
                                           # px of coverage are Voronoi-filled
                                           # from the nearest covered pixel

# ─── Gap fill / hole classification (Phase 5) ──────────────────────────────
# After composition we classify each connected hole as small / medium /
# large (as a fraction of the document bbox area) and feed a unified
# scar+hole mask to LaMa. Small holes ride along with the normal seam
# scar. Medium holes get their context expanded so LaMa has enough
# surrounding paper/text to hallucinate a plausible fill. Large holes
# mean a real missing fragment — LaMa is run best-effort and the hole
# shows up in `pipeline_meta.json` so the caller can surface the fact.
GAP_SMALL_FRAC                  = 0.005   # holes below this frac -> regular scar
GAP_MEDIUM_FRAC                 = 0.05    # holes below this frac -> expanded context
GAP_LARGE_CONTEXT_EXPAND_PX     = 20
GAP_EDGE_TOUCH_PX               = 4       # holes within this many px of canvas border
                                          # are treated as edge holes (not missing frag)

# ─── Reconstruction parameters ─────────────────────────────────────────────

# Contour / support points
POLY_EPSILON_FACTOR = 0.004     # Douglas-Peucker epsilon as fraction of perimeter

# ─── Curvature feature strings ────────────────────────────────────────────
CURV_N_SAMPLES     = 80
CURV_SMOOTH_WINDOW = 2
CURV_MIN_STD       = 0.01

# ─── Smith-Waterman alignment ─────────────────────────────────────────────
SW_MATCH_SCORE   =  2.0
SW_CLOSE_PENALTY = -0.1
SW_FAR_PENALTY   = -2.0
SW_GAP_PENALTY   = -1.0
SW_EPSILON_1     =  0.05
SW_EPSILON_2     =  0.20
SW_MIN_SCORE     =  5.0
SW_MIN_ALIGNED   =  6

# ─── Match geometry guards ────────────────────────────────────────────────
# rms>6 at ~1500px working res is almost always a false positive.
MATCH_MAX_RMS           = 6.0
MIN_TORN_EDGE_PX        = 30.0
MATCH_APPEARANCE_WEIGHT = 0.5

# Reject "direct"-oriented edge matches. For two torn edges from the same
# tear, curvatures MUST be complementary (concave↔convex). A direct match
# means the two edges have identical curvature traversal — geometrically
# impossible for a real seam, so these are always false positives that were
# only getting filtered out downstream by overlap checks.
MATCH_REJECT_DIRECT     = True

# ─── Signed-distance-transform pair gate (Richter §8.5.2 / §8.5.5) ────────
# After Procrustes produces a candidate (R, t) for a pair of fragments,
# we check the physical implications: does fragment B sit *inside*
# fragment A's foreground (overlap)? And do the matched seam endpoints
# actually coincide (gap)? These are cheap SDT lookups that catch
# "right-curvature wrong-edge" false positives the curvature score can't
# distinguish.
SDT_OVERLAP_FRAC_THRESH  = 0.20
SDT_OVERLAP_DEPTH_THRESH = 10.0
SDT_SEAM_GAP_THRESH_PX   = 8.0

# ─── Reconstruction overlap + rotation checking ───────────────────────────
RECON_OVERLAP_THRESH   = 0.08
RECON_MAX_ROTATION_DEG = 30.0    # synthetic benchmark scatters fragments up to ±22.5°,
                                 # so pairwise relative rotations go to ~22.5°.
                                 # Cap at 30° keeps real matches in while still
                                 # excluding gross mis-orientations. The SDT
                                 # physical gate (§8.5) rejects wrong-edge
                                 # candidates even when their rotation is small.

# Ceiling for the dynamically-sized overlap canvas. warpAffine memory
# scales with canvas area; anything beyond this is pathological and the
# overlap check degrades to "accept".
OVERLAP_CANVAS_MAX = 12000

# Two edges are considered to "face" each other if both outward normals
# have a positive cosine with the centroid offset. 0.0 means ">=90 deg".
# A small positive threshold kills narrow-miss configurations where the
# normals are only barely facing.
FACING_COSINE_MIN           = 0.1

# ─── Iterative closest point (ICP) jitter correction ──────────────────────
# Runs after Procrustes to micro-adjust the rigid transform so that text
# strokes crossing the tear line up exactly. Correspondences whose
# nearest-neighbor distance exceeds ICP_MAX_CORR_DIST_PX are treated as
# outliers and dropped in that iteration.
ICP_MAX_ITER                = 15
ICP_MAX_CORR_DIST_PX        = 6.0
# Two-phase ICP: a coarse pass pulls drifted edges together with a wide
# correspondence tolerance, then a fine pass (using ICP_MAX_CORR_DIST_PX)
# tightens the snug-up. The coarse tolerance is set high enough to bridge
# the ~20-30 px gap a pure Procrustes alignment often leaves along the
# off-diagonal stretches of a long torn edge, without pulling so hard
# that it folds back onto an unrelated part of the contour.
ICP_COARSE_DIST_PX          = 25.0
# ICP drift ceiling (degrees of rotation away from the Procrustes
# initial estimate). Paired with MATCH_ICP_DRIFT_DEG; matching.py takes
# the max of the two so the new 15° regime supersedes the historic 5°
# hard cap when the overhauled gates are active.
ICP_MAX_DRIFT_DEG           = 5.0

# ─── Full-edge physical-fit evaluator ─────────────────────────────────────
# After rigid alignment, we score how well the two edges actually MEET
# by sampling the entire edge polyline (not just the SW-matched sub-arc)
# and measuring three quantities:
#   fit_overlap_px   : total pixels of fragment B's body that penetrate
#                       fragment A's interior (SDT lookup).
#   fit_gap_px       : mean nearest-neighbour distance from every point on
#                       edge A to warped edge B (and vice versa).
#   fit_coverage     : fraction of edge-A arc-length whose nearest warped-B
#                       point is within COVERAGE_TOLERANCE_PX, averaged with
#                       the symmetric coverage of edge B.
#
# fit_cost is a single scalar combining these three, used to rank
# candidate matches (lower = better). The weights below were picked so
# that at working resolution (~1500 px long side):
#   - 1 px of overlap or gap contributes ~1 unit of cost
#   - losing 10 percentage points of edge coverage contributes ~5 units
# which keeps "minimal overlap" and "maximal coverage" on roughly equal
# footing, matching the user's stated priorities.
FIT_W_OVERLAP               = 1.5
FIT_W_GAP                   = 2.0
FIT_W_UNCOVERED             = 80.0
# Distance threshold (px) below which an edge-A sample is counted as
# "covered" by warped edge B. 2 px is tight enough to call a seam "touching"
# at working resolution but loose enough to survive sub-pixel rounding.
COVERAGE_TOLERANCE_PX       = 2.0
# Upper bound on fit_cost for a match to be accepted at all. Anything
# above this is structurally hopeless (either massive overlap or massive
# gap) and should never be considered, regardless of its SW score.
MAX_ATTACH_COST             = 400.0

# ─── Global pose-graph bundle adjustment (LM-style) ───────────────────────
# The assembly's final pass treats cached (i, j) pair matches as a
# POSE-GRAPH and jointly re-solves every non-seed fragment's (θ, tx, ty).
#
# Step 5 — contact-constrained BA: residuals come from two sources per
# placed pair:
#   * SW seam-coincidence — sparse, ~10 correspondences per pair, the
#     legacy term that anchored the previous BA.
#   * Dense edge correspondences — BA_DENSE_EDGE_SAMPLES samples per
#     torn edge resampled at equal arc length; each warped sample of B
#     is told to coincide with its nearest warped sample of A (and vice
#     versa via the symmetric residual). This is what closes seam gaps
#     globally — dropping an edge from 5 SW correspondences to ~50
#     contact constraints turns the seam from "two pinned points" into
#     "two welded polylines".
#
# The dense term costs one cKDTree per pair per LM iteration; on a 30-
# fragment scene with 30 placed pairs and ~200 iterations that's ~6k
# KDTree queries — sub-second in practice.
BA_MAX_ITER                 = 200
BA_FUNC_TOL                 = 1e-6
BA_MAX_ROTATION_DEG         = 8.0
BA_MAX_TRANSLATION_PX       = 60.0
BA_DENSE_EDGE_ENABLED       = True
BA_DENSE_EDGE_SAMPLES       = 32      # samples per torn edge (each side)
BA_DENSE_EDGE_WEIGHT        = 0.6     # relative weight vs. SW correspondences

# ─── Seam solver — post-MST edge-contact pose refinement (Step 4) ────────
# After the MST grows, every adjacent placed pair has a pose derived from
# Procrustes + ICP on a sub-arc of the matched correspondences. The seam
# solver runs a small derivative-free optimisation on (Δθ, Δdx, Δdy) for
# the attached fragment, minimising `evaluate_edge_fit.fit_cost` (gap +
# overlap + uncovered) plus an explicit penalty on absolute pixel-level
# overlap. Hard caps prevent the solver from drifting into a different
# basin; a small improvement threshold guards against pure-noise updates.
SEAM_SOLVER_ENABLED         = True
SEAM_SOLVER_MAX_ITER        = 40       # Nelder-Mead simplex evaluations
SEAM_SOLVER_LAMBDA_OVERLAP  = 0.6      # weight on absolute SDT-penetration px
SEAM_SOLVER_MAX_DRIFT_DEG   = 2.0
SEAM_SOLVER_MAX_DRIFT_PX    = 5.0
SEAM_SOLVER_MIN_IMPROVEMENT = 0.5      # only accept refinement if cost drops
                                        # by at least this many units

# ─── Orphan rescue ────────────────────────────────────────────────────────
# After MST growth, any fragment left unplaced drops all pretense of
# "reconstruction" for that piece. Orphan-rescue makes one final attempt
# per unplaced fragment: evaluate it against every placed fragment (not
# just the original candidate pairs) and attach at the lowest fit_cost
# placement that passes the global-overlap check. This catches fragments
# whose true partners weren't reachable through the main MST.
# Looser ceiling than MAX_ATTACH_COST because orphans are, by definition,
# hard cases and we'd rather have a slightly-worse touch than no touch.
ORPHAN_MAX_ATTACH_COST      = 700.0

# ─── Helpers ───────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create project directories if they don't exist."""
    for d in [INPUT_DIR, OUTPUT_DIR, DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
