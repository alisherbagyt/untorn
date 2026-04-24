"""
untorn.config
=============
Central configuration: paths, SAM 2.1 settings, reconstruction parameters.
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
ASSEMBLY_MIN_CONFIDENCE         = 0.55   # threshold to enter the MST queue
ASSEMBLY_ORPHAN_MIN_CONFIDENCE  = 0.45   # relaxed bar for last-resort pass
ASSEMBLY_GLOBAL_ROT_FIX_EVERY   = 5      # apply text-line rotation every N attach
ASSEMBLY_GLOBAL_ROT_MIN_LINES   = 3      # need at least this many placed baselines
ASSEMBLY_GLOBAL_ROT_MIN_DEG     = 0.5    # only apply if misalignment exceeds this
ASSEMBLY_EDGE_LENGTH_RATIO_MAX  = 3.0    # drop candidates whose edges differ by >3x
ASSEMBLY_MAX_CANDIDATE_PAIRS    = 4000   # safety cap on scored pairs
ASSEMBLY_MAX_STEPS              = 1024   # safety cap on MST growth steps

# ─── Reconstruction parameters ─────────────────────────────────────────────

# Contour / support points
POLY_EPSILON_FACTOR = 0.004     # Douglas-Peucker epsilon as fraction of perimeter

# Distance transform matching
DT_SEARCH_RANGE_MAX  = 200
DT_COARSE_STEP       = 8
DT_FINE_STEP         = 2
DT_FINEST_STEP       = 1
DT_BOUNDARY_SAMPLE   = 400
DT_OVERLAP_PENALTY   = 0.35

# Contour adjacency scoring
ADJ_CONTOUR_DILATE_PX = 3

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

# ─── Multi-component matching score ───────────────────────────────────────
MATCH_THRESHOLD         = 1.8     # tightened from 2.5 — too many false positives slipped through
MATCH_MAX_RMS           = 6.0     # tightened from 12 — rms>6 at ~1500px working res is almost always wrong
MIN_TORN_EDGE_PX        = 30.0
MATCH_APPEARANCE_WEIGHT = 0.5

# Reject "direct"-oriented edge matches. For two torn edges from the same
# tear, curvatures MUST be complementary (concave↔convex). A direct match
# means the two edges have identical curvature traversal — geometrically
# impossible for a real seam, so these are always false positives that were
# only getting filtered out downstream by overlap checks.
MATCH_REJECT_DIRECT     = True

# Cluster-to-cluster merges propagate a single bridge match's relative pose
# to every member of the moving cluster. Any rotation/translation error in
# that single M_rel gets amplified across the cluster (a 3° error on a 300px
# fragment displaces corners by ~15 px). To limit the damage we only accept
# cluster-level merges when the bridge's own per-pair rms is very tight —
# i.e. we trust only the strongest matches to perform multi-fragment moves.
# Within-cluster attachments (one fragment joining N already-placed ones)
# still use the full MATCH_MAX_RMS budget.
MATCH_CLUSTER_MAX_RMS   = 3.0

# ─── RANSAC cycle-consistency filter ──────────────────────────────────────
# After pairwise matching we look at every triangle of fragments (a, b, c)
# where all three pairs have candidate matches. The transform chain
# a<-b<-c should equal the direct a<-c match; if it disagrees by more than
# RANSAC_CYCLE_GATE_PX pixels (measured at c's centroid, re-expressed in a's
# frame) the triangle is inconsistent and all three matches get a "con" vote.
# Consistent triangles give all three a "pro" vote. The soft re-rank credits
# stotal by RANSAC_VOTE_BONUS per (pro - con); the hard filter drops matches
# that lose at least RANSAC_DROP_MIN_CON triangles AND whose con count
# exceeds RANSAC_DROP_RATIO x pro count.
RANSAC_ENABLED         = True
# Gate is in PIXELS measured at the fragment centroid after two
# compositions of rigid transforms. Each Procrustes brings ~2-4 px rms at
# the boundary; at a 200-300 px centroid distance a 1-2 deg rotation
# residual translates into ~5-12 px disagreement, and composing two
# transforms doubles that ceiling. For real torn-paper scans with many
# false-positive matches, real triangles often disagree by 30-100 px.
# Set the gate generously -- we use votes as a SOFT re-rank, not a hard
# filter, and blame the worst-rms match in failing triangles.
RANSAC_CYCLE_GATE_PX   = 40.0
RANSAC_VOTE_BONUS      = 0.04   # per net vote, applied to stotal
# Hard-drop rule: only the matches that fail MANY triangles (and look
# like repeat offenders) get dropped. Everything else is kept and merely
# re-ranked.
RANSAC_DROP_MIN_CON    = 6
RANSAC_DROP_RATIO      = 4.0

# ─── Signed-distance-transform pair gate (Richter §8.5.2 / §8.5.5) ────────
# After Procrustes produces a candidate (R, t) for a pair of fragments,
# we check the physical implications: does fragment B sit *inside*
# fragment A's foreground (overlap)? And do the matched seam endpoints
# actually coincide (gap)? These are cheap SDT lookups that catch
# "right-curvature wrong-edge" false positives the curvature score can't
# distinguish.
#
# Thresholds are tuned for working resolution (~1200-1600 px long side).
#  * overlap frac : fraction of B's contour points landing strictly
#                   inside A's foreground. Noise allows a few points, but
#                   > 20% is a real penetration.
#  * overlap depth: a point can be 1-2 px "inside" just from rounding;
#                   real overlap shows mean depth many pixels in.
#  * seam gap     : Procrustes minimises RMS at the matched points, so
#                   median residual > 8 px means the fit itself is poor.
SDT_OVERLAP_FRAC_THRESH  = 0.20
SDT_OVERLAP_DEPTH_THRESH = 10.0
SDT_SEAM_GAP_THRESH_PX   = 8.0

# ─── Reconstruction overlap + rotation checking ───────────────────────────
RECON_OVERLAP_THRESH   = 0.08
RECON_MAX_STALLS       = 8
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

# ─── Hierarchical reconstruction (neighbor graph & priorities) ────────────
#
# NEIGHBOR_K is the kNN graph density on fragment centroids; for a typical
# torn-document scan each piece has 4-8 true neighbors, so k=6 keeps recall
# high without flooding the candidate-match cache.
NEIGHBOR_K                  = 6

# Max boundary-to-boundary distance (in working-resolution pixels) for a
# pair to be considered proximity-neighbors. Real adjacent torn pieces are
# typically within a few tens of pixels. At 1500 px working resolution,
# 200 px is a generous ceiling that still prunes obvious non-neighbors.
NEIGHBOR_MAX_EDGE_DIST_PX   = 200.0

# Two edges are considered to "face" each other if both outward normals
# have a positive cosine with the centroid offset. 0.0 means ">=90 deg".
# A small positive threshold kills narrow-miss configurations where the
# normals are only barely facing.
FACING_COSINE_MIN           = 0.1

# Confidence thresholds (normalized by CONFIDENCE_STOTAL_SPAN; see below):
#   HIGH  : auto-lock a merge when its confidence is at least this.
#   PERIM : minimum confidence to include a piece in the perimeter frame.
#   INT   : minimum confidence for interior infill. Pieces below this go
#           through the final force-fit pass.
HIGH_CONFIDENCE_LOCK        = 0.95
PERIMETER_MIN_CONFIDENCE    = 0.55
INTERIOR_MIN_CONFIDENCE     = 0.40

# confidence = clip(1 - stotal / CONFIDENCE_STOTAL_SPAN, 0, 1)
# stotal lies in [0, ~3.5]. Excellent matches have stotal ~0.3-0.8, so a
# span of 2.5 maps that range onto [0.68, 0.88]. A match with stotal>=2.5
# gets confidence 0.
CONFIDENCE_STOTAL_SPAN      = 2.5

# Minimum factory-edge length (pixels) for a piece to be treated as a
# perimeter candidate. Filters out pieces whose straight-edge
# classification is just noise on a short segment.
PERIMETER_MIN_FACTORY_PX    = 40.0

# Document aspect-ratio sanity bounds.  A4 = 1.414, Letter = 1.294,
# legal = 1.647. Real documents beyond 2.5 are very uncommon.
DOC_MIN_ASPECT              = 1.0
DOC_MAX_ASPECT              = 2.5

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
# initial estimate). ICP can snap into nearby local minima if the
# two contours' shapes happen to "slide" past each other; a refined
# rotation more than this many degrees off the initial estimate is
# almost always spurious, so we fall back to the initial Procrustes.
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

# ─── Cluster jitter — post-placement snug-up ──────────────────────────────
# After Phase V interior infilling lays pieces down at their individual
# best pose, small drift accumulates along chains. The cluster-jitter pass
# takes each non-corner placed fragment and re-runs ICP against the CON-
# CATENATED boundaries of its already-placed neighbors, yielding a small
# pose correction that pulls the whole arrangement tighter.
CLUSTER_JITTER_ITERS        = 2
# Maximum translation (px) the jitter step may move any single fragment.
# This is a safety cap — if ICP wants to shift a piece further than this,
# something upstream picked the wrong neighborhood.
CLUSTER_JITTER_TOL_PX       = 20.0

# A second, wider jitter sweep runs AFTER bundle adjustment. By that point
# fragments sit in a globally-consistent arrangement; this pass uses a
# wider correspondence radius and more iterations to close the final
# sub-pixel-to-few-pixel residual gaps.
CLUSTER_JITTER_ITERS_FINAL  = 4
CLUSTER_JITTER_TOL_PX_FINAL = 15.0
CLUSTER_JITTER_WIDE_DIST_PX = 18.0

# ─── Global pose-graph bundle adjustment (LM-style) ───────────────────────
# Every placement so far optimised ONE edge pair at a time. Each piece's
# final pose only satisfies its anchoring neighbor; the other seams that
# piece shares with adjacent pieces stay slightly open. Bundle adjustment
# treats the collection of (i, j) edge-pair matches as a POSE-GRAPH:
# every cached match contributes a "seam-point coincidence" constraint
# (matched_a in A's local frame under T_A must equal matched_b in B's
# local frame under T_B), and we jointly re-solve for every non-corner
# fragment's (θ, tx, ty). This closes gaps that single-pair optimisation
# cannot see.
#
# Corners are PINNED (their scan-position anchor defines the global gauge
# frame). Per-fragment drift is bounded by BA_MAX_{ROTATION_DEG, TRANSLATION_PX}
# as a safety cap — large LM moves are almost always a sign that a
# spurious cached match is dominating the least-squares cost.
BA_ENABLED                  = True
BA_MAX_ITER                 = 200
BA_FUNC_TOL                 = 1e-6
BA_MAX_ROTATION_DEG         = 8.0
BA_MAX_TRANSLATION_PX       = 60.0

# ─── Orphan rescue ────────────────────────────────────────────────────────
# After force-fit, any fragment left at identity (= scan position) drops
# all pretense of "reconstruction" for that piece. Orphan-rescue makes
# one final attempt per unplaced fragment: evaluate it against EVERY
# placed fragment (not just its neighbor graph) and attach at the lowest
# fit_cost placement that passes the global-overlap check. This catches
# fragments whose true partners weren't reachable through the proximity-
# graph (e.g. the scan layout had them far from their real neighbors).
ORPHAN_RESCUE_ENABLED       = True
# Upper bound on fit_cost for an orphan rescue attach. Looser than the
# main MAX_ATTACH_COST because orphans are, by definition, hard cases
# and we'd rather have a slightly-worse touch than no touch at all.
ORPHAN_MAX_ATTACH_COST      = 700.0

# Inpainting
INPAINT_RADIUS = 5

# ─── Helpers ───────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create project directories if they don't exist."""
    for d in [INPUT_DIR, OUTPUT_DIR, DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
