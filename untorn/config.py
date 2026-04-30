"""untorn.config — central configuration.

Layout (engine rebuild April 2026):
  * Paths + working resolution        (untouched chassis)
  * SAM 2.1 + segmentation knobs      (untouched chassis)
  * Text-line + boundary refinement   (feature feeders for matcher)
  * DINOv2 appearance feeder          (optional, used by matcher gate)
  * ENGINE: matching, assembly, BA, seam_solver, edge_matcher
  * Composition + gap fill            (untouched chassis)

Engine knob count: ~30. Anything not in this file is hard-coded inside
the engine module that owns it (e.g. matching's _BOUNDARY_PROX_PX,
assembly's _CANVAS_PAD_PX). The 70+ knobs from prior overhauls
(GRID_FILTER_*, ASSEMBLY_*, ORPHAN_*, CLUSTER_MERGE_*, ...) were
removed because they referred to subsystems that no longer exist.
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

SAM2_POINTS_PER_SIDE  = 32
SAM2_PRED_IOU_THRESH  = 0.80
SAM2_STABILITY_THRESH = 0.88
SAM2_MIN_MASK_AREA    = 500
SAM2_CROP_N_LAYERS    = 1
SAM2_CROP_N_POINTS    = 1

# ─── Background auto-detection ────────────────────────────────────────────

BG_CORNER_SAMPLE_FRAC = 0.05
BG_DIST_LAB_THRESH    = 18.0

# ─── Fragment size thresholds ──────────────────────────────────────────────

MIN_FRAGMENT_AREA_FRAC  = 0.0003
MAX_FRAGMENT_AREA_FRAC  = 0.90
HOLE_CONTAINMENT_THRESH = 0.85
MAX_FRAGMENTS           = 40

# ─── Text-line detection (feature feeder) ──────────────────────────────────
# Per-fragment baseline detection sweeps a +/- TEXT_LINE_ANGLE_SEARCH_DEG
# window; rotated fragments outside that range produce unreliable text
# angles, which the matcher detects by checking text-line count >= 3.

TEXT_LINE_ANGLE_SEARCH_DEG   = 30.0
TEXT_LINE_ANGLE_STEP_DEG     = 2.0
TEXT_LINE_MIN_INK_FRAC       = 0.02
TEXT_LINE_INK_GRAYSCALE_MAX  = 140
TEXT_LINE_MAX_Y_DISC_PX      = 3.0
TEXT_LINE_MAX_ANGLE_DISC_DEG = 8.0
TEXT_LINE_SEAM_RADIUS_PX     = 20.0

# ─── Sub-pixel boundary refinement (feature feeder) ────────────────────────
# Keep ENABLED for real photos (gradient-rich edges); the engine tests
# disable it from conftest.py because synthetic flat-color edges have no
# gradient and the refinement wiggles them off-axis.

BOUNDARY_REFINE_ENABLED   = True
BOUNDARY_GRADIENT_BAND_PX = 5
BOUNDARY_SMOOTH_SIGMA     = 1.0
BOUNDARY_STEP_PX          = 0.5

# ─── DINOv2 appearance feeder ──────────────────────────────────────────────
# Optional. When enabled, untorn.appearance attaches a dense ViT-S/14
# patch-feature map per fragment; the matcher samples seam-side patches
# and contributes a cosine score to the appearance term.

DINOV2_ENABLED              = True
DINOV2_MODEL                = "dinov2_vits14"
DINOV2_PATCH_SIZE           = 14
DINOV2_INPUT_SIZE           = 224
DINOV2_DEVICE               = "cuda"
DINOV2_SEAM_N_PATCHES       = 8
DINOV2_SEAM_PATCH_OFFSET_PX = 10.0

# ══════════════════════════════════════════════════════════════════════════
#                              ENGINE
# ══════════════════════════════════════════════════════════════════════════

# ─── Contour / Douglas-Peucker (feeder, but matcher reads CURV_*) ──────────
POLY_EPSILON_FACTOR = 0.004

# ─── Curvature feature strings (Wolfson turning function) ──────────────────
CURV_N_SAMPLES     = 80
CURV_SMOOTH_WINDOW = 2
CURV_MIN_STD       = 0.01

# ─── Smith-Waterman alignment ──────────────────────────────────────────────
SW_MATCH_SCORE   =  2.0
SW_CLOSE_PENALTY = -0.1
SW_FAR_PENALTY   = -2.0
SW_GAP_PENALTY   = -1.0
SW_EPSILON_1     =  0.05
SW_EPSILON_2     =  0.20
SW_MIN_SCORE     =  5.0
SW_MIN_ALIGNED   =  6

# ─── Match geometry guards ─────────────────────────────────────────────────
MATCH_MAX_RMS          = 6.0
MIN_TORN_EDGE_PX       = 30.0
MATCH_PROCRUSTES_SEEDS = 3
FACING_COSINE_MIN      = 0.1

# ─── ICP refinement ────────────────────────────────────────────────────────
ICP_MAX_ITER         = 15
ICP_MAX_CORR_DIST_PX = 6.0
ICP_COARSE_DIST_PX   = 25.0

# ─── SDT physical gate ─────────────────────────────────────────────────────
SDT_OVERLAP_FRAC_THRESH  = 0.20
SDT_OVERLAP_DEPTH_THRESH = 10.0
SDT_SEAM_GAP_THRESH_PX   = 8.0

# ─── Full-edge fit cost ────────────────────────────────────────────────────
# fit_cost = FIT_W_OVERLAP * (overlap_px / edge_len)
#          + FIT_W_GAP     * mean_gap
#          + FIT_W_UNCOVERED * (1 - coverage)
FIT_W_OVERLAP         = 1.5
FIT_W_GAP             = 2.0
FIT_W_UNCOVERED       = 80.0
COVERAGE_TOLERANCE_PX = 2.0

# Matcher's pre-seam fit_cost cap. Anything above this is a structurally
# hopeless pose; reject before bothering the seam solver.
MAX_ATTACH_COST       = 30.0
# Post-seam fit_cost cap. After seam refinement, the metric should drop;
# if it can't get below this the merge was a wrong-pair latch.
POST_SEAM_FIT_COST_MAX = 25.0

# ─── Confidence (two-component) ────────────────────────────────────────────
# confidence = CONF_W_GEOMETRY * geom + CONF_W_APPEARANCE * appearance
# geom = max(0, 1 - fit_cost / MAX_ATTACH_COST)
# appearance = mean of (paper-LAB, DINOv2 cosine, strip-NCC) when present
CONF_W_GEOMETRY   = 0.70
CONF_W_APPEARANCE = 0.30

# ─── Bundle adjustment (LM) ────────────────────────────────────────────────
BA_MAX_ITER          = 200
BA_FUNC_TOL          = 1e-6
BA_DENSE_EDGE_SAMPLES = 32
BA_DENSE_EDGE_WEIGHT  = 0.6

# ─── Seam solver (Nelder-Mead) ─────────────────────────────────────────────
SEAM_SOLVER_MAX_ITER        = 40
SEAM_SOLVER_LAMBDA_OVERLAP  = 0.6
SEAM_SOLVER_MAX_DRIFT_DEG   = 2.0
SEAM_SOLVER_MAX_DRIFT_PX    = 5.0
SEAM_SOLVER_MIN_IMPROVEMENT = 0.5

# ─── Overlap safety ────────────────────────────────────────────────────────
RECON_OVERLAP_THRESH = 0.08
OVERLAP_CANVAS_MAX   = 12000

# ─── Siamese edge matcher (optional 7th gate) ──────────────────────────────
EDGE_MATCHER_ENABLED    = True
EDGE_MATCHER_CHECKPOINT = "models/edge_matcher.pt"
EDGE_MATCHER_DEVICE     = "cuda"
EDGE_MATCHER_MIN_SCORE  = 0.55

# ══════════════════════════════════════════════════════════════════════════
#                       Composition + Gap fill (chassis)
# ══════════════════════════════════════════════════════════════════════════

COMP_SUPERSAMPLE           = 2
COMP_LAB_HARMONISE_ENABLED = True
COMP_INK_THRESH            = 140
COMP_SEAM_FILL_MAX_PX      = 6.0

GAP_SMALL_FRAC              = 0.005
GAP_MEDIUM_FRAC             = 0.05
GAP_LARGE_CONTEXT_EXPAND_PX = 20
GAP_EDGE_TOUCH_PX           = 4

# ─── Helpers ───────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create project directories if they don't exist."""
    for d in [INPUT_DIR, OUTPUT_DIR, DEBUG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
