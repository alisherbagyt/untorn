# UNTORN — Full Technical Reference

This document covers the complete architecture, algorithm, and implementation of UNTORN v3.3. It is the authoritative technical reference for understanding how every part of the system works.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [Pipeline Phases](#3-pipeline-phases)
   - [Phase 0: Preprocessing](#phase-0-preprocessing)
   - [Phase 1: Segmentation](#phase-1-segmentation)
   - [Phase 2: Contour Analysis](#phase-2-contour-analysis)
   - [Phase 3: Reconstruction](#phase-3-reconstruction)
   - [Phase 4: Composition](#phase-4-composition)
   - [Phase 5: Gap Fill and Inpainting](#phase-5-gap-fill-and-inpainting)
4. [Matching Algorithm](#4-matching-algorithm)
5. [Assembly Algorithm](#5-assembly-algorithm)
6. [Backend API](#6-backend-api)
7. [Frontend](#7-frontend)
8. [Data Models](#8-data-models)
9. [Configuration Reference](#9-configuration-reference)
10. [Debug Artifacts](#10-debug-artifacts)
11. [Docker Deployment](#11-docker-deployment)
12. [Training the Siamese Edge Matcher](#12-training-the-siamese-edge-matcher)
13. [Design Decisions](#13-design-decisions)

---

## 1. System Overview

UNTORN takes a single photograph of scattered torn-paper fragments on a contrasting background and produces a full reconstruction of the original document.

The system has no prior knowledge of the document's content, colour, language, size, or number of fragments. Everything is derived from the image.

```
┌──────────────────────────────────────────────────────────┐
│  Next.js 14 Frontend  (port 3000)                        │
│  UploadZone → ProcessingView → ResultsView (6 tabs)      │
│  AssemblyView (interactive canvas with export)           │
└────────────────────┬─────────────────────────────────────┘
                     │  HTTP / WebSocket
┌────────────────────▼─────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                            │
│  /process  /status  /result  /debug  /board  /ws        │
│  In-memory job store  ·  single-GPU semaphore            │
└────────────────────┬─────────────────────────────────────┘
                     │  subprocess  (stdout parsing)
┌────────────────────▼─────────────────────────────────────┐
│  UNTORN Python Pipeline  (untorn/)                       │
│  preprocess → segment → contours →                       │
│  reconstruct → compose → inpaint                         │
│  SAM 2.1 · DINOv2 · Siamese CNN · LaMa                  │
└──────────────────────────────────────────────────────────┘
```

The backend runs the pipeline as a child process and streams stdout line by line. Embedded progress markers (`[PHASE N]`, `Phase N complete`, `DONE`) drive the job status API. The frontend renders live progress and the full result with six inspection tabs.

---

## 2. Repository Layout

```
/
├── run.py                    CLI entry point
├── requirements_web.txt      FastAPI + web dependencies
├── Dockerfile.backend        PyTorch/CUDA + SAM + FastAPI
├── Dockerfile.frontend       Node 20 multi-stage build
├── docker-compose.yml        GPU deployment (CUDA 12.1)
├── docker-compose.cpu.yml    CPU-only override
├── .env.example              Environment variable template
├── setup.bat                 Windows setup helper
├── start_backend.bat         Launch backend with --reload
├── start_frontend.bat        Launch frontend with npm run dev
│
├── untorn/                   Core algorithm (18 modules, ~7 000 LOC)
│   ├── pipeline.py           Phase orchestrator
│   ├── config.py             All tunable constants (~100 knobs, 412 lines)
│   ├── preprocess.py         Phase 0: downscale + scale tracking
│   ├── segmentation.py       Phase 1: SAM 2.1 AMG + filtering
│   ├── contours.py           Phase 2: support points, edge segments, SDT
│   ├── boundary.py           Sub-pixel ridge-snapping helper
│   ├── text_lines.py         Per-fragment baseline detection
│   ├── appearance.py         DINOv2 dense features + seam patch cosine
│   ├── matching.py           5-gate matcher cascade (1 177 LOC)
│   ├── grid_filter.py        LBP block fast-filter (597 LOC)
│   ├── edge_matcher.py       Siamese CNN inference adapter (368 LOC)
│   ├── edge_matcher_model.py Siamese CNN architecture (224 LOC)
│   ├── edge_rank.py          Per-edge partner ranking + mutual-best seeds
│   ├── fragment_profile.py   Per-fragment quality profiling
│   ├── fragment_io.py        Edge + SDT preparation helper
│   ├── assembly.py           MST growth + BA + orphan rescue (868 LOC)
│   ├── composition.py        Phase 4: supersampled warp + LAB harmonisation
│   ├── gap_fill.py           Phase 5 front-end: hole classification
│   ├── inpainting.py         LaMa backend wrapper
│   ├── seam_solver.py        Post-MST edge-contact Nelder-Mead refinement
│   ├── io_utils.py           Image read/write helpers
│   └── benchmark/            Benchmark suite (evaluate, generate, run_suite, synth_doc)
│
├── backend/                  FastAPI web service
│   ├── main.py               All endpoints (659 LOC)
│   ├── pipeline_wrapper.py   Subprocess launcher + phase-marker parser
│   ├── services/
│   │   └── job_service.py    In-memory job store + GPU semaphore
│   └── models/
│       └── schemas.py        Pydantic request/response models
│
├── frontend/                 Next.js 14 UI
│   ├── app/
│   │   ├── page.tsx          SPA root: upload → processing → results
│   │   ├── layout.tsx        Root layout + Inter font
│   │   └── globals.css
│   ├── components/
│   │   ├── UploadZone.tsx
│   │   ├── ProcessingView.tsx
│   │   ├── ResultsView.tsx
│   │   ├── FragmentTimeline.tsx
│   │   ├── ui/               Headless UI primitives
│   │   └── views/
│   │       ├── OverviewView.tsx
│   │       ├── SegmentationView.tsx
│   │       ├── ContoursView.tsx
│   │       ├── ReconstructionView.tsx
│   │       ├── CompositionView.tsx
│   │       └── AssemblyView.tsx  (~1 360 LOC interactive canvas)
│   ├── lib/
│   │   ├── api.ts            Typed backend client + all TypeScript interfaces
│   │   └── utils.ts          cn(), phaseLabel() (Russian), getApiBase()
│   ├── next.config.mjs       /api/* → backend rewrite, image domains
│   ├── tailwind.config.ts
│   └── package.json          Next 14.2.5, React 18, Recharts, Lucide
│
├── models/
│   └── edge_matcher.pt       Siamese CNN checkpoint (~2 MB, val_AUC=0.924)
│
├── scripts/
│   └── download_lama_jit.py  Downloads big-lama.pt TorchScript checkpoint
│
├── tools/                    21 CLI scripts: data pipeline, training, tests
│
├── docs/                     ARCHITECTURE.md, PROJECT_STATE.md, RELEASES.md
│
└── data/                     Runtime (gitignored, volume-mounted in Docker)
    ├── input/                Uploaded images
    ├── output/               Final reconstructed PNGs
    └── debug/<stem>/         Per-image debug artifact tree
```

---

## 3. Pipeline Phases

The pipeline is orchestrated by `untorn/pipeline.py::run(input_path, output_path)`. Each phase prints progress markers to stdout which the backend parses to update the job progress percentage.

### Phase 0: Preprocessing

**Module:** `untorn/preprocess.py`  
**Entry:** `prepare_image(image_rgb, debug_dir)`

The pipeline operates on images around 1 500 px on the long side. This fits SAM 2.1 comfortably within 4 GB of VRAM and keeps matching fast.

If the input image is larger, it is downscaled with `cv2.INTER_AREA` (which avoids aliasing artifacts). The `scale_factor = original_longest / working_longest` is stored and applied in Phase 4 to render the final composition at full resolution.

**Upscaling helpers used in Phase 4:**
- `upscale_transforms(transforms, scale_factor)` — multiplies the translation components `(tx, ty)` of every 3×3 affine by `scale_factor`; rotation is scale-invariant.
- `upscale_fragments(fragments, full_rgb, scale_factor)` — upscales binary masks with `INTER_NEAREST`, recomputes contours, bounding boxes, and centroids at full resolution.

**Progress markers:** `[PHASE 0]` → 5%, `Phase 0 complete` → 10%

---

### Phase 1: Segmentation

**Module:** `untorn/segmentation.py`  
**Entry:** `segment_fragments(image_rgb, debug_dir) → list[dict]`

#### 1A — Background Auto-Detection

Samples 5% of edge pixels (top, bottom, left, right margins) from the input image, converts to LAB colour space, and computes the median value as `bg_lab`. This works on any background: black scan bed, white table, coloured cloth, etc.

#### 1B — SAM 2.1 Automatic Mask Generator

Runs `SAM2AutomaticMaskGenerator` with `sam2.1_hiera_small.pt` and a dense 32×32 point grid. SAM internally runs at multiple crop scales (`crop_n_layers=1`), generating hundreds of candidate masks.

Key parameters:
```
points_per_side      = 32     (1 024 seed points)
pred_iou_thresh      = 0.80   (relaxed; appearance filter is the real gate)
stability_score_thresh = 0.88
min_mask_region_area = 500 px
crop_n_layers        = 1
```

#### 1C — Appearance Filter

For each SAM mask, the mean LAB colour of the masked pixels is computed. Masks where this colour is within `BG_DIST_LAB_THRESH = 18.0` ΔE of `bg_lab` are discarded. This separates paper from background without knowing anything about the paper colour in advance.

#### 1D — Post-Processing Chain

1. **Relative area bounds** — keep masks in `[MIN_FRAGMENT_AREA_FRAC × total_px, MAX_FRAGMENT_AREA_FRAC × total_px]`.
2. **Containment-aware hole filling** — if a background-coloured mask is ≥85% enclosed by a paper mask, fill it. This handles degraded paper with ink stains or yellowing that SAM mistakes for holes.
3. **Overlap merging** — merge pairs with IoU > 0.4 (SAM often generates overlapping duplicates at adjacent scales).
4. **Fragment cap** — maximum 40 fragments.
5. **Morphological cleanup** — close small holes, remove isolated islands; kernel sizes scale with image diagonal for DPI invariance.
6. Recompute contour, bbox, centroid, and area from cleaned mask.

**Output per fragment:**
```python
{
    "id": int,
    "mask": HxW uint8,       # 255 = fragment pixel
    "bbox": (x, y, w, h),
    "area": int,
    "contour": Nx2 array,    # OpenCV contour format
    "centroid": [cx, cy]
}
```

**Debug outputs:** `00_background_detection.png`, `01_raw_sam_overlay.png`, `02_after_appearance_filter.png`, `03_mask_raw_*.png`, `04_mask_clean_*.png`, `05_mask_final_*.png`, `06_final_fragments_overlay.png`, `07_crop_*.png`, `fragments_meta.json`

**Progress markers:** `[PHASE 1]` → 12%, `Phase 1 complete` → 44%

---

### Phase 2: Contour Analysis

**Module:** `untorn/contours.py`  
**Entry:** `analyze_fragments(fragments, work_rgb, debug_dir) → fragments (enriched)`

This phase produces the rich edge descriptors that the matching algorithm depends on.

#### 2A — Sub-pixel Boundary Refinement

Called via `boundary.py::attach_subpixel_contours_all`. For every boundary pixel, the function searches ±5 px along the inward normal direction (step size 0.5 px) for the local image-gradient magnitude maximum. The boundary pixel is snapped to this sub-pixel location.

**Why:** Integer-grid mask boundaries have ±1 px quantisation error. On a gentle torn edge this can dominate the curvature signal. Sub-pixel snapping reduces this to ~0.1 px, making the curvature features far more distinctive.

#### 2B — Text-Line Detection

Called via `text_lines.py::attach_text_lines_all`. For each fragment:
- Rotate the fragment mask through ±30° in 2° steps.
- At each angle, project the ink-pixel (grayscale < `COMP_INK_THRESH`) column onto the axis perpendicular to the sweep direction.
- Find projection peaks: rows with ≥2% ink pixel density are candidate baselines.
- Store detected lines as `frag["text_lines"]`, median angle as `frag["text_angle_deg"]`.

Text-line angles are used in Phase 3 gate 7 (text-line continuity) to reject matches where text baselines would be discontinuous across the proposed seam.

#### 2C — Support Point Extraction

Douglas-Peucker polygonal approximation with `ε = 0.4%` of perimeter length. Support points are vertices where the torn edge profile changes direction significantly — they define the boundary segments passed to matching.

#### 2D — Edge Segment Computation

Between consecutive support points, each edge segment carries:
- Length, midpoint, angle, inward normal vector
- `is_torn` classification: fit a line with RANSAC; low inlier ratio → torn edge

Only torn edges participate in matching. Factory (straight, scissor-cut) edges are never seams.

#### 2E — Signed Distance Transform

Computed inside `matching.prepare_edges_and_sdt()`, called at the start of Phase 3. For each fragment mask:
- `_sdt_interior`: positive values = interior distance from boundary, negative = exterior, zero = boundary.

The SDT is used in Phase 3 for physical penetration and gap measurement.

#### 2F — Fragment Profiling

`fragment_profile.py::build_fragment_profiles` runs after SDT construction. For each fragment:
- **Role:** `corner / boundary / interior` based on factory edge count and arrangement.
- **Ink density:** fraction of masked pixels below grayscale threshold.
- **Curvature distinctiveness:** variance of curvature string (higher = more distinctive torn edge).
- **Anchor strength:** per torn edge; how strongly the fragment's profile supports a confident match on that edge.

Profiles are written to `reconstruction/fragment_profiles.json` and used to weight seed candidate ranking in assembly.

**Progress markers:** `[PHASE 2]` → 45%, `Phase 2 complete` → 54%

---

### Phase 3: Reconstruction

**Modules:** `untorn/assembly.py`, `untorn/matching.py`, `untorn/grid_filter.py`, `untorn/edge_rank.py`, `untorn/seam_solver.py`  
**Entry:** `reconstruct(fragments, image_rgb, debug_dir) → dict[int, 3×3 affine]`

Phase 3 finds the rigid SE(2) transform for every fragment — rotation and translation — that places it back in its original document position.

#### 3A — Feature Preparation

- `matching.prepare_edges_and_sdt(fragments, image_rgb)` — populates `frag["edges"]`, `frag["_curv"]`, `frag["_sdt_interior"]`, `frag["boundary_pixels"]`, `frag["paper_lab"]`.
- `appearance.attach_dinov2_features_all(fragments, image_rgb)` — extracts `(H_p, W_p, 384)` DINOv2 ViT-S/14 patch token maps. Cached on `frag["dinov2"]`. Loaded once and reused.
- `fragment_profile.build_fragment_profiles(fragments, image_rgb)` — per-fragment profiling (see 2F above).

#### 3B — Candidate Pair Enumeration

All `(i, j)` pairs where both fragments have at least one torn edge are enumerated. A multi-stage prefilter cuts the pair list before any expensive computation:

1. Edge-length ratio ≤ 2.5 (very different length edges cannot match).
2. Centroid distance ≤ 3 × 200 px (= 600 px; coarse proximity gate).
3. Paper-colour LAB ΔE ≤ 24 (2 × `MATCH_PAPER_COLOR_DELTA_MAX`).
4. **LBP grid filter** (`grid_filter.py`): samples a 20 px inward band from each fragment's boundary, Otsu-binarizes, tiles into 16×16 blocks, computes Local Binary Pattern histograms, runs 2-point rigid RANSAC across the joint histogram space, retains top-8 partners per fragment. This fast filter cuts ~80% of the remaining candidate pairs.

Surviving pairs are ranked by `paper_lab_ΔE + anchor_weight × (1 − min_anchor_strength)` (lower = try first). The list is capped at `ASSEMBLY_MAX_CANDIDATE_PAIRS = 4000`.

#### 3C — 5-Gate Pair Matching

For each candidate pair, `matching.match_pair(frag_a, frag_b, image_rgb)` iterates over all torn-edge combinations and applies the gate cascade. See [Section 4: Matching Algorithm](#4-matching-algorithm) for full details.

Each evaluated pair is cached in a `_MatchCache` so the same `(i, j)` is never recomputed during assembly.

#### 3D — Edge Ranking and Mutual-Best Seeds

`edge_rank.py::rank_edges_per_fragment()` ranks each fragment's torn edges by the confidence of their best cached match candidate.

`edge_rank.py::build_pair_mutual_scores()` finds pairs where both fragments independently rank each other near the top of their candidate lists. These mutual-best pairs become priority seeds for the MST growth — they represent the highest-confidence information in the match cache.

#### 3E — MST Growth

Starting from a seed pair (highest mutual confidence, or highest absolute confidence if no mutual pair exists), the assembly grows a minimum spanning tree:

- The seed anchor fragment gets the identity transform.
- A priority queue of `(confidence, placed_anchor, free_fragment)` tuples tracks expansion candidates.
- Each round: collect all `(placed, free)` edge-pair matches; resolve conflicts with `networkx.max_weight_matching()` (when multiple free fragments compete for the same anchor position); attach the highest-score winning pair.
- **Seam gates at attach time:** `fit_cost ≤ MAX_ATTACH_COST`, `fit_overlap_frac ≤ SEAM_MAX_OVERLAP_FRAC`, `fit_gap_px ≤ SEAM_MAX_GAP_PX`, `fit_coverage ≥ SEAM_MIN_COVERAGE`.
- Auto-lock at `ASSEMBLY_HIGH_CONFIDENCE_LOCK = 0.92` (attached immediately, not re-evaluated).
- Minimum confidence gate: `ASSEMBLY_MIN_CONFIDENCE = 0.45`.
- Safety cap: `ASSEMBLY_MAX_STEPS = 1024`.

#### 3F — Seam-Contact Refinement

After the MST phase, `seam_solver.py::refine_pair()` runs Nelder-Mead on `(Δθ, Δdx, Δdy)` for every recorded attach, minimising `fit_cost + SDT_penetration_penalty`. Maximum 40 iterations; drift caps: ±2°, ±5 px; minimum required improvement: 0.5 cost units.

#### 3G — Bundle Adjustment

Levenberg-Marquardt pose-graph optimization over all non-seed placed fragment poses `(θ, tx, ty)`.

**Sparse SW constraint:** For every cached match `(i, j)` with aligned points `matched_a`, `matched_b`:
```
T_i(matched_a[k]) ≈ T_j(matched_b[k])
```
This is the primary signal: SW-matched curvature samples should coincide in world space.

**Dense edge constraint** (when `BA_DENSE_EDGE_ENABLED = True`): 32 equal-arc-length samples along every torn edge pair contribute symmetric nearest-neighbour residuals. This physically welds the polylines together and prevents residual gaps.

Safety caps: ±8° rotation, ±60 px translation per fragment from placement estimate. Max 200 iterations, `func_tol = 1e-6`.

#### 3H — Orphan Rescue

**Standard rescue:** Fragments not reached by MST growth are evaluated against all placed fragments with relaxed gates: `ASSEMBLY_ORPHAN_MIN_CONFIDENCE = 0.40`, `ORPHAN_MAX_ATTACH_COST = 700`.

**Aggressive rescue** (when `ASSEMBLY_AGGRESSIVE_ORPHAN_RESCUE = True`):
- Pass 1: Force-evaluate every `(orphan, placed)` pair even if it was dropped by the prefilter.
- Pass 2: Force-evaluate `(orphan, orphan)` pairs; seed a new cluster from the best pair.
- Minimum confidence: 0.30, max cost: 1 200.

#### 3I — Cluster Reconciliation

When `ASSEMBLY_CLUSTER_RECONCILE = True` (disabled by default), the algorithm identifies disconnected clusters via union-find over the merge log, finds the lowest-cost cross-cluster bridge from the cached match scores, and rigidly shifts one entire cluster so the bridge match is satisfied. Bundle adjustment runs again after reconciliation.

**Output:** `dict[fragment_index → 3×3 SE(2) homogeneous affine]`

```python
M = np.array([[R[0,0], R[0,1], tx],
              [R[1,0], R[1,1], ty],
              [0,      0,      1 ]])
```

**Debug outputs:** `merge_log.json`, `assembly_summary.json`, `final_translations.json`, step PNG snapshots.

**Progress markers:** `[PHASE 3]` → 55%, `Phase 3 complete` → 81%

---

### Phase 4: Composition

**Module:** `untorn/composition.py`  
**Entry:** `compose_final(image_rgb, fragments, transforms, debug_dir) → dict`

Phase 4 renders the final reconstruction at full input resolution.

**Steps:**

1. **Transform upscaling** — multiply `(tx, ty)` of every 3×3 affine by `scale_factor`; rotation is scale-invariant. Upscale fragment masks to full resolution.
2. **Canvas sizing** — compute the axis-aligned bounding box (AABB) of all transformed fragment bboxes; pad 10 px on each side.
3. **2× supersampled warp** — each fragment is warped onto a 2× canvas with `cv2.warpAffine(INTER_LINEAR)`, then the 2× canvas is downsampled to final resolution with `INTER_AREA`. This halves resampling artifacts at seam boundaries.
4. **SDT pixel arbitration** — where fragments overlap (due to reconstruction error or physical overlap), the winner is the fragment with the largest interior signed-distance value at that pixel. The deepest interior pixel wins, not z-order — this prevents thin slivers from covering large interior regions.
5. **LAB colour harmonisation** — the mean paper-pixel L (lightness) value of each fragment is shifted toward the cluster median. Ink pixels are attenuated so text contrast is preserved. Compensates for scanner-lightness variation between fragments.
6. **Voronoi seam fill** — uncovered pixels within `COMP_SEAM_FILL_MAX_PX = 6` px of the coverage boundary are filled from the nearest covered pixel. Remaining larger holes are passed to Phase 5.
7. Compute coverage mask, gap mask, and tight crop bbox.

**Output:**
```python
{
    "canvas":    HxWx3 uint8,
    "coverage":  HxW uint8,    # 0=gap, 255=covered
    "gap_mask":  HxW uint8,    # pixels to inpaint
    "crop_bbox": (x, y, w, h)
}
```

**Debug outputs:** `01_raw_composite.png`, `03_gap_mask.png`, `04_inpainted.png`, `composition_meta.json`

**Progress markers:** `[PHASE 4]` → 82%, `Phase 4 complete` → 89%

---

### Phase 5: Gap Fill and Inpainting

**Modules:** `untorn/gap_fill.py`, `untorn/inpainting.py`  
**Entry:** `inpaint_gaps(canvas, coverage, debug_dir, refine=False) → dict`

#### 5A — Hole Classification

Computes the convex hull of the coverage mask as a proxy for the document boundary. Finds connected components of `(hull area − coverage)`. Classifies each:

| Class | Criterion | Treatment |
|-------|-----------|-----------|
| Edge hole | Touches canvas border ≤4 px | Excluded (will be cropped) |
| Small | Area < 0.5% of document bbox | Seam scar treatment |
| Medium | 0.5%–5% | Expanded context (+20 px dilation) for LaMa |
| Large | > 5% | Flagged as `missing_fragment=true` in metadata |

#### 5B — Repair Mask Construction

1. Build a 4 px half-width ring around each seam boundary (the scar region left by fragment edges).
2. Mask out ink pixels (grayscale < 140) — text strokes are excluded from the inpaint mask.
3. Union with hole interiors to form the unified scar-and-hole repair mask.

This protects text. LaMa only operates on bare paper regions, never touching character strokes.

#### 5C — LaMa Inpainting Backends

Tried in priority order:

| Priority | Backend | How |
|----------|---------|-----|
| 1 | **TorchScript JIT** | `torch.jit.load("big-lama.pt")` — zero extra deps |
| 2 | **simple-lama-inpainting** | Pip package wrapper |
| 3 | **saicinpainting** | Full LaMa codebase |
| 4 | **Classical fallback** | Distance-weighted colour blend from nearest covered pixels |

**Tiled inference** is used when the canvas exceeds GPU VRAM: 512 px tiles with 64 px overlap; tiles are blended with a cosine weight window to prevent visible tile-boundary artifacts.

**Debug outputs:** `01_before.png`, `02_holes_interior.png`, `03_repair_mask.png`, `04_cleaned.png`, `inpainting_meta.json`

**Progress markers:** `[PHASE 5]` → 90%, `Phase 5 complete` → 98%, `DONE <stem> SAVED TO <path>` → 100%

---

## 4. Matching Algorithm

Implemented in `untorn/matching.py::match_pair(frag_a, frag_b, image_rgb)`.

For a candidate pair `(A, B)`, the function tries every combination of torn edges `(edge_i on A, edge_j on B)` and runs each through the gate cascade. The first combination that passes all gates produces the match result.

### Gate 0 — Minimum Edge Quality

Both edges must be classified as `is_torn = True`. Edge must be at least `MIN_TORN_EDGE_PX = 30` px long. Curvature string must have `curv_std > CURV_MIN_STD` (non-trivial profile).

### Gate 1 — Outward-Normal Facing

The offset vector from fragment A centroid to fragment B centroid is computed. The dot product with A's edge outward normal must be ≥ `FACING_COSINE_MIN = 0.1`. Fragments must face each other for a seam to be geometrically plausible.

### Gate 2 — Smith-Waterman Curvature Alignment

The 80-element curvature feature strings of the two edges are aligned using Smith-Waterman local sequence alignment (the same algorithm used for DNA sequence alignment). Curvature values are quantized and compared elementwise.

```
Match score:    +2.0   (curvature values within ε₁)
Near penalty:   -0.1   (within ε₂ but not ε₁)
Far penalty:    -2.0   (outside ε₂)
Gap penalty:    -1.0

Minimum total score:    5.0
Minimum aligned samples: 6
```

The alignment returns `(matched_a, matched_b)` — the corresponding sample indices on each edge where curvature profiles agree. This sub-arc is the region most likely to be the physical seam.

**Why SW over global alignment:** Torn fragments often have corner pieces or damage at tips — the matching sub-arc may be only a portion of one edge. SW finds the best-scoring contiguous sub-region even when the edges differ in total length or one is partially obscured.

### Gate 3 — Procrustes Rigid Fit

SVD-based least-squares rigid alignment of the 2D world coordinates of `(matched_a, matched_b)`. Run from `MATCH_PROCRUSTES_SEEDS = 3` starting sub-arcs (start, middle, end thirds) and keep the best.

Returns: `(R: 2×2 rotation, translation: [tx, ty], rms_error: float)`.

Gate: `rms_error ≤ MATCH_MAX_RMS = 6.0` px. Rotation: `|angle| ≤ RECON_MAX_ROTATION_DEG = 30°`.

### Gate 4 — Two-Phase ICP

Iterative Closest Point, coarse then fine, applied after Procrustes to remove residual drift.

| Phase | Correspondence distance | Purpose |
|-------|------------------------|---------|
| Coarse | 25 px | Pull drifted edge tails together |
| Fine | 6 px | Sub-pixel seam closure |

Maximum 15 iterations per phase. Maximum rotation drift: 15° total from Procrustes estimate (prevents ICP from flipping into a local minimum).

### Gate 5 — Siamese CNN Edge Scorer

`edge_matcher.py::score_edge_pair()` extracts a `(32, 256, 3)` RGB strip along the boundary of each fragment (32 px inward band, 256 px along the edge, normalised). The trained Siamese CNN (`edge_matcher_model.py`) processes both strips through shared weights and returns a match probability via cosine similarity with a learnable temperature.

Gate: `match_prob ≥ EDGE_MATCHER_MIN_SCORE = 0.55`.

The model (`models/edge_matcher.pt`, ~520K parameters) was trained on synthetic torn-edge pairs to `val_AUC = 0.924`. It acts as a learned appearance gate that catches cases where curvature geometry matches but the paper texture/colour is inconsistent.

### Gate 6 — DINOv2 Seam Patch Cosine

Samples 8 patch positions along the aligned seam, on both sides. For each position, retrieves the DINOv2 ViT-S/14 patch token (384-dimensional) from the pre-extracted feature map. Computes the mean cosine similarity across all 16 patches (8 pairs).

Gate: `mean_cosine ≥ MATCH_APPEARANCE_COS_MIN = 0.55`.

DINOv2 features capture semantic appearance (paper grain, printing, colouring) at a scale larger than individual pixels, making this gate more robust to seam-edge pixel noise than raw colour correlation.

### Gate 7 — Text-Line Continuity

When a proposed seam position means ≥2 expected text lines should cross it, the system checks whether the detected baselines on both fragments are approximately collinear across the seam. Requires ≥ `MATCH_TEXT_LINE_MIN_CONT = 0.30` continuity score.

This gate catches geometrically plausible matches where the text would run at the wrong angle — a physically impossible configuration.

### Gate 8 — Paper-Colour Similarity (Soft)

LAB ΔE between the two fragments' paper-colour fingerprints. Values < 12 contribute a positive paper score to the confidence formula. This is a soft score, not a hard gate.

### Gate 9 — SDT Physical Gates

After aligning fragment B onto fragment A using the current transform:

**Penetration check:** Look up B's boundary pixels in A's signed distance transform. If > 20% of B's contour pixels are inside A (positive SDT value) or maximum penetration depth > 10 px → reject.

**Gap check:** Median seam gap (nearest-neighbour distance between aligned boundary points) must be ≤ `SDT_SEAM_GAP_THRESH_PX = 8` px.

These are hard physical constraints. No curvature or appearance score can override them — fragments that geometrically cannot fit are rejected unconditionally.

### Gate 10 — Full-Edge Fit Cost

Evaluates alignment quality over the *entire* edge length, not just the SW-matched sub-arc:

```
fit_cost = 1.5 × fit_overlap_px
         + 2.0 × fit_gap_px
         + 80  × (1 - fit_coverage)
```

Where:
- `fit_overlap_px` — SDT-measured penetration of B into A
- `fit_gap_px` — mean nearest-neighbour distance along full edge
- `fit_coverage` — fraction of A's edge with a matching B point within 2 px

Gate: `fit_cost ≤ MAX_ATTACH_COST = 400.0`. This is the primary ranking signal for pair quality.

### Confidence Score

Five-component weighted score in [0, 1]:

```python
confidence = (0.30 × geom_conf
            + 0.20 × dinov2_score
            + 0.20 × strip_ncc
            + 0.20 × text_score
            + 0.10 × paper_score)
```

Where `geom_conf` is derived from `rms`, alignment arc length, and curvature cross-correlation.

### Match Result

```python
{
    "frag_i": int,
    "frag_j": int,
    "R": np.ndarray(2, 2),         # rotation matrix
    "translation": np.ndarray(2,), # [tx, ty] in fragment-A frame
    "angle": float,                # rotation in radians
    "rms": float,
    "confidence": float,           # [0, 1]
    "fit_cost": float,
    "fit_overlap_px": float,
    "fit_overlap_frac": float,
    "fit_gap_px": float,
    "fit_coverage": float,
    "edge_i": int,
    "edge_j": int,
    "matched_a": np.ndarray,
    "matched_b": np.ndarray,
    "edge_matcher_prob": float,
    "edge_matcher_cos": float,
}
```

---

## 5. Assembly Algorithm

`untorn/assembly.py::reconstruct()` orchestrates Phases 3D–3I described above. Here is the high-level control flow:

```
1. Feature prep (DINOv2, SDT, profiles)
2. Enumerate candidate pairs → prefilter → grid filter
3. For each candidate pair: run 5-gate matcher, cache result
4. Rank edges per fragment, find mutual-best seeds
5. Seed MST from highest-confidence mutual pair
6. Priority-queue MST growth:
     while unplaced fragments remain:
       collect all (placed, free) edge-pair matches from cache
       resolve conflicts with max_weight_matching
       attach winner if gates pass
       add winner's neighbors to queue
7. Seam-contact Nelder-Mead refinement on all attaches
8. Bundle adjustment (LM, sparse SW + dense edge constraints)
9. Orphan rescue (standard, then aggressive if needed)
10. Return {fragment_id: 3x3 affine}
```

The merge log records every successful attach: anchor fragment, attached fragment, edge indices, confidence, fit_cost, rotation angle, rms. This log drives the step-through viewer in the frontend's Reconstruction tab.

---

## 6. Backend API

**Framework:** FastAPI  
**Base URL:** `http://localhost:8000`

### Endpoints

#### Processing

| Method | Path | Description |
|--------|------|-------------|
| POST | `/process` | Upload image (multipart `file` field), create job, start pipeline |
| GET | `/status/{job_id}` | Poll job state: status, progress 0–100, current_phase, logs |
| GET | `/result/{job_id}` | Download final PNG (only when `status == "done"`) |
| WS | `/ws/{job_id}` | Real-time push every ~400 ms |
| GET | `/health` | `{"status": "ok", "timestamp": "..."}` |
| GET | `/jobs` | List all in-memory jobs |

**POST /process** response:
```json
{ "job_id": "uuid4-string", "message": "Processing started" }
```

**GET /status/{job_id}** response:
```json
{
  "job_id": "...",
  "status": "queued | processing | done | error",
  "progress": 0,
  "current_phase": "segmentation",
  "logs": ["last 15 stdout lines"],
  "error": null,
  "queue_position": 0,
  "queued_count": 0,
  "started_at": "ISO datetime",
  "finished_at": null
}
```

#### Debug Inspection

| Method | Path | Description |
|--------|------|-------------|
| GET | `/debug/{job_id}` | Aggregated pipeline metadata + image URL paths |
| GET | `/debug/image/{job_id}/{path}` | Serve any debug image by relative path |

`/debug/{job_id}` response keys:
- `pipeline_meta` — timings per phase, fragment count, scale factor, edge_matcher_loaded, missing_fragment, hole counts
- `fragments` — per-fragment area, bbox, centroid
- `contours` — edge count, perimeter, edge lengths
- `steps` — normalised merge log (step, anchor, attached, confidence, fit_cost, angle, rms)
- `translations` — final per-fragment dx/dy/rotation
- `composition` — canvas dims, gap pixel counts
- `inpainting` — backend used, mask_pixels, duration_s, device, refine flag
- `paths` — URLs to all debug images

#### Assembly Board

| Method | Path | Description |
|--------|------|-------------|
| GET | `/board/data/{job_id}` | Canvas dimensions + per-fragment placement info |
| GET | `/board/fragment/{job_id}/{fragment_id}` | Transparent RGBA PNG of one fragment |
| POST | `/board/export/{job_id}` | Composite user-positioned fragments → PNG download |

**POST /board/export** request:
```json
{
  "fragments": [{"id": 0, "x": 100, "y": 50, "rotation": -2.5}],
  "canvas_width": 2000,
  "canvas_height": 2800,
  "scale": 2,
  "clean": true,
  "refine": false
}
```

Response: PNG attachment (`Content-Disposition: attachment; filename=assembly.png`).

### Job Lifecycle

```
POST /process
    ↓
create_job()  →  status=queued, progress=0
    ↓
daemon thread acquires PIPELINE_SEMAPHORE (only one pipeline at a time)
    ↓
subprocess: python run.py <image>
    ↓  stdout line-by-line
parse progress markers → update_job(progress, current_phase)
    ↓
"DONE"   →  status=done, progress=100
"FAILED" →  status=error, error=last_stderr_line
```

Additional submissions while a pipeline runs set `status=queued` and expose `queue_position` in the status response. CORS is open (`allow_origins=["*"]`).

### Job Store

`backend/services/job_service.py`:
- Global `JOBS: dict[str, dict]` guarded by `threading.Lock`
- `PIPELINE_SEMAPHORE = threading.Semaphore(1)` — ensures single-GPU exclusive access
- States: `queued → processing → done | error`
- No persistence — jobs are lost on server restart (by design for a single-user local tool)

---

## 7. Frontend

**Framework:** Next.js 14.2.5 (App Router), React 18, TypeScript, Tailwind CSS 3.4.7

### Application State Machine

`frontend/app/page.tsx` manages three top-level states:

```
upload → processing → results
```

- `upload → processing`: user drops or selects an image; `POST /process` succeeds; WebSocket opened at `ws://{host}/ws/{job_id}`
- `processing → results`: WebSocket delivers `status == "done"` (fallback: `GET /status` poll every 800 ms)
- On results entry: `GET /debug/{job_id}` fetched; all debug image URLs preloaded via `new Image()` objects in a background effect for instant tab switching

Full-page drag-and-drop overlay is active on all states (drops are rejected during processing with a visual indicator).

### Components

#### `UploadZone.tsx`

Hidden `<input type="file">` accepting `.tif`, `.tiff`, `.jpg`, `.jpeg`, `.png`. Shows a spinner during upload. Emits `onJobCreated(jobId)` on success.

#### `ProcessingView.tsx`

Phase timeline: 6 phases (preprocessing / segmentation / contours / reconstruction / composition / gap_fill), each with an icon (CheckCircle done, Loader spinning active, Circle pending), label, and description. Progress percentage in accent colour. Queue position displayed when `status == "queued"`. Live log feed (last 6 lines, monospace, newest-first).

#### `ResultsView.tsx`

Six navigation pills (Russian labels): Обзор / Сегментация / Контуры / Реконструкция / Компоновка / Сборка. `AssemblyView` is lazy-mounted on first visit and kept alive hidden to preserve canvas state. Status badges: "Siamese gate активен" (if `edge_matcher_loaded`), "LaMa очистка" / "LaMa недоступна".

#### `views/OverviewView.tsx`

4-stat grid (fragments, original size, working size, scale factor). Gap fill summary: LaMa status, mask pixels, hole counts by class, missing-fragment badge. Phase timing bar chart (proportional, colour-coded). Full-width final image with download link.

#### `views/SegmentationView.tsx`

SAM mask overlay side-by-side with final fragments overlay. `FragmentTimeline` carousel: per-fragment crop, binary mask, SDF visualisation, area and bbox stats.

#### `views/ContoursView.tsx`

Recharts: perimeter bar chart, support-point radar chart, edge-length bar chart. SDF visualisation per fragment.

#### `views/ReconstructionView.tsx`

Step-through merge log viewer: index slider and prev/next buttons navigate the merge log; each step shows the associated snapshot PNG. Recharts LineChart of gap score over all placement steps. Neighbor pair list, dx/dy translation scatter.

#### `views/CompositionView.tsx`

Layer toggle pills: raw composite / gap mask / classical inpaint / LaMa cleaned. Canvas size, gap pixel count, gap coverage percentage. "Cleaned" layer hidden if LaMa was unavailable.

#### `views/AssemblyView.tsx` (~1 360 LOC)

Fully custom `<canvas>` component for manual fragment placement and export.

**Interaction:**
- **Drag** (mouse/touch) to move fragments
- **Rotate** via right-click drag or on-canvas rotation handle
- **Zoom** with scroll wheel: 5%–500%
- **Space hold:** pan mode (cursor becomes grab)
- **Layer panel:** visibility toggle, z-order reorder (ChevronUp/Down), lock toggle

**Keyboard shortcuts:**
- `Arrow`: nudge 1 px (+ `Shift` = 10 px)
- `[` / `]`: rotate ±1° (+ `Shift` = ±15°)
- `Ctrl+Z`: undo (50-operation stack)
- `Esc`: deselect
- `G`: toggle grid overlay (20 px grid)

**Fragment images:** Loaded once from `/board/fragment/{job_id}/{id}` as transparent PNGs, cached as Blob URLs in a `useRef<Map>`, revoked on component unmount.

**Export dialog:** Scale 1×/2×/4×, LaMa clean toggle, LaMa refine toggle → `POST /board/export/{job_id}` → PNG download.

**Custom scrollbars:** Virtual horizontal and vertical scrollbars with drag-to-pan support.

**Inspector panel:** Shows selected fragment position (x, y, rotation). "Reset all" button restores pipeline placement.

### API Client (`frontend/lib/api.ts`)

Typed wrappers for all backend calls:
```typescript
uploadImage(file: File): Promise<{ job_id: string }>
fetchStatus(jobId: string): Promise<JobStatus>
fetchDebug(jobId: string): Promise<DebugData>
fetchBoardData(jobId: string): Promise<BoardData>
exportBoard(jobId, fragments, canvasW, canvasH, scale, opts): Promise<Blob>
createWebSocket(jobId: string): WebSocket
debugImageUrl(jobId, path): string
resultImageUrl(jobId): string
boardFragmentUrl(jobId, fragmentId): string
```

Base URL: reads `NEXT_PUBLIC_API_BASE` env var; falls back to `http://<window.location.hostname>:8000`.

### Styling

- Tailwind CSS with custom CSS variables: `--color-primary/secondary/accent/muted/border/success/danger/warning`
- Accent: indigo `#4F46E5`
- Lucide React icons throughout
- Recharts for charts
- react-dropzone for upload
- Inter font via Google Fonts
- All UI labels in Russian (phaseLabel() mapping in `utils.ts`)

---

## 8. Data Models

### Fragment Dict (in-memory, passed between phases)

```python
{
    # From Phase 1 (segmentation)
    "id": int,
    "mask": np.ndarray,          # HxW uint8, 255 = fragment pixel
    "bbox": (x, y, w, h),
    "area": int,
    "contour": np.ndarray,       # OpenCV Nx1x2 contour
    "centroid": np.ndarray,      # [cx, cy]

    # Added by Phase 2 (contours)
    "edges": list[dict],         # per-edge geometry + is_torn flag
    "_curv": np.ndarray,         # (80,) curvature feature string
    "_sdt_interior": np.ndarray, # signed distance transform
    "boundary_pixels": np.ndarray,  # (N, 2) boundary pixel coords
    "text_lines": list[dict],
    "text_angle_deg": float | None,
    "paper_lab": np.ndarray,     # [L, a, b] paper colour fingerprint
    "max_anchor_strength": float,

    # Added by appearance.py
    "dinov2": np.ndarray,        # (H_p, W_p, 384) patch token map
}
```

### Transform Dict

`dict[fragment_id: int → affine: np.ndarray(3, 3)]`

SE(2) homogeneous affines in working-resolution space. Phase 4 upscales by multiplying `(tx, ty)` by `scale_factor`.

### Edge Segment Dict

```python
{
    "pts": np.ndarray,      # (N, 2) boundary pixel coordinates
    "length": float,
    "angle": float,         # radians
    "midpoint": np.ndarray,
    "normal": np.ndarray,   # outward unit normal
    "is_torn": bool,
}
```

### Match Result Dict

```python
{
    "frag_i": int,
    "frag_j": int,
    "R": np.ndarray,            # (2, 2) rotation matrix
    "translation": np.ndarray,  # (2,) [tx, ty]
    "angle": float,             # rotation in radians
    "rms": float,
    "confidence": float,        # [0, 1]
    "fit_cost": float,
    "fit_overlap_px": float,
    "fit_overlap_frac": float,
    "fit_gap_px": float,
    "fit_coverage": float,
    "edge_i": int,
    "edge_j": int,
    "matched_a": np.ndarray,    # aligned sample coords on frag_i
    "matched_b": np.ndarray,    # aligned sample coords on frag_j
    "edge_matcher_prob": float,
    "edge_matcher_cos": float,
}
```

### TypeScript Types (`frontend/lib/api.ts`)

```typescript
interface JobStatus {
    job_id: string
    status: "queued" | "processing" | "done" | "error"
    progress: number          // 0–100
    current_phase: string
    logs: string[]
    error: string | null
    queue_position: number
    queued_count: number
    started_at: string | null
    finished_at: string | null
}

interface DebugData {
    pipeline_meta: PipelineMeta
    fragments: FragmentMeta[]
    contours: ContourMeta[]
    neighbors: NeighborMeta[]
    steps: StepEntry[]
    translations: TranslationEntry[]
    composition: CompositionMeta
    inpainting: InpaintingMeta
    paths: DebugPaths
}

interface BoardData {
    canvas: { width: number; height: number }
    fragments: BoardFragment[]
}

interface BoardFragment {
    id: number
    x: number
    y: number
    width: number
    height: number
    rotation: number
    placed: boolean
    area: number
    centroid: [number, number]
    imageUrl: string
}
```

### JSON Debug Files

Written to `data/debug/<stem>/`:

| File | Contents |
|------|----------|
| `pipeline_meta.json` | Phase timings, n_fragments, scale_factor, status, edge_matcher_loaded, missing_fragment, hole counts |
| `segmentation/fragments_meta.json` | `[{id, area, bbox_xywh, centroid}]` |
| `segmentation/raw_masks_meta.json` | SAM raw mask scores + bg distances |
| `contours/contours_meta.json` | Per-fragment edge count, perimeter, edge lengths |
| `reconstruction/merge_log.json` | Step-by-step: phase, anchor, attached, confidence, fit_cost, angle, rms |
| `reconstruction/assembly_summary.json` | n_placed, seed info, cache hit rate |
| `reconstruction/final_translations.json` | `{frag_id: {dx, dy, placed}}` |
| `composition/composition_meta.json` | Canvas dims, offset, gap pixel counts |
| `inpainting/inpainting_meta.json` | backend_used, mask_pixels, duration_s, device, refine, hole_counts, largest_hole_frac |

---

## 9. Configuration Reference

All constants live in `untorn/config.py` (412 lines, ~100 knobs). Changes take effect on the next pipeline run — no rebuild needed.

### Working Resolution

| Parameter | Default | Effect |
|-----------|---------|--------|
| `WORKING_MAX_DIM` | 1500 | Longest side for working resolution; raise to 2000 if you have ≥6 GB VRAM |

### SAM 2.1

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SAM2_POINTS_PER_SIDE` | 32 | Grid density (32×32 = 1 024 seed points); raise to 64 for dense small fragments |
| `SAM2_PRED_IOU_THRESH` | 0.80 | SAM internal quality filter |
| `SAM2_STABILITY_THRESH` | 0.88 | SAM internal stability filter |
| `SAM2_MIN_MASK_AREA` | 500 | Minimum raw mask area in pixels |
| `BG_DIST_LAB_THRESH` | 18.0 | LAB ΔE for paper vs. background; lower for high-contrast scenes |
| `MIN_FRAGMENT_AREA_FRAC` | 0.0003 | Minimum fragment size as fraction of image area |
| `MAX_FRAGMENT_AREA_FRAC` | 0.90 | Maximum (catches near-full-image background masks) |
| `MAX_FRAGMENTS` | 40 | Hard cap on fragment count |

### Feature Flags

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BOUNDARY_REFINE_ENABLED` | True | Sub-pixel ridge snapping |
| `DINOV2_ENABLED` | True | DINOv2 appearance features |
| `DINOV2_MODEL` | `"dinov2_vits14"` | Model variant (vits14 = ViT-S/14, 384-dim) |
| `GRID_FILTER_ENABLED` | True | LBP fast-filter prescreen |
| `EDGE_MATCHER_ENABLED` | True | Siamese CNN gate |
| `COMP_LAB_HARMONISE_ENABLED` | True | LAB lightness harmonisation |
| `SEAM_SOLVER_ENABLED` | True | Post-MST Nelder-Mead refinement |
| `BA_DENSE_EDGE_ENABLED` | True | Dense edge constraint in bundle adjustment |
| `ASSEMBLY_AGGRESSIVE_ORPHAN_RESCUE` | True | Force-evaluate all orphan pairs |
| `ASSEMBLY_CLUSTER_RECONCILE` | False | Cross-cluster bridge (experimental) |

### Matching Gates

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MIN_TORN_EDGE_PX` | 30 | Minimum torn edge length for matching |
| `FACING_COSINE_MIN` | 0.1 | Outward-normal facing threshold |
| `SW_MATCH_SCORE` | 2.0 | SW reward for aligned curvature |
| `SW_FAR_PENALTY` | -2.0 | SW penalty for mismatched curvature |
| `SW_GAP_PENALTY` | -1.0 | SW gap penalty |
| `SW_MIN_SCORE` | 5.0 | Minimum SW alignment score |
| `SW_MIN_ALIGNED` | 6 | Minimum aligned samples |
| `MATCH_PROCRUSTES_SEEDS` | 3 | Number of sub-arc seeds for Procrustes |
| `MATCH_MAX_RMS` | 6.0 | Procrustes+ICP residual ceiling (px) |
| `RECON_MAX_ROTATION_DEG` | 30 | Maximum rotation per match (degrees) |
| `EDGE_MATCHER_MIN_SCORE` | 0.55 | Siamese probability threshold |
| `MATCH_APPEARANCE_COS_MIN` | 0.55 | DINOv2 cosine gate |
| `MATCH_TEXT_LINE_MIN_CONT` | 0.30 | Text-line continuity gate |
| `SDT_SEAM_GAP_THRESH_PX` | 8 | Maximum seam gap in SDT check (px) |

### Physical Fit

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MAX_ATTACH_COST` | 400.0 | fit_cost ceiling for placement |
| `ORPHAN_MAX_ATTACH_COST` | 700.0 | Relaxed ceiling for orphan rescue |
| `SEAM_MAX_GAP_PX` | 3.0 | Maximum gap at seam attachment |
| `SEAM_MAX_OVERLAP_FRAC` | 0.18 | Maximum overlap fraction at seam |
| `SEAM_MIN_COVERAGE` | 0.55 | Minimum edge coverage fraction |

### Confidence Weights

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CONF_W_GEOMETRY` | 0.30 | Weight of geometric component |
| `CONF_W_APPEARANCE` | 0.20 | Weight of DINOv2 component |
| `CONF_W_STRIP_NCC` | 0.20 | Weight of strip NCC component |
| `CONF_W_TEXT_LINE` | 0.20 | Weight of text-line continuity |
| `CONF_W_PAPER_COLOR` | 0.10 | Weight of paper-colour component |

### Assembly

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ASSEMBLY_MIN_CONFIDENCE` | 0.45 | Minimum confidence to place a fragment |
| `ASSEMBLY_HIGH_CONFIDENCE_LOCK` | 0.92 | Confidence above which placement auto-locks |
| `ASSEMBLY_MUTUAL_TOP_K` | 3 | Top-K window for mutual-best seed detection |
| `ASSEMBLY_MAX_CANDIDATE_PAIRS` | 4000 | Cap on candidate pair list length |
| `ASSEMBLY_MAX_STEPS` | 1024 | Safety cap on MST iterations |

### Bundle Adjustment

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BA_MAX_ITER` | 200 | LM optimizer iterations |
| `BA_MAX_ROTATION_DEG` | 8.0 | Maximum rotation drift per fragment (degrees) |
| `BA_MAX_TRANSLATION_PX` | 60.0 | Maximum translation drift per fragment (px) |
| `BA_DENSE_EDGE_SAMPLES` | 32 | Dense edge constraint sample count per edge |

### Composition

| Parameter | Default | Effect |
|-----------|---------|--------|
| `COMP_SUPERSAMPLE` | 2 | Supersampling factor for warp (1 = off, 2 = recommended) |
| `COMP_SEAM_FILL_MAX_PX` | 6.0 | Maximum Voronoi fill distance for small gaps |
| `COMP_INK_THRESH` | 140 | Grayscale threshold below which a pixel is considered ink |

### Gap Fill

| Parameter | Default | Effect |
|-----------|---------|--------|
| `GAP_SMALL_FRAC` | 0.005 | Hole area < 0.5% of document bbox → small class |
| `GAP_MEDIUM_FRAC` | 0.05 | Hole area 0.5%–5% → medium class |
| `GAP_LARGE_CONTEXT_EXPAND_PX` | 20 | Context expansion for medium holes in LaMa |

---

## 10. Debug Artifacts

Every pipeline run writes a debug tree at `data/debug/<image_stem_YYYYMMDD_HHMMSS>/`.

```
data/debug/<stem>/
│
├── pipeline_meta.json              Timings per phase, n_fragments, scale_factor,
│                                   edge_matcher_loaded, missing_fragment, status
│
├── segmentation/
│   ├── 00_background_detection.png  Corner samples + BG colour swatch
│   ├── 01_raw_sam_overlay.png       All SAM candidate masks
│   ├── 02_after_appearance_filter.png  Paper candidates only
│   ├── 03_mask_raw_*.png            Pre-morphology per-fragment mask
│   ├── 04_mask_clean_*.png          Post-morphology per-fragment mask
│   ├── 05_mask_final_*.png          Final binary mask
│   ├── 06_final_fragments_overlay.png  Coloured contours with fragment IDs
│   ├── 07_crop_*.png               Cropped fragment images
│   ├── raw_masks_meta.json         SAM raw mask scores + BG distances
│   └── fragments_meta.json         [{id, area, bbox_xywh, centroid}]
│
├── contours/
│   ├── sdf_*.png                   Signed distance maps (white = inside)
│   ├── support_pts_*.png           Support points per fragment
│   ├── all_support_points.png      All fragments overlaid
│   └── contours_meta.json          Per-fragment edge count, perimeter, lengths
│
├── reconstruction/
│   ├── step_*.png                  Canvas snapshot after each MST placement
│   ├── merge_log.json              Step-by-step placement record
│   ├── assembly_summary.json       Overall stats: n_placed, seed, cache_hit_rate
│   ├── fragment_profiles.json      Per-fragment role, ink_density, anchor_strength
│   └── final_translations.json     {frag_id: {dx, dy, placed}}
│
├── composition/
│   ├── 01_raw_composite.png        Placed fragments, uncovered gaps visible
│   ├── 03_gap_mask.png             Inpaint mask
│   ├── 04_inpainted.png            After classical inpaint (pre-LaMa)
│   └── composition_meta.json       Canvas dims, offset, gap_pixels
│
└── inpainting/
    ├── 01_before.png               Canvas before LaMa
    ├── 02_holes_interior.png       Hole interior mask
    ├── 03_repair_mask.png          Unified scar + hole repair mask
    ├── 04_cleaned.png              Final output (also at data/output/<stem>.png)
    └── inpainting_meta.json        backend_used, mask_pixels, duration_s, device,
                                    refine, hole_counts, largest_hole_frac
```

The backend API serves all these via `GET /debug/image/{job_id}/{relative_path}` and aggregates the JSON metadata into `GET /debug/{job_id}`.

---

## 11. Docker Deployment

### `Dockerfile.backend`

- Base: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
- System packages: `libglib2.0-0 libgl1 libsm6 libxext6 libxrender1` (OpenCV runtime)
- `pip install -e sam2/` (editable install for config resolution)
- `pip install -r requirements_web.txt`
- Copies: `backend/`, `untorn/`, `models/`, `run.py`
- `EXPOSE 8000`; CMD: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

### `Dockerfile.frontend`

- **Builder stage:** `node:20-alpine` — `npm ci` + `npm run build`
- **Runner stage:** `node:20-alpine` — production deps only, `.next/` + `next.config.mjs`
- `EXPOSE 3000`; CMD: `npm start`
- `NEXT_TELEMETRY_DISABLED=1`

### `docker-compose.yml` (GPU)

```yaml
services:
  backend:
    image: ghcr.io/alisherbagyt/untorn-backend:v3.3
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data                                   # runtime data
      - ./sam2/sam2.1_hiera_small.pt:/app/sam2/...:ro     # model weights
      - ./lama/big-lama:/app/lama/big-lama:ro
    environment:
      TORCH_DEVICE: cuda
      LAMA_JIT_PATH: lama/big-lama/big-lama.pt
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    image: ghcr.io/alisherbagyt/untorn-frontend:v3.3
    ports: ["3000:3000"]
    environment:
      BACKEND_URL: http://backend:8000
    depends_on: [backend]
```

### CPU Override (`docker-compose.cpu.yml`)

Overrides `TORCH_DEVICE=cpu` and removes the GPU `deploy` block:
```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### `next.config.mjs` (Frontend Routing)

Rewrites `/api/*` → `${BACKEND_URL}/*` for server-side API calls. Sets `images.unoptimized = true` with `localhost:8000` as an allowed remote image hostname.

---

## 12. Training the Siamese Edge Matcher

The Siamese CNN (`models/edge_matcher.pt`) was trained on synthetic torn-edge pairs generated from PubLayNet documents.

### Data Pipeline

```bash
# 1. Extract documents from PubLayNet parquet
python tools/extract_publaynet.py

# 2. Generate synthetic torn fragments
python tools/tear_simulator.py

# 3. Create composite scan images
python tools/composite_generator.py

# 4. Build 80/20 train/val split
python tools/build_dataset_index.py

# 5. Extract (32, 256, 3) RGB edge strips → HDF5
python tools/build_edge_dataset.py
```

The edge strip format: 32 px wide (inward from boundary) × 256 px long (along edge) × 3 channels RGB, normalised. Positive pairs: adjacent torn edges of the same original document. Negative pairs: 1:3 ratio, sampled from non-adjacent edges.

### Training

```bash
python tools/train_edge_matcher.py
# AdamW optimizer, cosine LR schedule, AMP (fp16), ~50 epochs
# Saves checkpoint to models/edge_matcher.pt
```

### Evaluation

```bash
python tools/eval_edge_matcher.py
# Reports AUC on val.h5 against 4 baselines: pixel NCC, SIFT, ORB, random
# Current val_AUC = 0.924
```

### Architecture (`edge_matcher_model.py`)

Two-tower shared-weight CNN:
- Input: `(32, 256, 3)` RGB strip per tower
- 4 convolutional stages with BatchNorm + ReLU + MaxPool
- Global average pool → 256-dim embedding
- Cosine similarity with learnable temperature scalar
- 3-DOF pose head (auxiliary, used during training only)
- ~520 000 parameters, ~2 MB checkpoint

---

## 13. Design Decisions

### Why SAM 2.1 for Segmentation

SAM 2.1's automatic mask generator is the only off-the-shelf segmentation model that reliably handles arbitrary shapes, sizes, and colours without task-specific fine-tuning. Fragment shapes are too irregular for classical region-growing, and the number of fragments is unknown. SAM's multi-scale grid-point approach finds even very small or thin fragments. The appearance filter (LAB ΔE from auto-detected background) replaces SAM's built-in quality score as the paper/background discriminator, giving universality across scanner backgrounds.

### Why Smith-Waterman for Curvature Matching

Torn paper produces irregular curvature profiles — the physics of fracture creates characteristic, reproducible signatures. SW local alignment handles three important properties: (1) only a sub-arc of one edge may be visible (corner pieces, tip damage); (2) SW is robust to small quantisation noise in contour extraction; (3) the 80-sample discretisation reduces sub-pixel jitter sensitivity without losing distinctiveness. SW returns matching sub-arc indices directly, giving Procrustes its point correspondences without an additional correspondence-finding step.

### Why Procrustes Before ICP

Procrustes on SW-matched samples gives a globally optimal rigid alignment of the sub-arc, with no initial pose required. ICP then corrects the residual drift at the full edge level. Starting ICP from scratch (no initial pose) would fail on curved edges where the convergence basin is narrow. The Procrustes → coarse ICP → fine ICP sequence is numerically stable and fast (< 20 ms per pair on GPU).

### Why SDT Physical Gates

Curvature matching alone can align edges that geometrically cannot fit — two convex edges with similar curvature profiles, for example. The SDT gate directly measures physical penetration (overlap) and gap. No amount of curvature or appearance score can override a physical impossibility. This gate is the primary reason the system does not produce chimeric assemblies where fragments interpenetrate.

### Why DINOv2 Instead of Pixel Correlation

DINOv2 ViT-S/14 features capture paper texture, ink distribution, and printing characteristics at a semantic level, at 14 px spatial resolution. Raw pixel NCC fails when one fragment is slightly lighter than its neighbour (scanner variation), or when text does not happen to fall near the seam. DINOv2 features are much more invariant to these photometric differences while still being distinctive enough to discriminate adjacent from non-adjacent fragments.

### Why a Trained Siamese CNN in Addition to DINOv2

DINOv2 operates at patch granularity (14 px). At actual seam boundaries, the relevant structure is finer: paper grain, exact colour blend, ink coverage at the very edge. The Siamese CNN operates at pixel level on a 32×256 px strip, trained specifically on the task of edge matching. It complements DINOv2 by catching cases where patch-level appearance agrees but the actual edge texture is inconsistent.

### Why Bundle Adjustment After MST

In chain-like fragment placements (A→B→C→D), each pairwise Procrustes alignment contributes a small angular error. By 10 fragments, cumulative drift can reach several degrees. Bundle adjustment redistributes this error globally via a Levenberg-Marquardt solve over all poses jointly. The dense edge constraint additionally enforces that torn-edge polylines physically weld together, not just that matched sample points coincide — eliminating residual gaps that the sparse SW constraint misses.

### Why MST Instead of Global Greedy

Global greedy (pick the highest-confidence pair globally, attach, repeat) accumulates errors. An early wrong placement corrupts all subsequent neighbours. The MST priority-queue approach naturally propagates only from already-placed fragments, and the seam gates at attach time prevent physically bad placements from propagating regardless of confidence score. The mutual-best seed selection ensures the anchor is the most reliable possible starting point.

### Why No Database for Jobs

The system runs on a single GPU workstation with one user at a time. An in-memory dict guarded by a threading lock is simpler, faster, and has zero setup cost. The single-GPU semaphore already serialises all pipeline execution, so concurrent state corruption is not a concern. A persistent store would be needed only for multi-machine or multi-user deployments, which is not the design target.

### Why WebSocket Primary with Polling Fallback

The pipeline runs for 15–180 seconds depending on fragment count. Polling at 800 ms wastes connection overhead; WebSocket gives sub-second latency at near-zero cost. The polling fallback handles proxy environments that block or drop WebSocket upgrades.

### Why LaMa for Seam Cleaning

LaMa (Large Mask inpainting) is state-of-the-art for filling large, irregular regions with photorealistic paper texture. The key operation is the scar mask construction: by excluding pixels below grayscale threshold 140 (ink), LaMa never inpaints over text — it fills only bare paper regions. This is critical: a naive dilated-seam mask would hallucinate text across character strokes. The text-protection mask means LaMa seamlessly heals paper grain at seams while leaving all text untouched.
