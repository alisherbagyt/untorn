# UNTORN — Architecture & Design Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Pipeline Phases](#pipeline-phases)
   - [Phase 0: Preprocessing](#phase-0-preprocessing)
   - [Phase 1: Segmentation](#phase-1-segmentation)
   - [Phase 2: Contour Analysis](#phase-2-contour-analysis)
   - [Phase 3: Hierarchical Reconstruction](#phase-3-hierarchical-reconstruction)
   - [Phase 4: Composition](#phase-4-composition)
   - [Phase 5: LaMa Inpainting](#phase-5-lama-inpainting)
4. [Core Matching Algorithm](#core-matching-algorithm)
5. [Backend API](#backend-api)
6. [Frontend](#frontend)
7. [Configuration Reference](#configuration-reference)
8. [Debug Artifacts](#debug-artifacts)
9. [Key Design Decisions](#key-design-decisions)

---

## Overview

UNTORN reconstructs torn paper documents from a single scanned photograph. Given an image containing scattered paper fragments on a contrasting background, the system:

1. Detects and isolates every fragment (segment)
2. Analyses each fragment's torn edges using curvature features
3. Algorithmically matches and reassembles fragments into the original document
4. Outputs a high-quality reconstructed image with seams cleaned by neural inpainting

The reconstruction is fully automatic and works without any prior knowledge of the document's content, layout, or number of fragments. A web UI provides real-time pipeline feedback and an interactive assembly board for manual refinement.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Next.js 14 Frontend  (localhost:3000)                    │
│  UploadZone → ProcessingView → ResultsView (6 tabs)       │
│  AssemblyView (interactive canvas with export)            │
└────────────────────┬─────────────────────────────────────┘
                     │  HTTP / WebSocket
┌────────────────────▼─────────────────────────────────────┐
│  FastAPI Backend  (localhost:8000)                        │
│  /process  /status  /result  /debug  /board  /ws         │
│  In-memory job store + single-GPU semaphore               │
└────────────────────┬─────────────────────────────────────┘
                     │  subprocess (stdout parsing)
┌────────────────────▼─────────────────────────────────────┐
│  UNTORN Python Pipeline  (untorn/)                        │
│  pipeline.py → preprocess → segment → contours →         │
│  reconstruct → compose → inpaint                         │
│  Dependencies: SAM 2.1, LaMa, OpenCV, scipy, networkx    │
└──────────────────────────────────────────────────────────┘
```

### Entry Points

| Mode | Command |
|------|---------|
| CLI  | `python run.py <image> [-o output.png]` |
| Web  | `POST /process` (multipart upload) |

The backend launches the pipeline as a child process and streams stdout line-by-line. Progress markers embedded in the pipeline's print statements (`[PHASE N]`, `Phase N complete`, `DONE`) are parsed to drive the job status API.

### File Layout

```
untorn/               # Core algorithm (Python)
  pipeline.py         # Phase orchestrator
  preprocess.py       # Phase 0
  segmentation.py     # Phase 1
  contours.py         # Phase 2
  reconstruction.py   # Phase 3 (7 sub-phases)
  neighbors.py        # Phase 3A — neighbor graph
  matching.py         # Core edge-pair matching
  composition.py      # Phase 4
  inpainting.py       # Phase 5
  config.py           # All tunable constants
backend/
  main.py             # FastAPI app, all endpoints
  pipeline_wrapper.py # Subprocess launcher, phase marker parsing
  services/
    job_service.py    # In-memory job store, semaphore
  models/
    schemas.py        # Pydantic request/response models
frontend/
  app/
    page.tsx          # SPA root: upload → processing → results state machine
    layout.tsx        # Root layout
  components/
    UploadZone.tsx
    ProcessingView.tsx
    ResultsView.tsx
    FragmentTimeline.tsx
    views/
      OverviewView.tsx
      SegmentationView.tsx
      ContoursView.tsx
      ReconstructionView.tsx
      CompositionView.tsx
      AssemblyView.tsx    # 1 360-line interactive canvas
  lib/
    api.ts            # Backend client & TypeScript types
    utils.ts          # Helpers, phase label mapping (Russian)
data/
  input/              # Uploaded images
  output/             # Final reconstructed PNGs
  debug/<stem>/       # Per-image debug tree
```

---

## Pipeline Phases

### Phase 0: Preprocessing

**Module:** `untorn/preprocess.py` — `prepare_image(image_rgb, debug_dir)`

**Problem:** SAM 2.1 and matching algorithms are tuned for images around 1 500 px on the long side. Very large scans (e.g., 4 000 px TIFF) would exceed GPU VRAM and make matching slow.

**Solution:**
- If the longest dimension exceeds 1 500 px, downscale with `cv2.INTER_AREA` (avoids aliasing).
- Record `scale_factor = original_longest / working_longest`.
- Keep the full-resolution original in memory for the composition phase.

**Progress markers:** `[PHASE 0]` → 5%, `Phase 0 complete` → 10%

---

### Phase 1: Segmentation

**Module:** `untorn/segmentation.py` — `segment_fragments(image_rgb, debug_dir)`

**Goal:** Isolate every paper fragment as a binary mask, without knowing anything about paper color, fragment count, or background.

#### 1A — SAM 2.1 Automatic Mask Generator

Uses `sam2.1_hiera_small.pt` with a dense 32×32 point grid. SAM returns hundreds of candidate masks at multiple scales.

Key parameters:
```
points_per_side     = 32
pred_iou_thresh     = 0.80   (relaxed — appearance filter is the real gate)
stability_score     = 0.88
min_mask_area       = 500 px
crop_n_layers       = 1      (multi-scale crop pass)
```

#### 1B — Background Color Auto-Detection

Samples 5% of pixels along each image edge (LAB color space). The modal cluster of these edge samples is the background color. A LAB ΔE threshold (default 18.0) then separates background from paper.

This is universal — works on black backgrounds, white tables, coloured cloth, etc.

#### 1C — Appearance-Based Filtering

For each SAM mask, compute the mean LAB color of the masked region. Masks whose color is within `BG_DIST_LAB_THRESH` of the detected background color are discarded. This replaces fragile area-ratio heuristics.

#### 1D — Post-Processing Chain

1. **Containment-aware hole filling** — if a background-colored mask is entirely enclosed by a paper mask, fill it (handles degraded paper with holes, yellowing, ink stains).
2. **Overlap merging** — SAM often generates overlapping duplicates; merged by IoU threshold.
3. **Morphological cleanup** — close small holes, remove isolated islands, smooth boundary. Kernel sizes scale with image diagonal for DPI invariance.

**Fragment output** (one dict per fragment):
```python
{
    "id": int,
    "mask": HxW uint8,          # 255 = fragment pixel
    "bbox": (x, y, w, h),
    "area": int,
    "contour": Nx2 array,
    "centroid": [cx, cy]
}
```

**Progress markers:** `[PHASE 1]` → 12%, `Phase 1 complete` → 44%

**Debug outputs:**
```
00_background_detection.png    corner samples + BG color swatch
01_raw_sam_overlay.png         all SAM masks
02_after_appearance_filter.png paper candidates only
03_mask_raw_*.png              per-fragment, pre-morphology
04_mask_clean_*.png            per-fragment, post-morphology
05_mask_final_*.png            final mask
06_final_fragments_overlay.png colored contours with IDs
07_crop_*.png                  individual fragment crops
raw_masks_meta.json
fragments_meta.json
```

---

### Phase 2: Contour Analysis

**Module:** `untorn/contours.py` — `analyze_fragments(fragments, work_rgb, debug_dir)`

**Goal:** Extract rich edge descriptors needed for matching — curvature feature strings, signed distance transforms, and torn/factory edge classification.

#### 2A — Support Point Extraction

Douglas-Peucker polygonal approximation with `ε = 0.4%` of perimeter length. Support points are the corners where the torn profile changes direction significantly.

#### 2B — Edge Segment Descriptors

Between consecutive support points, compute:
- Length, angle, midpoint, inward normal vector
- `is_torn` classification via RANSAC line fit (low inlier ratio → torn)

#### 2C — Curvature Feature Strings

Sample 80 evenly-spaced points around the contour. Compute discrete curvature `κ = dθ/ds` (change in tangent angle per arc-length unit) and quantize into an 80-element float vector. This is the primary matching signal.

#### 2D — Signed Distance Transform (SDT)

For each fragment mask, compute `SDT` where:
- Positive values = interior (distance from boundary inward)
- Negative values = exterior (distance from boundary outward)
- Zero = boundary

The SDT is used in Phase 3 to measure physical penetration and gap between candidate pairs.

**Fragment enrichment:**
```python
frag["edges"]            # list of edge segment dicts
frag["_curv"]            # 80-element curvature string
frag["_sdt_interior"]    # signed distance map
frag["boundary_pixels"]  # Nx2 boundary pixel array
```

**Progress markers:** `[PHASE 2]` → 45%, `Phase 2 complete` → 54%

---

### Phase 3: Hierarchical Reconstruction

**Modules:** `untorn/reconstruction.py`, `untorn/neighbors.py`, `untorn/matching.py`

**Goal:** Find the SE(2) rigid transform (rotation + translation) for each fragment that places it in its original document position.

The reconstruction is hierarchical — it grows from trusted anchors (document corners) outward, rather than doing a global search. This limits cumulative error and makes conflict resolution tractable.

#### 3A — Neighbor Discovery (`neighbors.py`)

Build a sparse proximity graph to define which fragment pairs are worth matching.

**Graph construction:**
1. kNN (k=6) on fragment centroids (Euclidean L2).
2. Delaunay triangulation on centroids (catches edge pieces missed by kNN).
3. Union of both → candidate pairs.

**Filtering per pair:**
- Boundary-to-boundary distance via KDTree: must be ≤ 200 px.
- Outward-normal facing check: normals dot product ≥ 0.1 (fragments must face each other).
- Directional classification: offset vector → `{right, left, down, up}`.

**Corner identification:**
Four fragments at document corners (TL, TR, BL, BR) found via 2D angle-scan heuristic on centroid positions.

#### 3B — Pair Matching (`matching.py`)

For every neighbor pair `(i, j)`, find the best rigid alignment. See [Core Matching Algorithm](#core-matching-algorithm) for full details.

A `_MatchCache` memoizes `(i, j) → match_dict` to avoid redundant computation when the same pair is evaluated in multiple assembly sub-phases.

#### 3C — Sub-Phase II: Corner Seed Batching

- Anchor each of the 4 corner fragments at identity (their scan positions become the global gauge frame).
- For each corner, attach its best horizontal neighbor (right for TL/BL, left for TR/BR) by lowest `fit_cost`.
- Result: four 2-piece baseline clusters at document corners.

#### 3D — Sub-Phase III: Vertical Expansion

- From each 2-piece corner cluster, attach the best vertical neighbor (down for top corners, up for bottom corners).
- Result: four L-shaped 3-piece clusters.

#### 3E — Sub-Phase IV: Perimeter Frame Infilling

- Walk perimeter edges between adjacent corner clusters.
- Attach fragments that own at least one factory (straight) edge > 40 px.
- Single-piece-at-a-time with re-scan after each successful attach.
- Confidence threshold: ≥ 0.55.

#### 3F — Sub-Phase V: Interior Infilling

Confidence-first priority queue over all unplaced pieces:

| Confidence | Action |
|------------|--------|
| ≥ 0.95 | Auto-lock immediately |
| 0.40 – 0.95 | Attach via bipartite matching on conflicting pairs |
| < 0.40 | Force-fit phase, constrained by document aspect-ratio (1.0–2.5) |

**Conflict resolution:** When two unplaced fragments compete for the same anchor slot, run `networkx.max_weight_matching()` on the bipartite conflict graph.

**Global overlap check:** Warp candidate onto dynamic canvas; reject if pixel intersection exceeds 8% of the smaller fragment's area.

#### 3G — Cluster Jitter (Post-Placement Refinement)

After bulk placement, run ICP micro-refinement for each placed fragment against the concatenated boundary pixels of its placed neighbors (2 iterations, 20 px cap).

#### 3H — Orphan Rescue

Final pass for any fragment unreachable via the neighbor graph. Evaluate against every placed fragment using loosened fit_cost ceiling (700 vs. 400). Lowest cost wins.

#### 3I — Global Bundle Adjustment

Levenberg-Marquardt pose-graph optimization over all non-corner fragment poses `(θ, tx, ty)`.

**Constraints:** Every cached edge-pair match contributes seam-point coincidence equations:
```
T_i(matched_a) ≈ T_j(matched_b)   for all (i,j) in match cache
```

**Safety caps:** ±8° rotation, ±60 px translation per fragment from placement estimate.

**Iterations:** 200 max, `func_tol = 1e-6`.

**Output:** `dict[fragment_id → 3×3 SE(2) homogeneous affine matrix]`

```python
M = [[R, t],   # R: 2×2 rotation, t: 2D translation
     [0, 1]]
```

**Progress markers:** `[PHASE 3]` → 55%, `Phase 3 complete` → 81%

**Debug outputs:**
```
neighbors.json           neighbor graph dump
match_scores.json        all candidate matches ranked by fit_cost
merge_log.json           step-by-step placement record
final_translations.json  per-fragment transforms
```

---

### Phase 4: Composition

**Module:** `untorn/composition.py` — `compose_final(image_rgb, fragments, transforms, debug_dir)`

**Goal:** Render the reconstructed document at full resolution.

**Steps:**
1. Upscale transforms: multiply translation components by `scale_factor`; rotation is scale-invariant.
2. Compute canvas AABB from all transformed fragment bounding boxes; pad 10 px.
3. Place fragments via `cv2.warpAffine()`. Sort by area descending (large pieces first, small overlay on top — avoids edge bleed-through).
4. Build coverage mask (union of all placed fragment alphas).
5. Fill gaps with classical inpainting (OpenCV Telea / Fast Marching) as a placeholder for Phase 5.

**Output dict:**
```python
{
    "canvas":    HxWx3 uint8,   # RGB composite
    "coverage":  HxW uint8,     # 0=gap, 255=covered
    "gap_mask":  HxW uint8,     # pixels to inpaint
    "crop_bbox": (x, y, w, h)   # tight content crop
}
```

**Progress markers:** `[PHASE 4]` → 82%, `Phase 4 complete` → 89%

**Debug outputs:**
```
01_raw_composite.png    placed fragments, white gaps visible
03_gap_mask.png         inpaint mask
04_inpainted.png        after classical inpaint (pre-LaMa)
composition_meta.json   canvas dims, per-fragment placement
```

---

### Phase 5: LaMa Inpainting

**Module:** `untorn/inpainting.py` — `clean_final(canvas, coverage, gap_mask, debug_dir, refine=False)`

**Goal:** Remove visible seams along fragment boundaries using neural inpainting, while preserving document text.

#### Scar Mask Construction

1. Build a 4 px half-width ring around each seam.
2. Mask out ink pixels (grayscale < 140) — avoid inpainting over real text.
3. The resulting mask covers only bare paper scar regions.

#### LaMa Backends (in priority order)

| Backend | How | Notes |
|---------|-----|-------|
| TorchScript JIT | `big-lama.pt` loaded with `torch.jit.load` | Zero extra dependencies |
| simple-lama-inpainting | pip package | Lightweight wrapper |
| saicinpainting | Full LaMa codebase | Original; most flexible |
| Classical fallback | Distance-weighted color blend | Used when no LaMa available |

#### Tiled Inference

For full-resolution images exceeding GPU VRAM, the canvas is processed in 512 px tiles with 64 px overlap; tiles are blended with a cosine weight window to avoid visible seams.

**Progress markers:** `[PHASE 5]` → 90%, `Phase 5 complete` → 98%, `DONE` → 100%

**Debug outputs:**
```
01_before.png         uncropped composite before LaMa
02_scar_mask.png      LaMa inpaint mask (scar regions only)
03_cleaned.png        final cleaned output
inpainting_meta.json  backend used, mask_pixels, duration, device
```

---

## Core Matching Algorithm

Implemented in `untorn/matching.py::match_pair(edge_a, edge_b, frag_a, frag_b)`.

This is the heart of the system — every placement decision ultimately rests on the score returned here.

### Step 1 — Smith-Waterman Local Alignment

Aligns the 80-element curvature feature strings of two edges as if they were biological sequences.

```
Match score:   +2.0   (curvature values within ε₁)
Near penalty:  -0.1   (within ε₂ but not ε₁)
Far penalty:   -2.0   (outside ε₂)
Gap penalty:   -1.0

Min score to accept:    5.0
Min aligned samples:    6 (out of 80)
```

The alignment returns corresponding sample indices `(matched_a, matched_b)` — the sub-arc where curvature profiles agree.

### Step 2 — Procrustes Rigid Fit

SVD-based least-squares rigid alignment of the 2D point sets `matched_a` and `matched_b`.

Returns: `(R: 2×2, translation: [tx, ty], rms_error: float)`.

Gate: `rms_error ≤ MATCH_MAX_RMS` (default 6.0 px).

### Step 3 — ICP Jitter Correction

Two-phase Iterative Closest Point refinement after Procrustes:

| Phase | Correspondence dist | Purpose |
|-------|---------------------|---------|
| Coarse | 25 px | Pull drifted edges together |
| Fine   | 6 px  | Sub-pixel seam closure |

Safety cap: rotation drift ≤ 5° total from Procrustes estimate (prevents local-minima flips).

### Step 4 — Signed Distance Transform Physical Gates

After aligning fragment B onto fragment A:

1. **Penetration check:** Look up B's boundary pixels in A's SDT. If > 20% of contour pixels are inside A (positive SDT), or maximum penetration depth > 10 px → reject.
2. **Gap check:** Median distance at matched seam points must be < 8 px.

These are hard physical constraints — no amount of curvature agreement overrides them.

### Step 5 — Full-Edge Fit Evaluation

Evaluates the alignment quality over the *entire* edge (not just the SW-matched sub-arc):

| Metric | Meaning |
|--------|---------|
| `fit_overlap_px` | SDT-measured penetration of B into A |
| `fit_gap_px` | Mean nearest-neighbour distance along edge |
| `fit_coverage` | Fraction of edge-A with matching point within 2 px of warped edge-B |

```
fit_cost = 1.5 × fit_overlap_px
         + 2.0 × fit_gap_px
         + 80  × (1 - fit_coverage)
```

Lower `fit_cost` = better physical alignment. This is the primary ranking signal.

### Step 6 — Confidence Score

Four-component score, all in [0, 1]:

| Component | Formula | Meaning |
|-----------|---------|---------|
| `sarea` | `min(rms / avg_arc × 5, 1)` | RMS residual normalized by arc length |
| `slen` | `1 - n_aligned / max_len` | Penalty for short alignments (prefer long arcs) |
| `scorr` | `(1 - corrcoef(curv_a, curv_b)) / 2` | Curvature cross-correlation penalty |
| `sappearance` | NCC of 8px interior-facing pixel strips | Visual continuity of paper texture/color |

```
stotal     = sarea + slen + scorr + 0.5 × sappearance   # in [0, ~3.5]
confidence = clip(1 - stotal / 2.5, 0, 1)
```

`sappearance` is computed by `_score_edge_appearance()` — it samples an 8px-wide strip of pixels facing inward from each matched edge and computes normalized cross-correlation (NCC) between the two strips. This is a classical pixel-based measure, not a learned feature.

High confidence (≥ 0.95) → auto-lock during Phase 3 interior infilling.

### Pre-Filters (Fast Rejects)

Applied before running SW to prune obviously impossible pairs:

- Both edges must be classified as torn (`is_torn = True`); factory edges never seam.
- Outward normals must face each other (dot product ≥ 0.1).
- `MATCH_REJECT_DIRECT`: rejects geometrically parallel curvature strings (same curve can't mate with itself).
- RANSAC cycle-consistency re-ranking: uses triangle-inconsistency vote counts to soft-demote unlikely matches across the whole graph.

### Match Result

```python
{
    "frag_i": int,
    "frag_j": int,
    "R": 2×2 ndarray,
    "translation": [tx, ty],
    "angle": float,           # radians
    "rms": float,
    "confidence": float,      # [0, 1]
    "fit_cost": float,
    "fit_overlap_px": float,
    "fit_gap_px": float,
    "fit_coverage": float,
    "edge_i": dict,
    "edge_j": dict,
    "matched_a": ndarray,
    "matched_b": ndarray
}
```

---

## Backend API

**Framework:** FastAPI. **Base URL:** `http://localhost:8000`

### Endpoints

#### Processing

| Method | Path | Description |
|--------|------|-------------|
| POST | `/process` | Upload image (multipart), create job, launch pipeline |
| GET | `/status/{job_id}` | Poll job status |
| GET | `/result/{job_id}` | Download final PNG (only when `status == "done"`) |
| WS | `/ws/{job_id}` | Real-time status stream (every 400–500 ms) |

**POST /process** — Request: `multipart/form-data` with `file` field.
Response:
```json
{ "job_id": "uuid", "message": "Processing started" }
```

**GET /status/{job_id}** — Response:
```json
{
  "job_id": "uuid",
  "status": "queued | processing | done | error",
  "progress": 0-100,
  "current_phase": "segmentation",
  "logs": ["last 15 stdout lines"],
  "error": null
}
```

#### Debug Inspection

| Method | Path | Description |
|--------|------|-------------|
| GET | `/debug/{job_id}` | All pipeline metadata + image paths |
| GET | `/debug/image/{job_id}/{path}` | Serve individual debug PNG/TIFF |

`/debug/{job_id}` returns a `DebugResponse` with:
- `pipeline_meta`: timing, fragment count, image sizes, status
- `fragments`: per-fragment area, bbox, centroid
- `contours`: edge count, perimeter, edge lengths
- `neighbors`: pair distances
- `steps`: normalized merge log (step, phase, anchor, attached, dx, dy, gap_score)
- `translations`: final per-fragment dx/dy/rotation
- `composition`: canvas dims, gap stats
- `inpainting`: backend used, mask size, duration
- `paths`: URLs to all debug images

#### Assembly Board

| Method | Path | Description |
|--------|------|-------------|
| GET | `/board/data/{job_id}` | Canvas dims + per-fragment placement info |
| GET | `/board/fragment/{job_id}/{fragment_id}` | Transparent PNG of one fragment |
| POST | `/board/export/{job_id}` | Composite user-positioned fragments and optionally apply LaMa |

**POST /board/export** — Request:
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
Response: PNG file (with `Content-Disposition: attachment` header).

#### Utility

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"status": "ok", "timestamp": "..."}` |
| GET | `/jobs` | List all known jobs |

### Job Lifecycle

```
POST /process
    ↓
create_job()  →  status=queued, progress=0
    ↓
run_pipeline() — daemon thread, acquires PIPELINE_SEMAPHORE
    ↓
python run.py <image>  (subprocess)
    ↓  stdout line-by-line
parse phase markers → update_job()
    ↓
"DONE"  →  status=done, progress=100
"FAILED" →  status=error
```

**Single-GPU semaphore:** Only one pipeline subprocess runs at a time; additional submissions queue.

**In-memory store:** Job state lives in a `dict[job_id → state]` guarded by a `threading.Lock`. No database — jobs are lost on server restart.

---

## Frontend

**Framework:** Next.js 14.2.5 (App Router), React 18, TypeScript, Tailwind CSS.

### Application Flow

The app is a single-page application with three top-level states managed in `app/page.tsx`:

```
upload   →   processing   →   results
UploadZone    ProcessingView   ResultsView (6 tabs)
```

State transitions:
- `upload → processing`: user drops image, `POST /process` succeeds
- `processing → results`: WebSocket delivers `status == "done"`
- `results`: fetch `GET /debug/{job_id}`, render all tabs

### ResultsView Tabs

| Tab | Component | Contents |
|-----|-----------|----------|
| overview | `OverviewView` | Final image, 4-stat grid, timing bar chart, download button |
| segmentation | `SegmentationView` | SAM overlay, fragment carousel with crop/mask/SDF |
| contours | `ContoursView` | Perimeter chart, support-point radar, edge-length bar chart, SDF vis |
| reconstruction | `ReconstructionView` | Step-through merge log, gap-score line chart, adjacency |
| composition | `CompositionView` | Layer toggle (raw/gap/inpainted/cleaned), gap coverage stats |
| assembly | `AssemblyView` | Interactive drag-and-drop canvas |

### AssemblyView — Interactive Canvas

A fully custom canvas component (1 360 lines) for manual fragment placement and export.

**Features:**
- Drag-and-drop fragments; zoom 5%–500% (wheel, pinch)
- Rotation: on-canvas handle drag or right-click drag
- Layer panel: visibility toggle, z-order reorder
- Keyboard shortcuts:
  - `Arrow` keys: nudge 1 px (+ `Shift` = 10 px)
  - `[` / `]`: rotate ±1° (+ `Shift` = ±15°)
  - `Space` hold: pan mode
  - `Ctrl+Z`: undo (50-operation stack)
  - `Esc`: deselect
- Grid overlay toggle
- Export: 1×/2×/4× scale, optional LaMa `clean` + `refine`

Fragment images are loaded once from `/board/fragment/{job_id}/{id}` as transparent PNGs, cached as Blob URLs (revoked on unmount).

### Real-Time Updates

WebSocket (`/ws/{job_id}`) is the primary channel. Payload every ~450 ms:
```json
{ "status": "...", "progress": 67, "current_phase": "reconstruction", "logs": [...] }
```
Falls back to polling (`GET /status`, 800 ms interval) if WS connection fails.

### Image Preloading

`ResultsView` preloads all debug image URLs into `new Image()` objects in a background effect, so tab switches are instant after the first load.

### API Client (`lib/api.ts`)

Typed wrappers for all backend calls. Key types:
```typescript
interface JobStatus { job_id, status, progress, current_phase, logs, error }
interface DebugData { pipeline_meta, fragments, contours, neighbors, steps, translations, composition, inpainting, paths }
interface BoardData { canvas: { width, height }, fragments: FragmentBoardItem[] }
interface BoardExportRequest { fragments, canvas_width, canvas_height, scale, clean, refine }
```

### Styling

- Tailwind CSS 3.4.7 with custom palette (accent indigo `#4F46E5`)
- Lucide React icons
- Recharts for bar, line, and radar charts
- Russian UI labels via `phaseLabel` mapping in `utils.ts`

---

## Configuration Reference

All constants live in `untorn/config.py`. Key parameters grouped by subsystem:

### SAM 2.1 Segmentation

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SAM2_POINTS_PER_SIDE` | 32 | Grid density (32×32 = 1 024 seed points) |
| `SAM2_PRED_IOU_THRESH` | 0.80 | SAM internal quality filter |
| `SAM2_STABILITY_THRESH` | 0.88 | SAM internal stability filter |
| `SAM2_MIN_MASK_AREA` | 500 px | Minimum raw mask area (pre-appearance filter) |
| `BG_DIST_LAB_THRESH` | 18.0 | LAB ΔE for paper vs. background |
| `MIN_FRAGMENT_AREA_FRAC` | 0.0003 | Min fragment area as fraction of image |
| `MAX_FRAGMENT_AREA_FRAC` | 0.90 | Max (rejects near-full-image backgrounds) |
| `MAX_FRAGMENTS` | 40 | Hard cap on fragment count |

### Curvature & Smith-Waterman

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CURV_N_SAMPLES` | 80 | Feature string length |
| `SW_MATCH_SCORE` | +2.0 | Reward for aligned curvature |
| `SW_FAR_PENALTY` | -2.0 | Penalty for mismatched curvature |
| `SW_GAP_PENALTY` | -1.0 | Gap open/extend penalty |
| `SW_MIN_SCORE` | 5.0 | Minimum alignment score |
| `SW_MIN_ALIGNED` | 6 | Minimum aligned sample count |

### Physical Fit Gating

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MATCH_THRESHOLD` | 1.8 | `s_total` ceiling for match acceptance |
| `MATCH_MAX_RMS` | 6.0 px | Procrustes + ICP residual ceiling |
| `MATCH_CLUSTER_MAX_RMS` | 3.0 px | Stricter gate for cluster-level merges |
| `MAX_ATTACH_COST` | 400.0 | `fit_cost` ceiling for placements |
| `ORPHAN_MAX_ATTACH_COST` | 700.0 | Looser ceiling for orphan rescue |
| `RECON_OVERLAP_THRESH` | 0.08 | Max allowed canvas pixel overlap (8%) |

### Neighbor Graph

| Parameter | Default | Effect |
|-----------|---------|--------|
| `NEIGHBOR_K` | 6 | kNN density |
| `NEIGHBOR_MAX_EDGE_DIST_PX` | 200 px | Boundary-to-boundary max |
| `FACING_COSINE_MIN` | 0.1 | Outward-normal facing threshold |

### Confidence Thresholds

| Parameter | Default | Effect |
|-----------|---------|--------|
| `HIGH_CONFIDENCE_LOCK` | 0.95 | Auto-lock threshold |
| `PERIMETER_MIN_CONFIDENCE` | 0.55 | Perimeter frame inclusion |
| `INTERIOR_MIN_CONFIDENCE` | 0.40 | Interior infilling gate |
| `CONFIDENCE_STOTAL_SPAN` | 2.5 | Confidence normalization span |

### ICP & Jitter

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ICP_MAX_ITER` | 15 | ICP iterations per pair |
| `ICP_COARSE_DIST_PX` | 25.0 | Coarse phase correspondence radius |
| `ICP_MAX_CORR_DIST_PX` | 6.0 | Fine phase correspondence radius |
| `ICP_MAX_DRIFT_DEG` | 5.0 | Max rotation refinement from Procrustes |
| `CLUSTER_JITTER_ITERS` | 2 | Post-placement jitter passes |
| `CLUSTER_JITTER_TOL_PX` | 20.0 | Jitter translation cap |

### Bundle Adjustment

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BA_ENABLED` | True | Enable global pose-graph optimization |
| `BA_MAX_ITER` | 200 | LM optimizer iterations |
| `BA_MAX_ROTATION_DEG` | 8.0 | Safety drift cap (rotation) |
| `BA_MAX_TRANSLATION_PX` | 60.0 | Safety drift cap (translation) |

### RANSAC Cycle-Consistency

| Parameter | Default | Effect |
|-----------|---------|--------|
| `RANSAC_ENABLED` | True | Enable global soft re-ranking |
| `RANSAC_CYCLE_GATE_PX` | 40.0 | Triangle inconsistency threshold |
| `RANSAC_VOTE_BONUS` | 0.04 | Fit-cost reduction per consistent triangle |
| `RANSAC_DROP_MIN_CON` | 6 | Hard drop: >6 failing triangles with bad pro/con ratio |

---

## Debug Artifacts

Every pipeline run writes a debug tree at `data/debug/<image_stem>/`:

```
data/debug/<stem>/
│
├── pipeline_meta.json          Timings per phase, fragment count, status ("SUCCESS" | "FAILED_*")
│
├── segmentation/
│   ├── 00_background_detection.png
│   ├── 01_raw_sam_overlay.png
│   ├── 02_after_appearance_filter.png
│   ├── 03_mask_raw_*.png
│   ├── 04_mask_clean_*.png
│   ├── 05_mask_final_*.png
│   ├── 06_final_fragments_overlay.png
│   ├── 07_crop_*.png
│   ├── raw_masks_meta.json
│   └── fragments_meta.json
│
├── contours/
│   ├── sdf_*.png               Signed distance maps (white=inside)
│   ├── support_pts_*.png       Support points per fragment
│   ├── all_support_points.png  All fragments overlaid
│   └── contours_meta.json
│
├── reconstruction/
│   ├── neighbors.json          Full neighbor graph
│   ├── match_scores.json       All candidate matches ranked by fit_cost
│   ├── merge_log.json          Step-by-step placement (phase, anchor, attached, confidence, fit_cost, angle, rms)
│   └── final_translations.json Per-fragment SE(2) transforms
│
├── composition/
│   ├── 01_raw_composite.png
│   ├── 03_gap_mask.png
│   ├── 04_inpainted.png
│   └── composition_meta.json
│
└── inpainting/
    ├── 01_before.png
    ├── 02_scar_mask.png
    ├── 03_cleaned.png          Final output (also copied to data/output/)
    └── inpainting_meta.json    Backend, mask_pixels, duration_s, device, refine
```

The backend API serves all these files via `GET /debug/image/{job_id}/{path}` and aggregates metadata into the `GET /debug/{job_id}` response for the frontend.

---

## Key Design Decisions

### Locality-First Assembly (Hierarchical Growth)

**Alternative considered:** Global search — score all (i,j) pairs, greedily pick the best globally.

**Why rejected:** Global greedy accumulates errors. Early wrong placements cascade — a misplaced interior fragment corrupts all neighbors. Corner-seeded growth limits the error boundary to a known set of reliable anchors.

**Result:** Each sub-phase only attaches to already-trusted geometry. The error from any single wrong placement is local and bounded.

---

### Appearance Over Area for Segmentation

**Alternative considered:** Filter SAM masks by area ratio (mask area / image area within a [min, max] band).

**Why rejected:** Fragment areas vary by 10× depending on how the paper was torn. Area thresholds are also DPI-dependent.

**Chosen approach:** LAB ΔE distance from auto-detected background. Works on any paper/background combination. The appearance filter is calibrated once per image from corner samples.

---

### Curvature Feature Strings + Smith-Waterman

**Alternative considered:** Template matching on edge images, or raw pixel correlation.

**Why chosen:** Torn edges have characteristic curvature profiles — the physics of paper tearing creates irregular but reproducible curvature signatures. SW local alignment handles partial overlaps (not all of one edge may be visible) and is robust to small contour-extraction noise. The 80-sample quantization reduces sensitivity to sub-pixel contour variations.

---

### SDT-Based Physical Gating After Geometric Alignment

**Why:** Curvature matching alone can align edges that geometrically cannot fit (e.g., two convex edges with similar curvature). The SDT gate directly measures physical penetration and gap — catching false positive alignments that pass the geometric score.

---

### ICP Coarse → Fine Two-Phase

**Why:** After Procrustes alignment on the SW-matched sub-arc only, the full edge tails may still drift by 15–25 px. A coarse ICP pass with large correspondence radius (25 px) pulls the edges together; the fine pass (6 px) then closes the seam to sub-pixel. Starting fine-only fails when Procrustes drift is large.

---

### Bundle Adjustment Over Greedy Propagation

**Why:** In chain-like placements (A→B→C→D), each Procrustes alignment adds a small rotational error. By 10 pieces the cumulative drift can be several degrees. Bundle adjustment redistributes the error globally — pinning corners means the overall document orientation is preserved while interior pieces find their optimal pose jointly.

---

### Text-Preserving LaMa Scar Mask

**Why:** A naive inpaint mask (dilated seam) would hallucinate text where ink crosses a seam boundary. By masking out pixels below grayscale threshold 140 (ink), LaMa only operates on bare paper regions — it fills paper texture without touching character strokes.

---

### No Database for Job State

**Why:** The system runs on a single GPU workstation with one user at a time. An in-memory dict guarded by a threading lock is simpler, faster, and has no setup cost. The single-GPU semaphore already serializes pipeline execution, so concurrent state corruption is not a concern for production use. A persistent store would be needed only for multi-machine or multi-user deployments.

---

### WebSocket Primary / Polling Fallback

**Why:** The pipeline takes 30–180 seconds depending on fragment count. Polling at 800 ms wastes connections; WebSocket gives sub-second latency at near-zero cost. The polling fallback handles environments where WS is blocked by a proxy.
