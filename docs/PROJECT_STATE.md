# UNTORN — Project State Overhaul

**Snapshot date:** 2026-04-28
**Release:** v3.3
**Total source:** ~11,900 lines across 18 pipeline modules + 21 tools
**Status:** All phases complete (1, 2, 3, 4, 5). Production pipeline runs end-to-end on real scans and synthetic puzzles.

---

## Table of Contents

1. [What UNTORN does](#1-what-untorn-does)
2. [System layout](#2-system-layout)
3. [The pipeline, end-to-end](#3-the-pipeline-end-to-end)
4. [Pipeline modules — reference](#4-pipeline-modules--reference)
5. [Matcher cascade — the heart of the system](#5-matcher-cascade--the-heart-of-the-system)
6. [Data pipeline & training](#6-data-pipeline--training)
7. [Configuration index](#7-configuration-index)
8. [Tooling & tests](#8-tooling--tests)
9. [Performance characteristics](#9-performance-characteristics)
10. [Maturity & known limits](#10-maturity--known-limits)
11. [Roadmap forward](#11-roadmap-forward)

---

## 1. What UNTORN does

UNTORN reconstructs torn paper documents from a single photograph of the scattered fragments. Input: a JPEG/TIFF/PNG with arbitrarily-rotated paper pieces lying on a contrasting (typically dark) background. Output: a clean reconstruction of the original document, with seam scars and missing-fragment holes inpainted by a frozen LaMa model.

Three things make the problem non-trivial:

1. **Fragments don't have unique IDs.** The system has to discover which torn edges belong together by analysing curvature, color, texture, and document-content continuity.
2. **Tear edges are noisy at sub-pixel scale.** Mask-boundary jitter from segmentation dominates curvature on gentle tears, so we maintain sub-pixel boundaries throughout.
3. **A single bad pairing breaks the whole assembly.** With N fragments there are O(N²) candidate pairings; one wrong attach cascades into wrong global pose. The matcher cascade exists to make false positives statistically rare.

---

## 2. System layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Next.js 14 Frontend  (localhost:3000)                                │
│  Upload → Processing → Results (6 tabs) + interactive Assembly board │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  HTTP / WebSocket
┌──────────────────────────▼───────────────────────────────────────────┐
│  FastAPI Backend  (localhost:8000)                                    │
│  /process /status /result /debug /board /ws                          │
│  In-memory job store + single-GPU semaphore                          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  subprocess (stdout parsing)
┌──────────────────────────▼───────────────────────────────────────────┐
│  UNTORN Python Pipeline  (untorn/)                                    │
│                                                                       │
│  pipeline.run(input)                                                  │
│   └─ Phase 0  preprocess        (downscale to <2400 px long edge)     │
│   └─ Phase 1  segment           (SAM 2.1 AMG)                         │
│   └─ Phase 2  contours          (subpixel boundary, text lines, SDF)  │
│   └─ Phase 3  reconstruct       (5-gate matcher + MST growth + BA)    │
│   └─ Phase 4  composition       (supersampled warp + LAB harmonise)   │
│   └─ Phase 5  gap_fill          (LaMa inpainting on classified holes) │
│                                                                       │
│  Dependencies: SAM 2.1, LaMa, OpenCV, scipy, networkx, PyTorch        │
└──────────────────────────────────────────────────────────────────────┘
```

### Directory map

```
untorn/                         pipeline modules (18 files, ~7000 LOC)
  pipeline.py                   orchestrates Phases 0-5
  preprocess.py                 downscale + transform back-projection
  segmentation.py               SAM 2.1 AMG + filtering
  contours.py                   support points, edge segments, SDF, color profile
  boundary.py                   sub-pixel ridge-snapping
  text_lines.py                 per-fragment baseline detection
  appearance.py                 DINOv2 dense features + seam patch cosine
  matching.py                   the 5-gate matcher cascade (1177 LOC)
  grid_filter.py                Phase 3 LBP block fast-filter
  edge_matcher.py               Phase 4 Siamese inference adapter
  edge_matcher_model.py         Siamese CNN architecture
  assembly.py                   MST growth + bundle adjust + orphan rescue
  composition.py                supersampled warp + LAB harmonisation
  gap_fill.py                   hole classification + repair mask
  inpainting.py                 LaMa wrapper
  config.py                     all knobs (412 lines, ~100 knobs)
  io_utils.py                   image read/write helpers

tools/                          CLI scripts (21 files)
  # data pipeline
  extract_publaynet.py          parquet -> PNG
  tear_simulator.py             Voronoi + jagged perturbation -> fragments
  composite_generator.py        scattered composite (X)
  build_dataset_index.py        train/val splits
  build_edge_dataset.py         (32, 256, 3) edge strips for training
  # training & eval
  train_edge_matcher.py         Siamese CNN training (cosine LR + AMP)
  eval_edge_matcher.py          AUC vs baselines
  benchmark_synthetic.py        Phase 5 A/B/C/D ablation
  run_benchmark.py              real-data benchmark harness
  debug_eval.py                 misc diagnostics
  # unit tests
  test_assembly.py
  test_matching_stage4.py
  test_edge_matcher_integration.py     (NEW in Phase 4)
  test_appearance.py
  test_appearance_scorer.py
  test_boundary.py
  test_composition.py
  test_evaluator.py
  test_gap_fill.py
  test_packer.py
  test_text_lines.py

backend/                        FastAPI app
frontend/                       Next.js 14 UI
sam2/                           pinned SAM 2.1 repo
lama/                           pinned LaMa weights + code
models/                         trained checkpoints (edge_matcher.pt)
data/                           inputs, outputs, debug, datasets, eval results

ARCHITECTURE.md                 high-level design reference
docs/PHASE_4_5_CHANGES.md       this overhaul's change report
docs/PROJECT_STATE.md           this document
```

---

## 3. The pipeline, end-to-end

Every input image follows the exact same six-phase path. Each phase has a clear contract and writes debug artefacts to `data/debug/<stem>/<phase>/` so a failure can be triaged without re-running.

### Phase 0 — Preprocess (`preprocess.py`)

- Loads RGB image (handles 8/16-bit TIFF, JPG, PNG).
- Computes a working-resolution copy (long edge ≤ 2400 px) so SAM and DINOv2 fit in 4 GB VRAM.
- Stores the scale factor for later: every transform is computed at working resolution and **upscaled** for the final composition pass.

### Phase 1 — Segment (`segmentation.py`)

- SAM 2.1 Automatic Mask Generator at `points_per_side=32`.
- Filters by:
  - LAB ΔE distance from background (auto-detected from corner samples).
  - Area within `[MIN_FRAGMENT_AREA_FRAC, MAX_FRAGMENT_AREA_FRAC]` of the image.
  - Mask convexity / hole count (rejects mask-leaks into the background).
- Output: `fragments[]` with `id, mask, bbox, area, contour, centroid`.

### Phase 2 — Contour analysis (`contours.py` + helpers)

- `attach_subpixel_contours_all` — for every fragment, snap mask-boundary points to the local image-gradient ridge along the inward normal (`boundary.py`). This kills the ±1 px integer-mask jitter that dominates curvature on gentle tears.
- `attach_text_lines_all` — per-fragment text-line baseline detection by rotation sweep + projection profile peak-finding (`text_lines.py`).
- `analyze_fragments` — Douglas-Peucker support points, per-edge geometric metadata, signed distance transform, edge color profile.

### Phase 3 — Reconstruct (`assembly.py` + `matching.py` + `grid_filter.py`)

This is the heart. See [§5](#5-matcher-cascade--the-heart-of-the-system) for the matcher detail. At a high level:

```
for every fragment pair (i, j):
    if grid_filter rejects it:           skip                         <-- Phase 3
    if paper-color delta too high:       skip
    for every torn-edge pairing:
        run the 5-gate cascade           (geometry -> Siamese -> appearance)
        if all gates pass: record (R, t, confidence)

seed:        highest-confidence match
grow:        MST attachment by descending confidence with overlap gate
adjust:      seed-pinned bundle adjustment over the placed pose graph
rescue:      orphans get one more attempt at relaxed confidence
```

### Phase 4 — Composition (`composition.py`)

- 2× supersampled canvas; each fragment warped by `cv2.warpAffine` with `INTER_LINEAR`.
- Cosine-feathered alpha blending at fragment boundaries (no hard pixel assignment).
- Per-fragment LAB color harmonisation: every fragment's paper-pixel L mean is matched to the cluster median; ink-pixel shift is attenuated (no text colour drift).

### Phase 5 — Gap fill (`gap_fill.py` + `inpainting.py`)

- Classify each connected hole in the composed canvas as small / medium / large by area fraction.
- Build a unified scar+hole mask. Small holes ride along with the seam scar; medium holes get expanded context so LaMa sees enough surrounding paper. Large holes are flagged in `pipeline_meta.json` as `missing_fragment=true`.
- LaMa fills the mask. If the checkpoint isn't present, Phase 5 passes the canvas through unchanged.

---

## 4. Pipeline modules — reference

### `pipeline.py` (237 LOC)

Single entry point: `run(input_path: str, output_path: str = None)`. Orchestrates all six phases, manages debug-directory creation, and writes `pipeline_meta.json` with timings and status flags.

The Phase 4 model lifecycle hooks live here:

```python
if cfg.EDGE_MATCHER_ENABLED:
    edge_matcher.load()       # before reconstruct()
try:
    transforms = reconstruct(fragments, work_rgb, debug_dir)
finally:
    edge_matcher.unload()     # frees VRAM before LaMa
```

### `matching.py` (1177 LOC) — see [§5](#5-matcher-cascade--the-heart-of-the-system)

The largest module, intentionally. Every gate of the cascade has its dedicated function with diagnostics so the trace tool (`UNTORN_MATCH_TRACE=1`) can explain every rejection.

### `grid_filter.py` (597 LOC, NEW Phase 3)

Self-contained binary fast-filter. Public surface:

```python
build_index(fragments, image_rgb) -> GridFilterIndex
screen_candidates(index, top_k=8) -> dict[int, list[(int, float)]]
filter_pairs(fragments, image_rgb, candidates, top_k=8) -> list[(int, int)]
```

Internals: edge-band sampling (`cv2.remap` along inward normal), uniform LBP (skimage with manual fallback), 16×16 block tiling, L1-normalised histograms, top-K argpartition cosine ranking, 2-point rigid SE(2) RANSAC with inlier counting.

### `edge_matcher.py` (368 LOC, NEW Phase 4)

Inference adapter for the trained Siamese CNN. Public surface:

```python
load(checkpoint_path=None, device=None) -> EdgeMatcher | None
unload() -> None
is_loaded() -> bool
extract_strip_pair(image_rgb, edge_a, edge_b, orientation) -> (np.ndarray, np.ndarray) | None
score_edge_pair(image_rgb, edge_a, edge_b, orientation) -> dict | None
```

Singleton pattern: one model loaded for the duration of `reconstruct()`, queried by `_match_edge_pair()` via `is_loaded()`. Every failure path is silent — missing torch / missing checkpoint / CUDA OOM all cause the gate to be skipped without breaking the pipeline.

### `edge_matcher_model.py` (224 LOC)

Pure model definition. Two-tower shared-weight CNN; cosine match head with learnable temperature; 3-DOF pose head trained on positives only. Total parameters: ~520 K.

### `assembly.py` (868 LOC)

`reconstruct(fragments, image_rgb, debug_dir) -> dict[int, 3x3 affine]`. Implements:

- `_enumerate_pair_candidates` — paper-color + edge-length prefilter, intersected with `grid_filter.screen_candidates` when enabled.
- `_MatchCache` — caches `match_pair` results so the same pair is never re-scored even when iterated from both sides.
- MST growth with conflict resolution (networkx max-weight bipartite matching when multiple free fragments compete for the same anchor).
- Periodic global-rotation correction by horizontalising placed text baselines.
- Seed-pinned LM bundle adjustment over the placed pose graph.
- Orphan rescue at relaxed confidence.

---

## 5. Matcher cascade — the heart of the system

A single torn-edge pair `(edge_a, edge_b)` from two distinct fragments runs through ten checks before being accepted:

```
0.  Length & curvature filter         min_torn_edge_px, curv_min_std         (matching.py)
1.  Outward-normal facing gate        cosine(normal, centroid_offset) > 0.1
2.  SW curvature alignment            score >= cfg.SW_MIN_SCORE
3.  Multi-seed Procrustes             rms <= cfg.MATCH_MAX_RMS, |angle| <= 30 deg
4.  Two-phase ICP refinement          coarse 25 px tol -> fine 6 px tol
5.  Post-ICP rotation bound           |angle| <= cfg.RECON_MAX_ROTATION_DEG
─── NEW: Phase 4 gate ─────────────────────────────────────────────────────
6.  Siamese edge matcher              match_prob >= cfg.EDGE_MATCHER_MIN_SCORE (0.985)
─── existing gates resume ─────────────────────────────────────────────────
7.  DINOv2 seam patch cosine          cosine >= cfg.MATCH_APPEARANCE_COS_MIN (0.55)
8.  Text-line continuity              when applicable (>=2 expected lines)
9.  Paper-color similarity            soft score, blended into confidence
10. SDT physical gate                 no fragment-body overlap, seam gap <= 8 px
11. Full-edge fit evaluator           overlap + gap + coverage cost <= MAX_ATTACH_COST
```

If any check rejects, the pair is dropped and the matcher moves on. The surviving (R, t) is emitted with a weighted-sum confidence score in [0, 1]:

```python
conf = (CONF_W_GEOMETRY    * geom_conf       # 0.30
      + CONF_W_APPEARANCE  * dinov2_score    # 0.20
      + CONF_W_STRIP_NCC   * strip_ncc       # 0.20
      + CONF_W_TEXT_LINE   * text_score      # 0.20 (-> 0 when no text)
      + CONF_W_PAPER_COLOR * paper_score)    # 0.10
```

The Siamese gate (step 6) does not contribute to `conf` — it's a HARD GATE. Its purpose is to catch "geometry plausible, texture wrong" pairs that survive 0–5 but visually don't belong. Stashed on the result dict as `edge_matcher_prob` and `edge_matcher_cos` for diagnostics.

### Why so many gates

Each gate addresses a **different failure mode** observed during development:

| Gate | Catches |
|---|---|
| 0–1 | Trivial junk: too-short edges, edges facing away from each other. |
| 2 | Generic mismatch: curvature signatures don't share an aligned subarc. |
| 3 | Procrustes locking onto the wrong end of an edge that has multiple plausible alignments. |
| 4 | Sub-pixel jitter that pure Procrustes leaves: ICP snaps text strokes to their counterpart across the seam. |
| 5 | ICP drift into an implausible-rotation basin. |
| **6** | **Right curvature, wrong edge: same shape on a different document or different torn edge of the same document.** |
| 7 | Smooth ink/paper texture difference (DINOv2 sees this where curvature can't). |
| 8 | Text baseline disagreement across a proposed seam. |
| 9 | Different paper colour — soft, not a hard reject. |
| 10 | Physically impossible overlap: fragment B inside fragment A's body. |
| 11 | Gappy seam: edges align at the matched sub-arc but drift apart at the ends. |

Each gate's threshold is documented inline in `untorn/config.py`.

---

## 6. Data pipeline & training

### Source data

PubLayNet (~24 GB) parquet files at `data/dataset/PubLayNet/parquet/`. Three parquet files (`train-00000`, `train-00001`, `train-00002`) plus two validation parquets. Each ~470 MB, ~1600 documents.

### Synthetic puzzle generation

```
extract_publaynet.py         parquet rows -> PNG
tear_simulator.py            Voronoi tessellation + jagged edge perturbation -> fragments
composite_generator.py       random rotation + non-overlapping placement -> composite (X)
build_dataset_index.py       80/20 train/val split, written to data/dataset/index.json
```

Output per source document:

```
data/dataset/synthetic/<doc_id>/
  original.png                       <-- Y (ground truth)
  fragments/fragment_00.png ... NN   <-- per-fragment RGBA crops
  ground_truth.json                  <-- bbox_in_original, neighbors, ...
  composite.png                      <-- X (scrambled, what the scanner gives you)
  composite_transforms.json          <-- M_place, M_to_original per fragment
```

### Edge-strip dataset

`build_edge_dataset.py` walks the synthetic dataset, extracts (32, 256, 3) RGB strips along every shared boundary, and produces:

```
data/dataset/edge_strips/
  train.h5    (~3,910 docs, ~50K positive pairs)
  val.h5      (~940 docs, ~13K positive pairs)
```

Negatives are sampled at training time (cross-document strips with `partner_strip_index` mismatch).

### Training

`train_edge_matcher.py` — AdamW + cosine LR + AMP. 1:3 positive:negative batch ratio. Pose augmentation applies a random rigid transform to strip B for positives so the pose head learns alignment. Brightness/contrast jitter + Gaussian noise as scanner-domain augmentation.

Trained checkpoint: `models/edge_matcher.pt` — best epoch 4, **val_AUC = 0.9340**, ~520 K parameters, 2 MB on disk.

### Eval

`eval_edge_matcher.py` benchmarks the Siamese model against four deterministic baselines (pixel NCC, row correlation, LAB mean L1, SW-curvature proxy) on 8 K val pairs. Result: Siamese **AUC=0.924**, baselines AUC=0.52–0.65. The model dominates every baseline by a wide margin.

---

## 7. Configuration index

`untorn/config.py` is 412 lines and houses every tunable knob. Sections:

| Section | Knobs |
|---|---|
| Project paths | `PROJECT_ROOT`, `DATA_DIR`, ... |
| SAM 2.1 | `SAM2_*` (8 knobs) |
| Background detection | `BG_*` (2 knobs) |
| Fragment limits | `MIN/MAX_FRAGMENT_AREA_FRAC`, `MAX_FRAGMENTS` |
| Text-line detection | `TEXT_LINE_*` (7 knobs) |
| Sub-pixel boundary | `BOUNDARY_*` (4 knobs) |
| DINOv2 | `DINOV2_*` (7 knobs) |
| Matching gates | `MATCH_*` (7 knobs), `CONF_W_*` (5 weights) |
| Assembly | `ASSEMBLY_*` (8 knobs) |
| **Grid filter (Phase 3)** | `EDGE_MATCHER_*` (5 knobs) |
| Composition | `COMP_*` (4 knobs) |
| Gap fill | `GAP_*` (4 knobs) |
| Curvature | `CURV_*`, `POLY_EPSILON_FACTOR` |
| Smith-Waterman | `SW_*` (8 knobs) |
| SDT physical gate | `SDT_*` (3 knobs) |
| ICP | `ICP_*` (4 knobs) |
| Full-edge fit | `FIT_W_*` (3 knobs), `COVERAGE_TOLERANCE_PX`, `MAX_ATTACH_COST` |
| Bundle adjustment | `BA_*` (4 knobs) |
| Orphan rescue | `ORPHAN_MAX_ATTACH_COST` |

Total: ~100 knobs, every one inline-documented with the *why*, not just the *what*.

### Disabling features

Every Phase-3-and-later feature is gated by an `_ENABLED` flag:

```python
DINOV2_ENABLED        = True
BOUNDARY_REFINE_ENABLED = True
GRID_FILTER_ENABLED   = True
EDGE_MATCHER_ENABLED  = True
COMP_LAB_HARMONISE_ENABLED = True
```

Setting any to `False` reverts that section to its bypass behavior. The legacy 4-gate cascade is recoverable in 30 seconds.

---

## 8. Tooling & tests

### Pipeline tests (run as scripts)

| Tool | Tests |
|---|---|
| `tools/test_assembly.py` | Phase 3 end-to-end on 3 synthesized fragments, exercises grid filter + matcher. |
| `tools/test_matching_stage4.py` | Single edge-pair match rejection on a known-bad pair. |
| `tools/test_edge_matcher_integration.py` | Phase 4 smoke (NEW): strip extraction, missing-ckpt graceful degradation, real-ckpt scoring, full integration. |
| `tools/test_appearance.py` | DINOv2 feature extraction. |
| `tools/test_appearance_scorer.py` | Seam patch cosine. |
| `tools/test_boundary.py` | Sub-pixel ridge snapping. |
| `tools/test_composition.py` | Phase 4 supersampled warp + LAB harmonise. |
| `tools/test_evaluator.py` | Full-edge fit cost computation. |
| `tools/test_gap_fill.py` | Hole classification + repair mask. |
| `tools/test_packer.py` | Final canvas crop bbox. |
| `tools/test_text_lines.py` | Per-fragment baseline detection. |

### Module self-tests

```
python -m untorn.grid_filter      # Phase 3 self-test (synthetic 2-fragment scene)
python -m untorn.edge_matcher     # Phase 4 self-test (strip extraction + lifecycle)
python -m untorn.edge_matcher_model  # model architecture + parameter count
```

### Benchmarks

```
python tools/benchmark_synthetic.py      # Phase 5 A/B/C/D ablation (NEW)
python tools/run_benchmark.py            # real-data harness (existing)
python tools/eval_edge_matcher.py        # Siamese vs baselines on val.h5
```

### Frontend / backend dev

```
start_backend.bat                # uvicorn backend.main:app
start_frontend.bat               # next dev
docker-compose up                # full stack with GPU passthrough
docker-compose -f docker-compose.cpu.yml up  # CPU-only fallback
```

---

## 9. Performance characteristics

Measurements at working resolution (long edge ~1500 px) on a single RTX 3070 Ti (8 GB VRAM):

| Phase | Time | Notes |
|---|---|---|
| 0. Preprocess | 0.2 s | Pure NumPy / OpenCV. |
| 1. SAM segmentation | **5–9 s** | Dominant cost on small fragment counts. |
| 2. Contour analysis | 1–3 s | Sub-pixel boundary + text lines + DINOv2 cache. |
| 3. Reconstruction | 5–25 s | Scales with O(N²) candidate pairs but grid filter cuts most. |
| 4. Composition | 1 s | Supersampled warp + LAB harmonise. |
| 5. LaMa inpainting | 2–4 s | First load is ~10 s; warm calls are fast. |
| **Total (4 fragments)** | **~15 s** | Real-scanner typical. |
| **Total (8 fragments)** | **~35 s** | Synthetic-puzzle typical. |

VRAM peak: ~3.5 GB during SAM, ~1.5 GB during DINOv2, ~520 K params for Siamese (negligible). All three load and unload independently so a 4 GB GPU has headroom.

The Siamese gate adds ~1 ms per pair. Grid filter adds ~0.04 s per fragment. Both are well below the per-pair matcher cost (~0.5–2 s including DINOv2 seam sampling).

---

## 10. Maturity & known limits

### Production-ready components

- **Phases 0, 1, 4, 5:** Stable, well-tested, identical interface to legacy versions.
- **Phase 2:** Stable. Sub-pixel boundary refinement substantially improves curvature quality on gentle tears (tested on `data/input/t1.jpg` through `t4.tif`).
- **Phase 3 grid filter:** Stable. Adds 50–80 ms per fragment; correctly identifies high-likelihood pair candidates on real scanner inputs.
- **Phase 3 Siamese gate:** Stable. Falls back transparently when checkpoint is missing. Known calibration artifact: model temperature is high (~10), so probabilities saturate near 0/1; the threshold of 0.985 reflects this.

### Components with limitations

- **Phase 3 reconstruction on random-rotation puzzles.** `RECON_MAX_ROTATION_DEG = 30` is correct for real scanner data (where fragments don't rotate much) but rejects most pairs when fragments are randomly placed at 0–360°. The synthetic benchmark exposes this clearly.
- **Edge matcher AUC plateau at 0.92.** Good but not state-of-the-art. The training pipeline uses cross-document negatives only; same-document hard negatives would close some of the gap to AUC ~0.97. Lower training temperature would also unsaturate the probability distribution.
- **Pose head unused at inference.** The trained model emits (Δθ, Δdx, Δdy) per pair but we don't blend it with the geometric estimate. Reserved for a future ICP warm-start feature.
- **Real-data benchmarking is qualitative.** No scriptable accuracy metric on real scans; we eyeball the output. A hand-labelled benchmark (10–20 documents with annotated corner positions) is the cleanest path forward.

### Failure modes the cascade does NOT catch

- **Two true matches whose curvatures are identical.** Long, smoothly-tapered tears that look the same from either end can produce two equally-good Procrustes seeds. The multi-seed approach in `MATCH_PROCRUSTES_SEEDS=3` mitigates but doesn't eliminate this.
- **Documents with very high text-line density and short fragment perimeter.** Text-line continuity gate is robust but can miss when a tear runs through a single line that breaks both sides equally.
- **Severely curled or photographed-not-scanned input.** SAM finds the fragments, but downstream gates assume planar paper. Out of scope.

---

## 11. Roadmap forward

### Tier 1 — clear next steps (low risk, measurable wins)

1. **Hard-negative training run** for the edge matcher. Target AUC 0.95+ by mining same-document negatives. Estimated effort: 1 day, ~6 h GPU.
2. **Lower training temperature** from 10 → 2 to unsaturate probabilities. Threshold becomes tunable in [0.5, 0.9], easier to reason about. Same training pipeline.
3. **Pose-head ICP warm start.** Use Siamese (Δθ, Δdx, Δdy) as the initial guess before Procrustes. Should accelerate convergence and might catch a few cases ICP currently misses.

### Tier 2 — useful but bigger

4. **Real-data benchmark harness.** Hand-label 20 documents in `data/input/`, write a script that compares predicted reconstruction to labelled ground truth, integrate into CI. The first scriptable real-data metric.
5. **A model-served version.** Currently the pipeline runs in-process per request. A dedicated inference server (Triton or vLLM-style) holding SAM + DINOv2 + LaMa + Siamese resident would cut per-request latency by ~5 s.
6. **Active learning loop.** When the cascade rejects a pair late (e.g., gate 10 SDT physical), feed those cases back into a training set as hard negatives.

### Tier 3 — research-y

7. **End-to-end differentiable pose head.** Train a head that emits the FINAL (R, t) directly, with the SDT physical gate as a soft loss. Could replace gates 3–5.
8. **Multi-document mode.** Currently one image = one document. Support for sorting fragments belonging to different source documents would unlock real archival use cases.
9. **Active human-in-the-loop.** When the matcher's top-1 confidence is below a threshold, surface the ambiguity to the user via the assembly board UI (already exists for manual override, just needs the pipeline to emit "uncertain" signals).

---

## Appendix: trace tools

```
$ UNTORN_MATCH_TRACE=1 python -m untorn.pipeline data/input/t1.jpg
[trace] frag0.e2 <> frag1.e3: reject sw_weak
[trace] frag0.e2 <> frag1.e5: reject rms_high  rms=8.34
[trace] frag1.e3 <> frag2.e4: reject edge_matcher_low  prob=0.412 min=0.985
[trace] frag1.e7 <> frag3.e1: reject dinov2_low  cos=0.31
```

Every rejection is logged with a reason and the relevant numerical evidence.

```
$ python -c "from untorn import edge_matcher; print(edge_matcher._LOAD_REASON)"
checkpoint missing: C:\dev\untorn\models\__nonexistent__.pt
```

Sticky load-failure reasons are inspectable for debugging.

---

*End of project state document.*
