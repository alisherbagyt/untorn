# UNTORN — Architecture Reference

> For full technical documentation, see [UNTORN.md](UNTORN.md).  
> For installation and usage, see [README.md](README.md).

This document gives a concise architectural overview: how the system is structured, what each module does, and why key decisions were made.

---

## System Diagram

```
┌──────────────────────────────────────────────────────────┐
│  Next.js 14 Frontend  (port 3000)                        │
│  UploadZone → ProcessingView → ResultsView (6 tabs)      │
│  AssemblyView — interactive canvas, drag/rotate/export   │
└────────────────────┬─────────────────────────────────────┘
                     │  HTTP / WebSocket
┌────────────────────▼─────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                            │
│  /process  /status  /result  /debug  /board  /ws        │
│  In-memory job store  ·  single-GPU semaphore            │
└────────────────────┬─────────────────────────────────────┘
                     │  subprocess  (stdout parsing)
┌────────────────────▼─────────────────────────────────────┐
│  UNTORN Pipeline  (untorn/)                              │
│  preprocess → segment → contours →                       │
│  reconstruct → compose → inpaint                         │
│  SAM 2.1 · DINOv2 ViT-S/14 · Siamese CNN · LaMa         │
└──────────────────────────────────────────────────────────┘
```

### Entry Points

| Mode | Command |
|------|---------|
| CLI | `python run.py <image> [-o output.png]` |
| Web | `POST /process` (multipart file upload) |

The backend runs `run.py` as a child process and reads its stdout line by line. Embedded progress markers (`[PHASE N]`, `Phase N complete`, `DONE`) are parsed to drive the job status and WebSocket APIs.

---

## Module Map

### `untorn/` — Core Algorithm

| Module | Role |
|--------|------|
| `pipeline.py` | Phase orchestrator; calls all phases; writes `pipeline_meta.json` |
| `config.py` | ~100 tunable constants, 412 lines |
| `preprocess.py` | Phase 0: downscale to `WORKING_MAX_DIM`, store `scale_factor` |
| `segmentation.py` | Phase 1: SAM 2.1 AMG + LAB appearance filter + morphological cleanup |
| `contours.py` | Phase 2: sub-pixel boundary, support points, edge descriptors, SDT |
| `boundary.py` | Sub-pixel ridge-snapping: snaps boundary pixels to local gradient maximum |
| `text_lines.py` | Per-fragment text-baseline detection via rotation-sweep projection |
| `appearance.py` | DINOv2 ViT-S/14 dense features; seam patch cosine similarity |
| `fragment_profile.py` | Fragment role (corner/boundary/interior), ink density, anchor strength |
| `fragment_io.py` | Edge + SDT preparation; paper-colour fingerprints |
| `grid_filter.py` | LBP histogram fast-filter; 2-point RANSAC; cuts ~80% of candidate pairs |
| `matching.py` | 10-gate matcher cascade: SW → Procrustes → ICP → Siamese → DINOv2 → SDT → fit cost |
| `edge_matcher.py` | Siamese CNN inference adapter; strip extraction; graceful degradation |
| `edge_matcher_model.py` | Two-tower shared-weight CNN, ~520K params, val_AUC=0.924 |
| `edge_rank.py` | Per-edge partner ranking; mutual-best seed candidates |
| `seam_solver.py` | Post-MST Nelder-Mead seam refinement on (Δθ, Δdx, Δdy) |
| `assembly.py` | MST growth, bundle adjustment (LM), orphan rescue, cluster reconciliation |
| `composition.py` | Phase 4: 2× supersampled warp, SDT arbitration, LAB harmonisation |
| `gap_fill.py` | Phase 5 front-end: hole classification (edge/small/medium/large) |
| `inpainting.py` | LaMa JIT → simple-lama → saicinpainting → classical fallback |
| `io_utils.py` | `load_image()` (8/16-bit TIFF/JPG/PNG) and `save_image()` |

### `backend/` — Web Service

| Module | Role |
|--------|------|
| `main.py` | FastAPI app; all 11 endpoints; board export with LaMa |
| `pipeline_wrapper.py` | Subprocess launcher; phase-marker parser; job updater |
| `services/job_service.py` | In-memory job store + `threading.Semaphore(1)` |
| `models/schemas.py` | Pydantic request/response models |

### `frontend/` — UI

| Module | Role |
|--------|------|
| `app/page.tsx` | SPA state machine: upload → processing → results |
| `components/ProcessingView.tsx` | Phase timeline, live logs, queue position |
| `components/ResultsView.tsx` | 6-tab results with image preloading |
| `components/views/AssemblyView.tsx` | 1 360-line interactive canvas: drag/rotate/zoom/undo/export |
| `components/views/OverviewView.tsx` | Final image, stats, timing, download |
| `components/views/SegmentationView.tsx` | SAM overlay, fragment carousel |
| `components/views/ReconstructionView.tsx` | Step-through merge log, gap-score chart |
| `components/views/CompositionView.tsx` | Layer toggle (raw/gap/inpainted/cleaned) |
| `lib/api.ts` | Typed backend client + all TypeScript interfaces |
| `lib/utils.ts` | `cn()`, `phaseLabel()` (Russian), `getApiBase()` |

---

## Pipeline Phases

| Phase | Module | Progress | What happens |
|-------|--------|----------|-------------|
| 0 | `preprocess.py` | 5–10% | Downscale to ≤1500 px; store scale factor |
| 1 | `segmentation.py` | 12–44% | SAM 2.1 mask generation; LAB appearance filter; morphological cleanup |
| 2 | `contours.py` | 45–54% | Sub-pixel boundary; text lines; support points; curvature strings; SDT; DINOv2; fragment profiles |
| 3 | `assembly.py` + `matching.py` | 55–81% | Candidate enumeration; 10-gate matching; MST growth; seam refinement; bundle adjustment; orphan rescue |
| 4 | `composition.py` | 82–89% | Upscale transforms; 2× supersampled warp; SDT arbitration; LAB harmonisation; Voronoi seam fill |
| 5 | `gap_fill.py` + `inpainting.py` | 90–100% | Hole classification; scar mask; LaMa inpainting |

---

## Matching Cascade

For each candidate fragment pair, the matcher applies 10 sequential gates. A pair is accepted only if it passes all gates.

```
Gate 0  Edge quality      Length ≥ 30 px; curvature std > threshold
Gate 1  Facing            Outward normals dot product ≥ 0.1
Gate 2  SW alignment      Smith-Waterman on 80-sample curvature strings; score ≥ 5.0
Gate 3  Procrustes        SVD rigid fit of aligned samples; rms ≤ 6.0 px; |angle| ≤ 30°
Gate 4  ICP               Coarse (25 px) → fine (6 px); drift cap 15°
Gate 5  Siamese CNN       (32, 256, 3) strip pair scored by edge_matcher.pt; prob ≥ 0.55
Gate 6  DINOv2 cosine     8-patch seam cosine via ViT-S/14 features; mean ≥ 0.55
Gate 7  Text lines        Baseline continuity when ≥2 lines cross seam; ≥ 0.30
Gate 8  Paper colour      LAB ΔE < 12 contributes soft score
Gate 9  SDT physical      Penetration ≤ 20% of contour / ≤ 10 px; gap ≤ 8 px
Gate 10 Fit cost          1.5×overlap + 2.0×gap + 80×(1-coverage) ≤ 400
```

Confidence score:
```
0.30 × geom + 0.20 × DINOv2 + 0.20 × strip_NCC + 0.20 × text + 0.10 × paper
```

---

## Assembly Strategy

1. **LBP grid filter** cuts ~80% of candidate pairs before any gate runs.
2. Surviving pairs are scored by the 10-gate cascade and cached.
3. **Mutual-best seeds** (pairs where both fragments rank each other in their top-3) seed the MST.
4. **Priority-queue MST growth** expands from placed fragments; conflict resolution via max-weight bipartite matching.
5. **Seam-contact Nelder-Mead** refines each placed pair on (Δθ, Δdx, Δdy).
6. **Bundle adjustment** (Levenberg-Marquardt) jointly optimises all poses with sparse SW + dense edge constraints.
7. **Orphan rescue** (standard then aggressive) handles fragments not reached by MST.

---

## Transform Format

All fragment poses are `3×3 SE(2) homogeneous affine matrices`:

```python
M = [[cos θ, -sin θ, tx],
     [sin θ,  cos θ, ty],
     [0,      0,     1 ]]
```

Translation `(tx, ty)` is in working-resolution pixels. Phase 4 multiplies both by `scale_factor` before warping at full resolution. Rotation is scale-invariant.

---

## Backend API Summary

| Method | Path | Description |
|--------|------|-------------|
| POST | `/process` | Upload image, create job, start pipeline |
| GET | `/status/{job_id}` | Job status: queued/processing/done/error, progress 0–100 |
| GET | `/result/{job_id}` | Download final PNG |
| WS | `/ws/{job_id}` | Real-time push every ~400 ms |
| GET | `/debug/{job_id}` | All pipeline metadata and image URL paths |
| GET | `/debug/image/{job_id}/{path}` | Serve any debug image |
| GET | `/board/data/{job_id}` | Canvas dims + per-fragment placement |
| GET | `/board/fragment/{job_id}/{id}` | Transparent PNG of one fragment |
| POST | `/board/export/{job_id}` | Composite user placement → PNG download |
| GET | `/health` | Health check |
| GET | `/jobs` | List all jobs |

---

## Debug Artifacts

Every run writes to `data/debug/<stem>/`:

```
pipeline_meta.json              Timings, fragment count, status, model flags
segmentation/                   SAM overlays, per-fragment masks, fragments_meta.json
contours/                       SDF images, support point images, contours_meta.json
reconstruction/                 Step PNGs, merge_log.json, final_translations.json
composition/                    Layer PNGs, composition_meta.json
inpainting/                     Before/mask/after PNGs, inpainting_meta.json
```

---

## Key Design Decisions

**Appearance-based segmentation filter** — LAB ΔE from auto-detected corner background, not area ratios. Works on any paper/background combination regardless of DPI.

**Smith-Waterman over global curvature alignment** — handles partial edge visibility and contour noise; returns sample correspondences directly for Procrustes.

**SDT physical gates** — hard rejection for geometric impossibilities (penetration/gap) regardless of feature scores. Prevents chimeric assemblies.

**MST over global greedy** — limits error propagation; only expands from already-trusted geometry; conflict resolution is local.

**Bundle adjustment** — redistributes cumulative angular drift from chain placements; dense edge constraint eliminates residual seam gaps.

**LaMa with text-protection mask** — scar mask excludes ink pixels (grayscale < 140); LaMa fills only bare paper regions; text is never hallucinated across seams.

**In-memory job store** — single GPU workstation, single user; no database setup needed; single-GPU semaphore already serialises all execution.
