# UNTORN

Reconstruct torn paper documents from a single photograph — fully automatic, no prior knowledge of content, layout, or fragment count.

Given a scanned image of scattered paper fragments on a contrasting background, UNTORN detects every fragment, analyses each torn edge, algorithmically assembles the pieces, and outputs a clean high-resolution reconstruction with seams erased by neural inpainting.

---

## How It Works

The pipeline runs six sequential phases:

| Phase | What happens |
|-------|-------------|
| **0 — Preprocessing** | Downscales the input to fit GPU VRAM (~1500 px long edge), preserving the original for final rendering |
| **1 — Segmentation** | SAM 2.1 Automatic Mask Generator isolates every fragment; a LAB colour filter separates paper from background automatically |
| **2 — Contour Analysis** | Sub-pixel boundary refinement, curvature feature extraction, text-line detection, signed distance transforms |
| **3 — Reconstruction** | 5-gate matcher (curvature → Procrustes → ICP → Siamese CNN → DINOv2) builds a minimum spanning tree of matches; bundle adjustment removes cumulative drift |
| **4 — Composition** | Warps all fragments onto a full-resolution canvas with 2× supersampling and LAB colour harmonisation |
| **5 — Gap Fill** | LaMa neural inpainting erases seam scars and fills small holes; text pixels are protected |

A web UI (FastAPI + Next.js) wraps the pipeline and provides real-time progress monitoring plus an interactive assembly board for manual refinement and export.

---

## Prerequisites

### Models

Download these before running:

```
sam2/sam2.1_hiera_small.pt     # SAM 2.1 weights (~190 MB)
lama/big-lama/big-lama.pt      # LaMa TorchScript checkpoint (~200 MB)
models/edge_matcher.pt         # Siamese edge matcher (included in repo, ~2 MB)
```

Download LaMa:
```bash
python scripts/download_lama_jit.py
```

SAM 2.1 weights must be obtained from the [SAM 2 GitHub releases](https://github.com/facebookresearch/segment-anything-2).

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB |
| RAM | 8 GB | 16 GB |
| CUDA | 11.8 | 12.1 |
| Python | 3.10 | 3.11 |

CPU-only mode is supported but significantly slower (SAM alone takes ~60 s per image on CPU vs. ~7 s on GPU).

---

## Installation

### Local (Conda)

```bash
# Create and activate environment
conda create -n untorn python=3.11
conda activate untorn

# Install PyTorch with CUDA (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install SAM 2
pip install -e sam2/

# Install pipeline and web dependencies
pip install -r requirements_web.txt
```

### Docker (GPU)

```bash
# Pull and start (requires NVIDIA Container Toolkit)
docker compose up -d

# Or build from source
docker compose up --build -d
```

### Docker (CPU)

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

The frontend is at `http://localhost:3000`, backend API at `http://localhost:8000`.

---

## Running

### Command Line

```bash
conda activate untorn
python run.py data/input/photo.jpg
python run.py data/input/photo.tif -o results/reconstruction.png
```

Output is written to `data/output/<image_stem>.png`. Full debug artifacts (intermediate images, JSON metadata) are written to `data/debug/<image_stem>/`.

### Web UI

```bash
# Start backend (terminal 1)
start_backend.bat           # Windows
# or: uvicorn backend.main:app --port 8000 --reload

# Start frontend (terminal 2)
start_frontend.bat          # Windows
# or: cd frontend && npm run dev
```

Open `http://localhost:3000`, drop an image, and watch the live progress timeline.

---

## Web UI Overview

**Upload** — Drop or select a TIFF/JPG/PNG image. The pipeline starts immediately; uploads queue if a job is already running.

**Processing** — A phase timeline with live log output shows progress in real time via WebSocket.

**Results** — Six tabs:

| Tab | Contents |
|-----|----------|
| Overview | Final reconstructed image, fragment count, timing breakdown, download button |
| Segmentation | SAM mask overlays, per-fragment crop/mask/SDF carousel |
| Contours | Perimeter chart, support-point radar, edge-length bar chart |
| Reconstruction | Step-through merge log with snapshots, gap-score chart |
| Composition | Layer toggle: raw composite / gap mask / inpainted / cleaned |
| Assembly | Interactive canvas — drag, rotate, zoom, undo, export |

**Assembly Board** — Drag fragments to manually adjust placement, reorder layers, then export at 1×/2×/4× resolution with optional LaMa cleaning.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_DEVICE` | `cuda` | `cuda` or `cpu` |
| `LAMA_JIT_PATH` | `lama/big-lama/big-lama.pt` | Path to LaMa TorchScript checkpoint |
| `BACKEND_URL` | `http://localhost:8000` | Consumed by Next.js for API routing |
| `NEXT_PUBLIC_API_BASE` | _(auto from hostname)_ | Override backend URL in browser |
| `UNTORN_INPAINT_REFINE` | `0` | Set `1` to enable LaMa refinement pass |
| `UNTORN_MATCH_TRACE` | _(unset)_ | Set `1` for verbose per-pair rejection logging |

---

## Key Configuration

All tuning constants are in `untorn/config.py`. Common adjustments:

```python
WORKING_MAX_DIM = 1500       # Raise to 2000 if you have ≥6 GB VRAM
MAX_FRAGMENTS = 40           # Upper limit on detected fragments
BG_DIST_LAB_THRESH = 18.0   # LAB ΔE for paper vs. background detection
EDGE_MATCHER_MIN_SCORE = 0.55  # Siamese gate threshold (raise to tighten)
ASSEMBLY_MIN_CONFIDENCE = 0.45 # Min confidence to place a fragment
MAX_ATTACH_COST = 400.0      # Physical fit cost ceiling
```

---

## Project Structure

```
untorn/          Core algorithm (18 modules)
backend/         FastAPI web service
frontend/        Next.js 14 UI
models/          Trained Siamese edge matcher
scripts/         Model download helpers
tools/           Training pipeline, synthetic data generation, tests
docs/            Extended design documentation
data/            Runtime data (input/, output/, debug/)
```

For a full technical description of every module and algorithm, see [UNTORN.md](UNTORN.md).

---

## Supported Input Formats

- TIFF (8-bit and 16-bit, including scientific scans)
- JPEG
- PNG

Images should have the fragments arranged on a background that is visually distinct from the paper (white table, dark cloth, etc.). The background colour is detected automatically from corner samples.

---

## Benchmarks

On an RTX 3070 Ti (8 GB VRAM):

| Fragments | Approx. time |
|-----------|-------------|
| 4 | ~15 s |
| 8 | ~35 s |
| 16 | ~90 s |

VRAM peaks: SAM ~3.5 GB, DINOv2 ~1.5 GB, Siamese + LaMa < 500 MB.

---

## Version

Current release: **v3.3** — see [docs/RELEASES.md](docs/RELEASES.md) for changelog.
