"""
UNTORN — FastAPI backend
========================
Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import io
import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from .models.schemas import DebugResponse, JobStatusResponse, ProcessResponse
from .pipeline_wrapper import run_pipeline
from .services.job_service import create_job, get_job, list_jobs, update_job

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_DIR    = DATA_DIR / "input"
OUTPUT_DIR   = DATA_DIR / "output"
DEBUG_DIR    = DATA_DIR / "debug"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="UNTORN API",
    description="Torn Paper Reconstruction Pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/jobs")
async def get_jobs():
    return list_jobs()


@app.post("/process", response_model=ProcessResponse)
async def process_image(file: UploadFile = File(...)):
    """Upload an image and start the reconstruction pipeline."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a unique stem to avoid debug-dir collisions
    original_stem = Path(file.filename).stem
    suffix = Path(file.filename).suffix or ".png"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_stem = f"{original_stem}_{timestamp}"
    save_filename = f"{unique_stem}{suffix}"
    save_path = INPUT_DIR / save_filename

    async with aiofiles.open(str(save_path), "wb") as f:
        content = await file.read()
        await f.write(content)

    job_id = create_job(stem=unique_stem, image_path=str(save_path))

    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, str(save_path), str(PROJECT_ROOT)),
        daemon=True,
        name=f"pipeline-{job_id[:8]}",
    )
    thread.start()

    return ProcessResponse(job_id=job_id, message="Processing started")


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status and progress."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status":       job["status"],
        "progress":     job["progress"],
        "current_phase": job["current_phase"],
        "logs":         job.get("logs", []),
        "error":        job.get("error"),
    }


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Return the final reconstructed image."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")

    stem = job["stem"]
    output_path = OUTPUT_DIR / f"{stem}_reconstructed.png"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output image not found")

    return FileResponse(str(output_path), media_type="image/png",
                        filename=f"{stem}_reconstructed.png")


@app.get("/debug/{job_id}")
async def get_debug(job_id: str):
    """Return all debug metadata for a job as structured JSON."""
    def resolve_debug_dir() -> Path:
        job = get_job(job_id)
        if job:
            return DEBUG_DIR / job["stem"]
        # Fallback: allow direct access by debug stem (useful after backend restart)
        stem_dir = DEBUG_DIR / job_id
        if stem_dir.exists() and stem_dir.is_dir():
            return stem_dir
        raise HTTPException(status_code=404, detail="Job not found")

    debug_dir = resolve_debug_dir()
    stem = debug_dir.name
    result: dict = {}

    def read_json(path: Path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    result["pipeline_meta"] = read_json(debug_dir / "pipeline_meta.json")
    result["fragments"] = read_json(debug_dir / "segmentation" / "fragments_meta.json")
    result["contours"] = read_json(debug_dir / "contours" / "contours_meta.json")
    result["neighbors"] = (
        read_json(debug_dir / "matching" / "neighbors.json")
        or read_json(debug_dir / "matching" / "match_scores.json")
    )
    raw_steps = read_json(debug_dir / "reconstruction" / "merge_log.json") or []
    result["translations"] = read_json(debug_dir / "reconstruction" / "final_translations.json") or {}
    result["composition"] = read_json(debug_dir / "composition" / "composition_meta.json")
    result["inpainting"] = read_json(debug_dir / "inpainting" / "inpainting_meta.json")

    translations = result["translations"] if isinstance(result["translations"], dict) else {}

    def _to_float(value, default=0.0):
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    normalized_steps = []
    if isinstance(raw_steps, list):
        for idx, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                continue
            attached = step.get("attached", step.get("frag_j", -1))
            anchor = step.get("anchor", step.get("frag_i", -1))
            attached_trans = translations.get(str(attached), {}) if isinstance(translations, dict) else {}

            normalized_steps.append({
                "step": int(step.get("step", idx + 1)),
                "phase": step.get("phase") or ("joint" if step.get("joint_merge") else "merge"),
                "anchor": int(anchor) if isinstance(anchor, (int, float, str)) and str(anchor).lstrip("-").isdigit() else -1,
                "attached": int(attached) if isinstance(attached, (int, float, str)) and str(attached).lstrip("-").isdigit() else -1,
                "dx": _to_float(step.get("dx", attached_trans.get("dx", 0.0))),
                "dy": _to_float(step.get("dy", attached_trans.get("dy", 0.0))),
                "gap_score": _to_float(step.get("gap_score", step.get("stotal", 0.0))),
            })
    result["steps"] = normalized_steps

    frag_ids = []
    if isinstance(result["fragments"], list):
        for frag in result["fragments"]:
            if isinstance(frag, dict):
                fid = frag.get("id")
                if isinstance(fid, int):
                    frag_ids.append(fid)
    if not frag_ids:
        frag_ids = list(range(len(result["fragments"] or [])))

    base = f"/debug/image/{job_id}"

    def pick_single(*rel_candidates: str) -> Optional[str]:
        for rel in rel_candidates:
            if (debug_dir / rel).exists():
                return f"{base}/{rel}"
        return None

    def pick_many(candidates: list[str]) -> list[str]:
        items = []
        for rel in candidates:
            if (debug_dir / rel).exists():
                items.append(f"{base}/{rel}")
            else:
                items.append("")
        return items

    reconstruction_step_files = sorted((debug_dir / "reconstruction").glob("step_*.png"))
    reconstruction_steps = [
        f"{base}/{p.relative_to(debug_dir).as_posix()}"
        for p in reconstruction_step_files
    ]

    result["paths"] = {
        "input": pick_single("00_input.png"),
        "sam_overlay": pick_single("segmentation/01_raw_sam_overlay.png"),
        "segmentation_overlay": pick_single(
            "segmentation/06_final_fragments_overlay.png",
            "segmentation/05_final_fragments_overlay.png",
        ),
        "contours_overlay": pick_single("contours/all_support_points.png"),
        "neighbor_graph": pick_single("matching/neighbor_graph.png"),
        "composition_raw": pick_single("composition/01_raw_composite.png"),
        "composition_gap": pick_single("composition/03_gap_mask.png"),
        "composition_inpainted": pick_single("composition/04_inpainted.png"),
        "inpainting_before": pick_single("inpainting/01_before.png"),
        "inpainting_mask": pick_single("inpainting/02_scar_mask.png"),
        "inpainting_cleaned": pick_single("inpainting/03_cleaned.png"),
        "fragment_crops": pick_many([
            f"segmentation/07_crop_{i:02d}.png" if (debug_dir / f"segmentation/07_crop_{i:02d}.png").exists() else f"segmentation/06_crop_{i:02d}.png"
            for i in frag_ids
        ]),
        "fragment_masks": pick_many([
            f"segmentation/05_mask_final_{i:02d}.png" if (debug_dir / f"segmentation/05_mask_final_{i:02d}.png").exists() else f"segmentation/04_mask_final_{i:02d}.png"
            for i in frag_ids
        ]),
        "fragment_sdfs": pick_many([f"contours/sdf_{i:02d}.png" for i in frag_ids]),
        "fragment_support": pick_many([f"contours/support_pts_{i:02d}.png" for i in frag_ids]),
        "reconstruction_steps": reconstruction_steps,
    }

    return result


@app.get("/debug/image/{job_id}/{path:path}")
async def get_debug_image(job_id: str, path: str):
    """Serve a debug image from the job's debug directory."""
    job = get_job(job_id)
    stem = job["stem"] if job else job_id
    image_path = DEBUG_DIR / stem / path
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")

    # Determine media type
    suffix = image_path.suffix.lower()
    media_map = {".png": "image/png", ".jpg": "image/jpeg",
                 ".jpeg": "image/jpeg", ".tif": "image/tiff"}
    media_type = media_map.get(suffix, "image/png")

    return FileResponse(str(image_path), media_type=media_type)


# ── Assembly Board endpoints ──────────────────────────────────────────────────


def _make_transparent_fragment(debug_dir: Path, frag_id: int) -> Image.Image:
    """Combine fragment crop with its mask to produce a transparent PNG.
    Caches the result to disk so subsequent requests are fast file reads."""
    cache_dir = debug_dir / "board_cache"
    cached_path = cache_dir / f"frag_{frag_id:02d}.png"

    if cached_path.exists():
        return Image.open(cached_path)

    crop_path = debug_dir / "segmentation" / f"07_crop_{frag_id:02d}.png"
    if not crop_path.exists():
        crop_path = debug_dir / "segmentation" / f"06_crop_{frag_id:02d}.png"

    mask_path = debug_dir / "segmentation" / f"05_mask_final_{frag_id:02d}.png"
    if not mask_path.exists():
        mask_path = debug_dir / "segmentation" / f"04_mask_final_{frag_id:02d}.png"

    if not crop_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Fragment {frag_id} images not found")

    crop = Image.open(crop_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")

    # Mask may be full-image size; crop it to the fragment bbox
    if mask.size != crop.size:
        # We need the fragment metadata to know the bbox
        meta_path = debug_dir / "segmentation" / "fragments_meta.json"
        with open(meta_path) as f:
            fragments = json.load(f)
        frag = fragments[frag_id]
        bx, by, bw, bh = frag["bbox_xywh"]
        mask = mask.crop((bx, by, bx + bw, by + bh))

    if mask.size != crop.size:
        mask = mask.resize(crop.size, Image.NEAREST)

    crop.putalpha(mask)

    # Cache to disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    crop.save(str(cached_path), format="PNG")

    return crop


@app.get("/board/data/{job_id}")
async def get_board_data(job_id: str):
    """Return combined data needed for the interactive assembly board."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    stem = job["stem"]
    debug_dir = DEBUG_DIR / stem

    def read_json(path: Path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    fragments = read_json(debug_dir / "segmentation" / "fragments_meta.json")
    translations = read_json(debug_dir / "reconstruction" / "final_translations.json")
    composition = read_json(debug_dir / "composition" / "composition_meta.json")
    contours = read_json(debug_dir / "contours" / "contours_meta.json")

    if not fragments or not translations or not composition:
        raise HTTPException(status_code=400, detail="Incomplete pipeline data")

    # Build per-fragment board data
    board_fragments = []
    for frag in fragments:
        fid = frag["id"]
        fid_str = str(fid)
        trans = translations.get(fid_str, {"dx": 0, "dy": 0, "placed": False})
        bx, by, bw, bh = frag["bbox_xywh"]

        # Position on the composition canvas
        offset_x = composition.get("offset_x", 0)
        offset_y = composition.get("offset_y", 0)
        canvas_x = bx + trans["dx"] - offset_x
        canvas_y = by + trans["dy"] - offset_y

        board_fragments.append({
            "id": fid,
            "x": round(canvas_x, 1),
            "y": round(canvas_y, 1),
            "width": bw,
            "height": bh,
            "rotation": 0,
            "placed": trans.get("placed", False),
            "area": frag["area"],
            "centroid": frag["centroid"],
            "imageUrl": f"/board/fragment/{job_id}/{fid}",
        })

    return {
        "canvas": {
            "width": composition["canvas_w"],
            "height": composition["canvas_h"],
        },
        "fragments": board_fragments,
    }


@app.get("/board/fragment/{job_id}/{fragment_id}")
async def get_board_fragment(job_id: str, fragment_id: int):
    """Return a transparent PNG of the specified fragment."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    debug_dir = DEBUG_DIR / job["stem"]

    # Check if cached file exists — serve directly as FileResponse (fastest)
    cached_path = debug_dir / "board_cache" / f"frag_{fragment_id:02d}.png"
    if cached_path.exists():
        return FileResponse(
            str(cached_path),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # Generate (and cache) the transparent fragment
    try:
        img = _make_transparent_fragment(debug_dir, fragment_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Cache-Control": "public, max-age=3600"})


class FragmentPlacement(BaseModel):
    id: int
    x: float
    y: float
    rotation: float  # degrees


class BoardExportRequest(BaseModel):
    fragments: list[FragmentPlacement]
    canvas_width: int
    canvas_height: int
    scale: int = 1  # 1=original, 2=2x, 4=4x — multiplies output resolution
    clean: bool = True    # run LaMa scar-cleaning on the final composite
    refine: bool = False  # use LaMa refinement mode (slow, high quality)


@app.post("/board/export/{job_id}")
async def export_board(job_id: str, req: BoardExportRequest):
    """Composite fragments at user-specified positions with seam healing, return PNG."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    debug_dir = DEBUG_DIR / job["stem"]

    # Clamp scale to safe range (1–8)
    scale = max(1, min(req.scale, 8))

    # Load all transparent fragment images
    frag_images: dict[int, Image.Image] = {}
    for fp in req.fragments:
        try:
            frag_images[fp.id] = _make_transparent_fragment(debug_dir, fp.id)
        except FileNotFoundError:
            continue

    if not frag_images:
        raise HTTPException(status_code=400, detail="No valid fragments")

    # Create canvas at scaled resolution for high-quality output
    out_w = req.canvas_width * scale
    out_h = req.canvas_height * scale
    canvas = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))

    placed_masks: list[tuple[np.ndarray, int, int]] = []

    for fp in req.fragments:
        img = frag_images.get(fp.id)
        if not img:
            continue

        # Upscale fragment with high-quality resampling
        if scale > 1:
            img = img.resize(
                (img.width * scale, img.height * scale),
                Image.LANCZOS,
            )

        # Rotate if needed
        if abs(fp.rotation) > 0.01:
            orig_w, orig_h = img.size
            img = img.rotate(-fp.rotation, resample=Image.BICUBIC, expand=True)
            new_w, new_h = img.size
            cx_offset = (new_w - orig_w) / 2
            cy_offset = (new_h - orig_h) / 2
            px = round(fp.x * scale - cx_offset)
            py = round(fp.y * scale - cy_offset)
        else:
            px = round(fp.x * scale)
            py = round(fp.y * scale)

        canvas.paste(img, (px, py), img)

        alpha = np.array(img.split()[-1])
        placed_masks.append((alpha, px, py))

    # Compose RGB + coverage mask from placed alpha maps
    output = Image.new("RGB", canvas.size, (255, 255, 255))
    output.paste(canvas, mask=canvas.split()[-1])
    rgb_arr = np.array(output)

    coverage = _build_coverage(canvas.size, placed_masks)

    if req.clean:
        # LaMa seam cleaning replaces the classical seam_healing entirely.
        try:
            from untorn.inpainting import build_scar_mask, inpaint, is_available as lama_available
            if not lama_available():
                raise RuntimeError("LaMa checkpoint missing")

            band_px = max(3, 4 * scale)
            scar = build_scar_mask(rgb_arr, coverage, gap_mask=None, band_px=band_px)
            if int((scar > 0).sum()) > 0:
                rgb_arr = inpaint(rgb_arr, scar, tile=True, refine=bool(req.refine))
            output = Image.fromarray(rgb_arr, "RGB")
        except Exception as exc:
            # Fall back to the legacy classical seam healing so export never fails
            print(f"[export] LaMa cleaning failed, falling back to seam healing: {exc}")
            _apply_seam_healing(canvas, placed_masks, threshold=5 * scale)
            output = Image.new("RGB", canvas.size, (255, 255, 255))
            output.paste(canvas, mask=canvas.split()[-1])
    else:
        # User explicitly asked for raw output — run classical seam healing only.
        _apply_seam_healing(canvas, placed_masks, threshold=5 * scale)
        output = Image.new("RGB", canvas.size, (255, 255, 255))
        output.paste(canvas, mask=canvas.split()[-1])

    buf = io.BytesIO()
    output.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    tag = f"_{scale}x" if scale > 1 else ""
    if req.clean:
        tag += "_clean" + ("_refined" if req.refine else "")
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=untorn_assembly_{job['stem']}{tag}.png"},
    )


def _build_coverage(canvas_size: tuple[int, int],
                    placed_masks: list[tuple[np.ndarray, int, int]]) -> np.ndarray:
    """Build an HxW uint8 coverage mask from placed fragment alpha maps."""
    w, h = canvas_size
    cov = np.zeros((h, w), dtype=np.uint8)
    for alpha, px, py in placed_masks:
        ah, aw = alpha.shape
        y1, y2 = max(0, py), min(h, py + ah)
        x1, x2 = max(0, px), min(w, px + aw)
        ay1, ay2 = y1 - py, y2 - py
        ax1, ax2 = x1 - px, x2 - px
        region = alpha[ay1:ay2, ax1:ax2] > 128
        cov[y1:y2, x1:x2][region] = 255
    return cov


def _apply_seam_healing(canvas: Image.Image, placed_masks: list, threshold: int = 5):
    """
    Simple seam healing: for pixels in the gap between two close fragments,
    blend colors from the nearest edge pixels of both fragments.
    Only affects the thin strip between nearly-touching fragments.
    """
    canvas_arr = np.array(canvas)
    h, w = canvas_arr.shape[:2]

    # Build a combined alpha map and a fragment-ID map
    frag_map = np.full((h, w), -1, dtype=np.int16)
    alpha_map = np.zeros((h, w), dtype=np.uint8)

    for idx, (alpha, px, py) in enumerate(placed_masks):
        ah, aw = alpha.shape
        # Clip to canvas bounds
        y1, y2 = max(0, py), min(h, py + ah)
        x1, x2 = max(0, px), min(w, px + aw)
        ay1, ay2 = y1 - py, y2 - py
        ax1, ax2 = x1 - px, x2 - px

        mask_region = alpha[ay1:ay2, ax1:ax2] > 128
        frag_map[y1:y2, x1:x2][mask_region] = idx
        alpha_map[y1:y2, x1:x2][mask_region] = 255

    # Find gap pixels: not covered by any fragment
    gap_mask = alpha_map < 128

    if not np.any(gap_mask):
        return

    # For each gap pixel, check distance to nearest fragment edge
    from scipy.ndimage import distance_transform_edt, label

    # Distance from gap to any fragment
    dist_to_frag = distance_transform_edt(gap_mask)

    # Only heal pixels within threshold distance of fragments
    heal_candidates = gap_mask & (dist_to_frag <= threshold)

    if not np.any(heal_candidates):
        return

    # For healing, dilate each fragment slightly and check for overlap zones
    from scipy.ndimage import binary_dilation

    struct = np.ones((threshold * 2 + 1, threshold * 2 + 1), dtype=bool)
    dilated_maps: list[np.ndarray] = []
    for idx in range(len(placed_masks)):
        frag_mask = frag_map == idx
        dilated = binary_dilation(frag_mask, structure=struct)
        dilated_maps.append(dilated)

    # Find pixels where at least 2 dilated fragments overlap AND it's a gap pixel
    overlap_count = np.zeros((h, w), dtype=np.int16)
    for dm in dilated_maps:
        overlap_count += dm.astype(np.int16)

    seam_pixels = heal_candidates & (overlap_count >= 2)

    if not np.any(seam_pixels):
        return

    # For seam pixels, blend using distance-weighted average from nearby fragment edges
    ys, xs = np.where(seam_pixels)

    for y, x in zip(ys, xs):
        colors = []
        weights = []
        for idx, (alpha, px, py) in enumerate(placed_masks):
            if not dilated_maps[idx][y, x]:
                continue
            # Find nearest opaque pixel from this fragment
            frag_mask = frag_map == idx
            # Simple: sample the nearest edge pixel in a small window
            r = threshold + 2
            y1c, y2c = max(0, y - r), min(h, y + r + 1)
            x1c, x2c = max(0, x - r), min(w, x + r + 1)
            local_frag = frag_mask[y1c:y2c, x1c:x2c]
            if not np.any(local_frag):
                continue
            local_ys, local_xs = np.where(local_frag)
            dists = np.sqrt((local_ys - (y - y1c)) ** 2 + (local_xs - (x - x1c)) ** 2)
            nearest_idx = np.argmin(dists)
            nearest_y = y1c + local_ys[nearest_idx]
            nearest_x = x1c + local_xs[nearest_idx]
            d = max(dists[nearest_idx], 0.5)
            colors.append(canvas_arr[nearest_y, nearest_x, :3].astype(np.float64))
            weights.append(1.0 / d)

        if colors:
            weights_arr = np.array(weights)
            weights_arr /= weights_arr.sum()
            blended = sum(c * w for c, w in zip(colors, weights_arr))
            canvas_arr[y, x, :3] = np.clip(blended, 0, 255).astype(np.uint8)
            canvas_arr[y, x, 3] = 255

    # Write back
    healed = Image.fromarray(canvas_arr, "RGBA")
    canvas.paste(healed)


# ── WebSocket real-time progress ──────────────────────────────────────────────

@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """Push job status updates to connected clients every 500ms."""
    await websocket.accept()
    last_payload: dict = {}

    try:
        while True:
            job = get_job(job_id)
            if job:
                payload = {
                    "status":        job["status"],
                    "progress":      job["progress"],
                    "current_phase": job["current_phase"],
                    "logs":          (job.get("logs") or [])[-15:],
                    "error":         job.get("error"),
                }
                if payload != last_payload:
                    await websocket.send_json(payload)
                    last_payload = payload

                if job["status"] in ("done", "error"):
                    break

            await asyncio.sleep(0.4)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
