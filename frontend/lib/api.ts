import { getApiBase, getWsBase } from "./utils";

export interface JobStatus {
  job_id: string;
  status: "queued" | "processing" | "done" | "error";
  progress: number;
  current_phase: string;
  logs: string[];
  error?: string;
  queue_position?: number;
  queued_count?: number;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface DebugData {
  pipeline_meta?: {
    input: string;
    output: string;
    image_size: { w: number; h: number };
    working_size: { w: number; h: number };
    scale_factor: number;
    n_fragments: number;
    status: string;
    timings: Record<string, number>;
    edge_matcher_loaded?: boolean;
    missing_fragment?: boolean;
    hole_counts?: Record<string, number>;
  };
  fragments?: Array<{
    id: number;
    area: number;
    bbox_xywh: [number, number, number, number];
    centroid: [number, number];
  }>;
  contours?: Array<{
    id: number;
    n_support_points: number;
    n_edge_segments: number;
    n_boundary_pixels: number;
    total_perimeter: number;
    edge_lengths: number[];
  }>;
  neighbors?: Array<{
    pair: string;
    bbox_gap: number;
    centroid_dist: number;
  }>;
  steps?: Array<{
    step: number;
    phase: string;
    anchor: number;
    attached: number;
    dx: number;
    dy: number;
    gap_score: number;
  }>;
  translations?: Record<string, { dx: number; dy: number; placed: boolean }>;
  composition?: {
    canvas_w: number;
    canvas_h: number;
    gap_pixels_inpainted: number;
    gap_pixels_detected?: number;
    final_w: number;
    final_h: number;
  };
  inpainting?: {
    status: string;
    error?: string | null;
    backend?: "jit" | "simple" | "saic" | null;
    mask_pixels?: number;
    duration_s?: number;
    device?: string;
    refine?: boolean;
    band_px?: number;
    ink_threshold?: number;
    hole_counts?: Record<string, number>;
    largest_hole_frac?: number;
    missing_fragment?: boolean;
  };
  paths?: {
    input?: string;
    sam_overlay?: string;
    segmentation_overlay?: string;
    contours_overlay?: string;
    neighbor_graph?: string;
    composition_raw?: string;
    composition_gap?: string;
    composition_inpainted?: string;
    inpainting_before?: string;
    inpainting_mask?: string;
    inpainting_cleaned?: string;
    inpainting_holes?: string;
    inpainting_repair_mask?: string;
    fragment_crops: string[];
    fragment_masks: string[];
    fragment_sdfs: string[];
    fragment_support: string[];
    reconstruction_steps: string[];
  };
}

export async function uploadImage(file: File): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${getApiBase()}/process`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

export async function fetchStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${getApiBase()}/status/${jobId}`);
  if (!res.ok) throw new Error(`Status fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchDebug(jobId: string): Promise<DebugData> {
  const res = await fetch(`${getApiBase()}/debug/${jobId}`);
  if (!res.ok) throw new Error(`Debug fetch failed: ${res.statusText}`);
  return res.json();
}

export function debugImageUrl(path: string): string {
  // path is already like /debug/image/{job_id}/...
  return `${getApiBase()}${path}`;
}

export function resultImageUrl(jobId: string): string {
  return `${getApiBase()}/result/${jobId}`;
}

export function createWebSocket(jobId: string): WebSocket {
  return new WebSocket(`${getWsBase()}/ws/${jobId}`);
}

// ── Assembly Board ──────────────────────────────────────────────────────────

export interface BoardFragment {
  id: number;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  placed: boolean;
  area: number;
  centroid: [number, number];
  imageUrl: string;
}

export interface BoardData {
  canvas: { width: number; height: number };
  fragments: BoardFragment[];
}

export async function fetchBoardData(jobId: string): Promise<BoardData> {
  const res = await fetch(`${getApiBase()}/board/data/${jobId}`);
  if (!res.ok) throw new Error(`Board data fetch failed: ${res.statusText}`);
  return res.json();
}

export function boardFragmentUrl(jobId: string, fragmentId: number): string {
  return `${getApiBase()}/board/fragment/${jobId}/${fragmentId}`;
}

export async function exportBoard(
  jobId: string,
  fragments: Array<{ id: number; x: number; y: number; rotation: number }>,
  canvasWidth: number,
  canvasHeight: number,
  scale: number = 1,
  opts: { clean?: boolean; refine?: boolean } = {}
): Promise<Blob> {
  const res = await fetch(`${getApiBase()}/board/export/${jobId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      fragments,
      canvas_width: canvasWidth,
      canvas_height: canvasHeight,
      scale,
      clean: opts.clean !== undefined ? opts.clean : true,
      refine: !!opts.refine,
    }),
  });
  if (!res.ok) throw new Error(`Export failed: ${res.statusText}`);
  return res.blob();
}
