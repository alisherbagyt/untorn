"""
Pydantic schemas for the UNTORN API.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ProcessResponse(BaseModel):
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | processing | done | error
    progress: int  # 0-100
    current_phase: Optional[str] = None  # preprocessing | segmentation | contours | reconstruction | composition | gap_fill | done
    logs: List[str] = []
    error: Optional[str] = None
    queue_position: Optional[int] = None
    queued_count: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class DebugPathsResponse(BaseModel):
    input: Optional[str] = None
    sam_overlay: Optional[str] = None
    segmentation_overlay: Optional[str] = None
    contours_overlay: Optional[str] = None
    neighbor_graph: Optional[str] = None
    composition_raw: Optional[str] = None
    composition_gap: Optional[str] = None
    composition_inpainted: Optional[str] = None
    inpainting_before: Optional[str] = None
    inpainting_mask: Optional[str] = None
    inpainting_cleaned: Optional[str] = None
    inpainting_holes: Optional[str] = None
    inpainting_repair_mask: Optional[str] = None
    fragment_crops: List[str] = []
    fragment_masks: List[str] = []
    fragment_sdfs: List[str] = []
    fragment_support: List[str] = []
    reconstruction_steps: List[str] = []


class DebugResponse(BaseModel):
    pipeline_meta: Optional[Dict[str, Any]] = None
    fragments: Optional[List[Dict[str, Any]]] = None
    contours: Optional[List[Dict[str, Any]]] = None
    neighbors: Optional[List[Dict[str, Any]]] = None
    steps: Optional[List[Dict[str, Any]]] = None
    translations: Optional[Dict[str, Any]] = None
    composition: Optional[Dict[str, Any]] = None
    inpainting: Optional[Dict[str, Any]] = None
    paths: Optional[Dict[str, Any]] = None
