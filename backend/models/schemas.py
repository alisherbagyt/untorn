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
    current_phase: Optional[str] = None  # preprocessing | segmentation | contours | reconstruction | composition | done
    logs: List[str] = []
    error: Optional[str] = None


class DebugPathsResponse(BaseModel):
    input: Optional[str] = None
    sam_overlay: Optional[str] = None
    segmentation_overlay: Optional[str] = None
    contours_overlay: Optional[str] = None
    neighbor_graph: Optional[str] = None
    composition_raw: Optional[str] = None
    composition_gap: Optional[str] = None
    composition_inpainted: Optional[str] = None
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
    paths: Optional[Dict[str, Any]] = None
