"""
In-memory job state management for UNTORN pipeline jobs.
"""

import threading
from datetime import datetime
from typing import Dict, Any, Optional

# Global job store  (job_id -> job dict)
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

# Single-job semaphore — GPU can only handle one pipeline at a time
PIPELINE_SEMAPHORE = threading.Semaphore(1)


def create_job(stem: str, image_path: str) -> str:
    """Create a new job entry and return its ID."""
    import uuid
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "stem": stem,
            "image_path": image_path,
            "status": "queued",
            "progress": 0,
            "current_phase": "queued",
            "logs": [],
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
        }
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs() -> list:
    with JOBS_LOCK:
        return [dict(j) for j in JOBS.values()]


def get_queue_state(job_id: str) -> tuple[int, int]:
    """Return (queue_position, queued_count). Position is 0 when not queued."""
    with JOBS_LOCK:
        queued = [dict(j) for j in JOBS.values() if j.get("status") == "queued"]
    queued_sorted = sorted(queued, key=lambda x: x.get("created_at", ""))
    queued_count = len(queued_sorted)
    queue_position = 0
    for idx, job in enumerate(queued_sorted, start=1):
        if job.get("job_id") == job_id:
            queue_position = idx
            break
    return queue_position, queued_count


def update_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)
