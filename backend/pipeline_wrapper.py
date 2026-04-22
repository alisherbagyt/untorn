"""
Pipeline wrapper — runs run.py as a subprocess, parses stdout for progress,
and updates the job state store in real time.
"""

import os
import sys
import subprocess
from pathlib import Path

from .services.job_service import update_job, PIPELINE_SEMAPHORE

# Map stdout markers to (progress_pct, phase_name)
_PHASE_START = {
    "[PHASE 0]": (5,  "preprocessing"),
    "[PHASE 1]": (12, "segmentation"),
    "[PHASE 2]": (45, "contours"),
    "[PHASE 3]": (55, "reconstruction"),
    "[PHASE 4]": (82, "composition"),
    "[PHASE 5]": (90, "inpainting"),
}

_PHASE_DONE = {
    "Phase 0 complete": (10, "preprocessing"),
    "Phase 1 complete": (44, "segmentation"),
    "Phase 2 complete": (54, "contours"),
    "Phase 3 complete": (81, "reconstruction"),
    "Phase 4 complete": (89, "composition"),
    "Phase 5 complete": (98, "inpainting"),
}

def _parse_line(line: str):
    """Return (progress, phase) from a pipeline output line, or (None, None)."""
    for marker, val in _PHASE_START.items():
        if marker in line:
            return val
    for marker, val in _PHASE_DONE.items():
        if marker in line:
            return val
    # Detect completion (handles unicode checkmark rendered as ?, replacement char, or ascii)
    line_upper = line.upper()
    if "DONE" in line_upper and ("SAVED TO" in line_upper or "=" * 10 in line):
        return (100, "done")
    # Detect failure
    if "FAILED" in line_upper and any(k in line_upper for k in ["STATUS", "FAILED:", "FAILED "]):
        return (0, "error")
    return (None, None)


def run_pipeline(job_id: str, image_path: str, project_root: str) -> None:
    """
    Blocking — meant to run in a dedicated daemon thread.
    Acquires PIPELINE_SEMAPHORE so only one pipeline runs at a time.
    """
    update_job(job_id, status="queued", progress=0, current_phase="queued",
               logs=["Waiting for GPU slot..."])

    with PIPELINE_SEMAPHORE:
        _execute(job_id, image_path, project_root)


def _execute(job_id: str, image_path: str, project_root: str) -> None:
    update_job(job_id, status="processing", progress=2, current_phase="preprocessing",
               logs=["Starting pipeline..."])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, "run.py", image_path]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        logs: list[str] = []
        current_progress = 2

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue

            logs.append(line)
            progress, phase = _parse_line(line)

            if phase == "error":
                update_job(job_id, status="error", current_phase="error",
                           logs=logs[-80:], error=line)
                proc.wait()
                return

            if phase == "done":
                update_job(job_id, status="done", progress=100,
                           current_phase="done", logs=logs[-80:])
                proc.wait()
                return

            if progress is not None and progress > current_progress:
                current_progress = progress
                update_job(job_id, progress=progress, current_phase=phase,
                           logs=logs[-80:])
            elif len(logs) % 8 == 0:
                # Throttled log flush
                update_job(job_id, logs=logs[-80:])

        proc.wait()

        if proc.returncode != 0:
            err_msg = f"Pipeline exited with code {proc.returncode}"
            update_job(job_id, status="error", error=err_msg, logs=logs[-80:])
        else:
            # Only set done if we haven't already detected error or done via stdout
            current_status = _get_job_status(job_id)
            if current_status not in ("done", "error"):
                update_job(job_id, status="done", progress=100,
                           current_phase="done", logs=logs[-80:])

    except Exception as exc:
        update_job(job_id, status="error", error=str(exc),
                   logs=[f"Fatal error: {exc}"])


def _get_job_status(job_id: str) -> str:
    from .services.job_service import get_job
    job = get_job(job_id)
    return job["status"] if job else "error"
