"""Presence analyzer — pose-based scoring routes.

Runs the standalone presence-analyzer CLI as a subprocess (see
`presence/pipeline/runner.py`) so its numpy/mediapipe pins don't clash
with py-feat's pins in the face pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Dict

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from presence.pipeline.runner import run_presence

log = logging.getLogger("effortlessness.presence")

# Runs in a subprocess so concurrency is fine — pin MAX_CONCURRENT_PRESENCE
# to 2 to keep memory bounded.
PRESENCE_MAX_CONCURRENT_JOBS = int(os.getenv("PRESENCE_MAX_CONCURRENT_JOBS", "2"))
PRESENCE_PIPELINE_TIMEOUT_SEC = int(os.getenv("PRESENCE_PIPELINE_TIMEOUT_SEC", "900"))
PRESENCE_UPLOAD_DIR = Path(os.getenv("PRESENCE_UPLOAD_DIR", "/tmp/presence_uploads"))
PRESENCE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
KEEP_PRESENCE_UPLOADS = os.getenv("KEEP_PRESENCE_UPLOADS", "1") == "1"
# Reuses MAX_UPLOAD_MB from the face routes' env var for parity.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))

router = APIRouter()

presence_jobs: Dict[str, dict] = {}
_presence_semaphore = asyncio.Semaphore(PRESENCE_MAX_CONCURRENT_JOBS)
_presence_executor = ThreadPoolExecutor(
    max_workers=PRESENCE_MAX_CONCURRENT_JOBS, thread_name_prefix="presence",
)


def shutdown_executor() -> None:
    _presence_executor.shutdown(wait=False, cancel_futures=True)


_MIME_BY_SUFFIX = {
    ".mp4":  "video/mp4",
    ".mov":  "video/quicktime",
    ".m4v":  "video/mp4",
    ".webm": "video/webm",
    ".mkv":  "video/x-matroska",
    ".avi":  "video/x-msvideo",
}


def _set_presence_stage(job_id: str, stage: str, progress: float) -> None:
    job = presence_jobs.get(job_id)
    if job is not None:
        job["stage"] = stage
        job["progress"] = round(progress, 3)
        log.info("[presence %s] %s (%.0f%%)", job_id[:8], stage, progress * 100)


def _presence_pipeline_body(job_id: str, video_path: Path, filename: str) -> None:
    """Heavy work — shells out to the presence-analyzer CLI."""
    started = time.time()
    _set_presence_stage(job_id, "starting presence analyzer", 0.05)
    result = run_presence(
        video_path,
        progress_cb=lambda stage, p: _set_presence_stage(job_id, stage, p),
        timeout_sec=PRESENCE_PIPELINE_TIMEOUT_SEC,
    )

    # Augment the raw CLI output with backend metadata so the frontend can
    # show filename, elapsed, etc. without inventing fields.
    augmented = {
        "job_id":       job_id,
        "filename":     filename,
        "video_path":   result.get("video_path", str(video_path)),
        "content_hash": result.get("content_hash"),
        "duration_secs": result.get("features", {}).get("duration_sec"),
        "fps_effective": result.get("features", {}).get("fps_effective"),
        "features":     result.get("features", {}),
        "score":        result.get("score", {}),
        "timings":      result.get("timings", {}),
        "warnings":     result.get("warnings", []),
    }
    presence_jobs[job_id].update({
        "status":       "done",
        "progress":     1.0,
        "stage":        "done",
        "result":       augmented,
        "elapsed_secs": round(time.time() - started, 2),
    })
    log.info("[presence %s] finished in %.1fs", job_id[:8], time.time() - started)


async def _run_presence_pipeline(job_id: str, video_path: Path, filename: str) -> None:
    try:
        async with _presence_semaphore:
            presence_jobs[job_id]["status"] = "processing"
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                _presence_executor, _presence_pipeline_body, job_id, video_path, filename,
            )
            try:
                await asyncio.wait_for(future, timeout=PRESENCE_PIPELINE_TIMEOUT_SEC + 60)
            except (asyncio.TimeoutError, FuturesTimeoutError):
                future.cancel()
                raise RuntimeError(
                    f"Presence pipeline exceeded {PRESENCE_PIPELINE_TIMEOUT_SEC}s"
                )
    except Exception as exc:                                    # noqa: BLE001
        log.error("[presence %s] pipeline failed:\n%s", job_id[:8], traceback.format_exc())
        presence_jobs[job_id].update({
            "status":  "error",
            "message": f"{type(exc).__name__}: {exc}",
            "stage":   "error",
        })
    finally:
        if not KEEP_PRESENCE_UPLOADS:
            try:
                video_path.unlink(missing_ok=True)
            except Exception:                                   # noqa: BLE001
                pass


@router.post("/api/presence/analyse")
async def analyse_presence(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(400, "Missing filename.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}:
        raise HTTPException(400, f"Unsupported video format: {suffix}")

    job_id = str(uuid.uuid4())
    video_path = PRESENCE_UPLOAD_DIR / f"{job_id}{suffix}"

    size = 0
    async with aiofiles.open(video_path, "wb") as fout:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                await fout.close()
                video_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB.")
            await fout.write(chunk)

    presence_jobs[job_id] = {
        "job_id":     job_id,
        "status":     "queued",
        "progress":   0.0,
        "stage":      "queued",
        "filename":   file.filename,
        "size_bytes": size,
        "video_path": str(video_path) if KEEP_PRESENCE_UPLOADS else None,
        "video_suffix": suffix,
    }

    asyncio.create_task(_run_presence_pipeline(job_id, video_path, file.filename))
    return {"job_id": job_id}


@router.get("/api/presence/result/{job_id}")
def get_presence_result(job_id: str) -> dict:
    job = presence_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown presence job: {job_id}")
    return job


@router.get("/api/presence/video/{job_id}")
def get_presence_video(job_id: str):
    job = presence_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown presence job: {job_id}")
    path_str = job.get("video_path")
    if not path_str:
        raise HTTPException(404, "Video not retained for this job.")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(404, "Video file no longer on disk.")
    suffix = job.get("video_suffix", "").lower()
    media = _MIME_BY_SUFFIX.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=job.get("filename") or path.name)


@router.delete("/api/presence/result/{job_id}")
def delete_presence_result(job_id: str) -> dict:
    job = presence_jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(404, f"Unknown presence job: {job_id}")
    path_str = job.get("video_path")
    if path_str:
        try:
            Path(path_str).unlink(missing_ok=True)
        except Exception:                                       # noqa: BLE001
            pass
    return {"deleted": True}
