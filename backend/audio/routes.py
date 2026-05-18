"""Sonic Rebellion — audio analysis routes."""
from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Dict

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from audio.config import (
    AUDIO_MAX_CONCURRENT_JOBS, AUDIO_MAX_UPLOAD_MB, AUDIO_PIPELINE_TIMEOUT_SEC,
    AUDIO_UPLOAD_DIR, KEEP_AUDIO_UPLOADS, MAX_REELS,
)
from audio.pipeline.fetcher import fetch_from_urls, fetch_reels
from audio.pipeline.runner import run_pipeline as run_audio_pipeline

log = logging.getLogger("effortlessness.audio")

router = APIRouter()

# Job store + executor — independent of the face pipeline so the two can run
# on overlapping schedules without head-of-line blocking.
audio_jobs: Dict[str, dict] = {}
_audio_semaphore = asyncio.Semaphore(AUDIO_MAX_CONCURRENT_JOBS)
_audio_executor = ThreadPoolExecutor(
    max_workers=AUDIO_MAX_CONCURRENT_JOBS, thread_name_prefix="audio",
)


def shutdown_executor() -> None:
    _audio_executor.shutdown(wait=False, cancel_futures=True)


def _set_audio_stage(job_id: str, stage: str, progress: float) -> None:
    job = audio_jobs.get(job_id)
    if job is not None:
        job["stage"] = stage
        job["progress"] = round(progress, 3)
        log.info("[audio %s] %s (%.0f%%)", job_id[:8], stage, progress * 100)


def _audio_pipeline_body(job_id: str, reel_paths: list[str], retained: list[str]) -> None:
    """Heavy work — runs on the audio executor thread."""
    started = time.time()
    try:
        result = run_audio_pipeline(
            reel_paths,
            progress_cb=lambda stage, p: _set_audio_stage(job_id, stage, p),
        )
        audio_jobs[job_id].update({
            "status":       "done",
            "progress":     1.0,
            "stage":        "done",
            "result":       result,
            "elapsed_secs": round(time.time() - started, 2),
        })
        log.info("[audio %s] finished in %.1fs", job_id[:8], time.time() - started)
    except Exception as exc:                            # noqa: BLE001
        log.error("[audio %s] pipeline failed:\n%s", job_id[:8], traceback.format_exc())
        audio_jobs[job_id].update({
            "status":  "error",
            "message": f"{type(exc).__name__}: {exc}",
            "stage":   "error",
        })
    finally:
        if not KEEP_AUDIO_UPLOADS:
            for p in retained:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:                       # noqa: BLE001
                    pass


async def _run_audio_pipeline(job_id: str, reel_paths: list[str], retained: list[str]) -> None:
    try:
        async with _audio_semaphore:
            audio_jobs[job_id]["status"] = "processing"
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                _audio_executor, _audio_pipeline_body, job_id, reel_paths, retained,
            )
            try:
                await asyncio.wait_for(future, timeout=AUDIO_PIPELINE_TIMEOUT_SEC)
            except (asyncio.TimeoutError, FuturesTimeoutError):
                future.cancel()
                raise RuntimeError(
                    f"Audio pipeline exceeded AUDIO_PIPELINE_TIMEOUT_SEC={AUDIO_PIPELINE_TIMEOUT_SEC}s"
                )
    except Exception as exc:                            # noqa: BLE001
        log.error("[audio %s] schedule failed:\n%s", job_id[:8], traceback.format_exc())
        audio_jobs[job_id].update({
            "status":  "error",
            "message": f"{type(exc).__name__}: {exc}",
            "stage":   "error",
        })


@router.post("/api/audio/analyse")
async def analyse_audio(
    files:       list[UploadFile] | None = File(default=None),
    profile_url: str | None = Form(default=None),
    reel_urls:   str | None = Form(default=None),    # comma-separated
) -> dict:
    """
    Three input modes (at least one is required):
      • `files`:       multipart file upload of up to MAX_REELS video files.
      • `profile_url`: Instagram profile URL — fetched via instaloader.
      • `reel_urls`:   comma-separated list of Reel URLs.
    """
    if not any([files, profile_url, reel_urls]):
        raise HTTPException(400, "Provide files, profile_url, or reel_urls.")

    job_id = str(uuid.uuid4())
    reel_paths: list[str] = []
    retained: list[str] = []

    # ---- Mode 1: file uploads ----
    _AUDIO_EXTS = {
        ".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi",        # video containers
        ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",  # audio-only containers
    }
    if files:
        size_total = 0
        for f in files[:MAX_REELS]:
            if not f.filename:
                continue
            suffix = Path(f.filename).suffix.lower()
            if suffix not in _AUDIO_EXTS:
                continue
            dest = AUDIO_UPLOAD_DIR / f"{job_id}_{len(reel_paths):02d}{suffix}"
            async with aiofiles.open(dest, "wb") as fout:
                while chunk := await f.read(1024 * 1024):
                    size_total += len(chunk)
                    if size_total > AUDIO_MAX_UPLOAD_MB * 1024 * 1024:
                        await fout.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(413, f"Combined upload exceeds {AUDIO_MAX_UPLOAD_MB} MB.")
                    await fout.write(chunk)
            reel_paths.append(str(dest))
            retained.append(str(dest))

    # ---- Mode 2: explicit Reel URLs ----
    if reel_urls and not reel_paths:
        urls = [u.strip() for u in reel_urls.split(",") if u.strip()]
        try:
            fetched = fetch_from_urls(urls)
        except Exception as exc:                        # noqa: BLE001
            raise HTTPException(400, f"reel_urls fetch failed: {exc}")
        reel_paths.extend(fetched)
        retained.extend(fetched)

    # ---- Mode 3: profile URL ----
    if profile_url and not reel_paths:
        try:
            fetched = fetch_reels(profile_url)
        except Exception as exc:                        # noqa: BLE001
            raise HTTPException(400, f"profile_url fetch failed: {exc}")
        reel_paths.extend(fetched)
        retained.extend(fetched)

    if not reel_paths:
        raise HTTPException(400, "No usable reels found from the supplied inputs.")

    audio_jobs[job_id] = {
        "job_id":      job_id,
        "status":      "queued",
        "progress":    0.0,
        "stage":       "queued",
        "n_reels":     len(reel_paths),
        "filenames":   [Path(p).name for p in reel_paths],
        "input_mode":  ("files" if files else "reel_urls" if reel_urls else "profile_url"),
    }

    asyncio.create_task(_run_audio_pipeline(job_id, reel_paths, retained))
    return {"job_id": job_id, "n_reels": len(reel_paths)}


@router.get("/api/audio/result/{job_id}")
def get_audio_result(job_id: str) -> dict:
    job = audio_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown audio job: {job_id}")
    return job


@router.delete("/api/audio/result/{job_id}")
def delete_audio_result(job_id: str) -> dict:
    job = audio_jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(404, f"Unknown audio job: {job_id}")
    return {"deleted": True}
