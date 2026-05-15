"""
Effortlessness Analyzer — FastAPI entrypoint.

Endpoints:
    POST /api/analyse           — upload a video, start a background job
    GET  /api/result/{job_id}   — poll job status / get final result
    GET  /api/health            — liveness probe

Robustness in this build:
  • Pipeline runs are **serialised** through a process-wide Semaphore(1)
    because the underlying torch + py-feat model graph is not safe under
    concurrent invocation. The HTTP request still returns immediately;
    queued jobs wait on the semaphore.
  • Each pipeline run has a **wall-clock timeout** (`PIPELINE_TIMEOUT_SEC`).
    On expiry the job is marked `error` instead of hanging forever.
  • Uploaded video files are deleted in `finally` regardless of how the
    pipeline exits.
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
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Load backend/.env into os.environ on startup if python-dotenv is available.
# Soft dependency — falls back to OS env vars if dotenv isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:                                  # noqa: BLE001
    pass

from pipeline.aggregator import aggregate_scores
from pipeline.au_detector import detect_aus, warm_detector
from pipeline.baseline import calibrate_baseline
from pipeline.extractor import extract_frames
from pipeline.face_tracker import track_face
from pipeline.insights import generate_insights
from pipeline.methods.asymmetry import compute_asymmetry
from pipeline.methods.coherence import compute_coherence
from pipeline.methods.drift_blink import analyse_drift_blink
from pipeline.methods.micro_expr import detect_micro_expressions
from pipeline.methods.tension_aus import score_tension_aus
from pipeline.visualization import build_visualization

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("effortlessness")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/effortlessness_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FPS = int(os.getenv("TARGET_FPS", "12"))
# Default ON so /api/video/{job_id} can stream back the original video for the
# overlay after a page refresh. Set KEEP_UPLOADS=0 to revert to delete-on-done.
KEEP_UPLOADS = os.getenv("KEEP_UPLOADS", "1") == "1"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
PIPELINE_TIMEOUT_SEC = int(os.getenv("PIPELINE_TIMEOUT_SEC", "1800"))   # 30 min default
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

app = FastAPI(
    title="Effortlessness Analyzer",
    version="1.1.0",
    description="Analyse facial effort/strain from a presenter video using FACS AUs + 5 scoring methods.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# In-memory job store. Swap for Redis/DB for multi-worker deployments.
jobs: Dict[str, dict] = {}
# Separate store for the per-frame visualization payload — keeps the polled
# /api/result/{id} response small and lets the UI fetch it on demand.
visualizations: Dict[str, dict] = {}

# Serialise the heavy torch model — py-feat is not safe under concurrency.
_pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
# Dedicated executor with a single worker — keeps the model graph on one
# thread for the duration of one run, then frees it for the next queued job.
_pipeline_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="pipeline")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    log.info("Warming py-feat detector …")
    try:
        await asyncio.get_event_loop().run_in_executor(None, warm_detector)
        log.info("py-feat detector ready.")
    except Exception as exc:                     # noqa: BLE001
        log.warning("Detector warm-up failed (will lazy-load on first request): %s", exc)


@app.on_event("shutdown")
async def _shutdown() -> None:
    _pipeline_executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_stage(job_id: str, stage: str, progress: float) -> None:
    job = jobs.get(job_id)
    if job is not None:
        job["stage"] = stage
        job["progress"] = round(progress, 3)
        log.info("[%s] %s (%.0f%%)", job_id[:8], stage, progress * 100)


def _pipeline_body(job_id: str, video_path: Path, filename: str) -> None:
    """The actual heavy work — runs on the pipeline executor thread."""
    warnings: list[str] = []
    started = time.time()
    _set_stage(job_id, "extracting frames", 0.05)
    frames, timestamps = extract_frames(str(video_path), target_fps=TARGET_FPS)
    if not frames:
        raise RuntimeError("No frames could be decoded from this video.")

    _set_stage(job_id, "detecting facial action units", 0.20)
    au_df = detect_aus(frames, fps=TARGET_FPS)

    _set_stage(job_id, "tracking face landmarks", 0.55)
    tracking_df = track_face(frames, fps=TARGET_FPS)

    del frames

    _set_stage(job_id, "calibrating baseline", 0.65)
    baseline = calibrate_baseline(au_df, tracking_df, fps=TARGET_FPS, duration_secs=30)
    if baseline.get("short_video"):
        warnings.append("Video shorter than 30s — baseline used a shorter window.")
    if baseline.get("scoreable_fraction", 1.0) < 0.5:
        warnings.append(
            f"Only {baseline['scoreable_fraction']*100:.0f}% of frames were scoreable — "
            "head turns or low face confidence may make the result unreliable."
        )

    _set_stage(job_id, "scoring · tension AUs", 0.70)
    m1 = score_tension_aus(au_df, baseline, tracking_df=tracking_df, fps=TARGET_FPS)
    _set_stage(job_id, "scoring · micro-expressions", 0.75)
    m2 = detect_micro_expressions(au_df, fps=TARGET_FPS, tracking_df=tracking_df)
    _set_stage(job_id, "scoring · asymmetry", 0.80)
    m3 = compute_asymmetry(au_df, baseline, tracking_df=tracking_df)
    _set_stage(job_id, "scoring · coherence", 0.85)
    m4 = compute_coherence(au_df, fps=TARGET_FPS, tracking_df=tracking_df)
    _set_stage(job_id, "scoring · drift & blink", 0.90)
    m5 = analyse_drift_blink(au_df, tracking_df, fps=TARGET_FPS, baseline=baseline)

    _set_stage(job_id, "aggregating", 0.95)
    ensemble = aggregate_scores(m1, m2, m3, m4, m5, timestamps)

    _set_stage(job_id, "generating insights", 0.98)
    insights = generate_insights(m1, m2, m3, m4, m5, ensemble, au_df)

    # Build the per-frame simulation payload. Stored separately so the
    # polled /api/result/{id} stays small.
    try:
        visualizations[job_id] = build_visualization(au_df, tracking_df, m1, TARGET_FPS)
    except Exception as exc:                       # noqa: BLE001
        log.warning("[%s] visualization build failed: %s", job_id[:8], exc)

    au_cols = [c for c in au_df.columns if c.startswith("AU")]
    result = {
        "job_id": job_id,
        "filename": filename,
        "duration_secs": round(len(au_df) / TARGET_FPS, 2),
        "frames_analysed": int(len(au_df)),
        "fps": TARGET_FPS,
        "overall_score": ensemble["overall_score"],
        "overall_label": ensemble["label"],
        "timeline": ensemble["timeline"],
        "method_scores": {
            "tension_aus": m1["summary"],
            "micro_expressions": m2["summary"],
            "asymmetry": m3["summary"],
            "coherence": m4["summary"],
            "drift_blink": m5["summary"],
        },
        "au_means": {k: float(v) for k, v in au_df[au_cols].mean().to_dict().items()},
        "au_stds": {k: float(v) for k, v in au_df[au_cols].std().fillna(0).to_dict().items()},
        "micro_expression_events": m2["events"],
        "blink_rate_per_min": m5["blink_rate_per_min"],
        "asymmetry_mean": m3["mean"],
        "insights": insights,
        "baseline": {
            "calibration_frames": baseline["calibration_frames"],
            "blink_rate_per_min": baseline["blink_rate_per_min"],
            "ear_threshold": baseline.get("ear_threshold"),
            "scoreable_fraction": baseline.get("scoreable_fraction"),
            "method": baseline.get("method"),
        },
        "warnings": warnings,
    }

    jobs[job_id].update({
        "status": "done",
        "progress": 1.0,
        "stage": "done",
        "result": result,
        "elapsed_secs": round(time.time() - started, 2),
    })
    log.info("[%s] pipeline finished in %.1fs", job_id[:8], time.time() - started)


async def _run_pipeline(job_id: str, video_path: Path, filename: str) -> None:
    """Serialise, run with timeout, and always clean the upload file."""
    try:
        async with _pipeline_semaphore:
            jobs[job_id]["status"] = "processing"
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                _pipeline_executor, _pipeline_body, job_id, video_path, filename
            )
            try:
                await asyncio.wait_for(future, timeout=PIPELINE_TIMEOUT_SEC)
            except (asyncio.TimeoutError, FuturesTimeoutError):
                # We cannot truly kill a thread in CPython; best-effort cancel.
                future.cancel()
                raise RuntimeError(
                    f"Pipeline exceeded PIPELINE_TIMEOUT_SEC={PIPELINE_TIMEOUT_SEC}s"
                )
    except Exception as exc:                     # noqa: BLE001
        log.error("[%s] pipeline failed:\n%s", job_id[:8], traceback.format_exc())
        jobs[job_id].update({
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "stage": "error",
        })
    finally:
        if not KEEP_UPLOADS:
            try:
                video_path.unlink(missing_ok=True)
            except Exception:                    # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "target_fps": TARGET_FPS,
        "jobs_in_memory": len(jobs),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "pipeline_timeout_sec": PIPELINE_TIMEOUT_SEC,
    }


@app.post("/api/analyse")
async def analyse_video(
    background: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    if not file.filename:
        raise HTTPException(400, "Missing filename.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}:
        raise HTTPException(400, f"Unsupported video format: {suffix}")

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"

    size = 0
    async with aiofiles.open(video_path, "wb") as fout:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                await fout.close()
                video_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB.")
            await fout.write(chunk)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "stage": "queued",
        "filename": file.filename,
        "size_bytes": size,
        "video_path": str(video_path) if KEEP_UPLOADS else None,
        "video_suffix": suffix,
    }

    # Schedule as a fire-and-forget task on the event loop. The semaphore
    # inside `_run_pipeline` is what actually serialises the heavy work.
    asyncio.create_task(_run_pipeline(job_id, video_path, file.filename))

    return {"job_id": job_id}


@app.get("/api/result/{job_id}")
def get_result(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id}")
    return job


@app.get("/api/visualization/{job_id}")
def get_visualization(job_id: str) -> dict:
    viz = visualizations.get(job_id)
    if viz is None:
        raise HTTPException(404, f"No visualization for job: {job_id}")
    return viz


_MIME_BY_SUFFIX = {
    ".mp4":  "video/mp4",
    ".mov":  "video/quicktime",
    ".m4v":  "video/mp4",
    ".webm": "video/webm",
    ".mkv":  "video/x-matroska",
    ".avi":  "video/x-msvideo",
}


@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    """Stream the original uploaded video if it was retained on disk."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id}")
    path_str = job.get("video_path")
    if not path_str:
        raise HTTPException(
            404,
            "Video preview unavailable — uploads were not retained for this job "
            "(set KEEP_UPLOADS=1 to enable).",
        )
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(404, "Video file no longer on disk.")
    suffix = job.get("video_suffix", "").lower()
    media = _MIME_BY_SUFFIX.get(suffix, "application/octet-stream")
    # FileResponse supports HTTP range requests out of the box — that's what
    # the <video> element needs for scrubbing/seeking.
    return FileResponse(path, media_type=media, filename=job.get("filename") or path.name)


@app.delete("/api/result/{job_id}")
def delete_result(job_id: str) -> dict:
    job = jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id}")
    visualizations.pop(job_id, None)
    # Best-effort cleanup of the retained upload.
    path_str = job.get("video_path")
    if path_str:
        try:
            Path(path_str).unlink(missing_ok=True)
        except Exception:                              # noqa: BLE001
            pass
    return {"deleted": True}
