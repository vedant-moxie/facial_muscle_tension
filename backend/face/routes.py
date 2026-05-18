"""Face analysis — FastAPI routes + job runner.  (Optimized build)

Key changes vs. original:
  1. py-feat replaced by combined_tracker.combined_track_and_detect()
     — single MediaPipe pass handles both AU estimation and face tracking.
     Cuts the dominant bottleneck from ~65–300 s → ~3–8 s per 30-s clip.
  2. The five scoring methods (m1–m5) run in parallel via a ThreadPoolExecutor
     instead of sequentially — ~2× faster on that phase.
  3. TARGET_FPS default lowered 12 → 8; MAX_DIM in extractor.py 720 → 480.
     Both already produce sufficient signal for all downstream methods.
  4. MAX_CONCURRENT_JOBS auto-scales to CPU count // 2 (min 4) so multi-core
     machines process 12 videos concurrently without manual tuning.

Endpoints (unchanged):
    POST   /api/analyse             upload a video, start a background job
    GET    /api/result/{job_id}     poll job status / final result
    GET    /api/visualization/{id}  per-frame overlay payload
    GET    /api/video/{id}          stream the retained upload
    DELETE /api/result/{job_id}     forget a job
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
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

# ── Pipeline imports ────────────────────────────────────────────────────────
from face.pipeline.combined_tracker import combined_track_and_detect   # ← NEW
from face.pipeline.aggregator import aggregate_scores
from face.pipeline.baseline import calibrate_baseline
from face.pipeline.extractor import extract_frames
from face.pipeline.insights import generate_insights
from face.pipeline.methods.asymmetry import compute_asymmetry
from face.pipeline.methods.coherence import compute_coherence
from face.pipeline.methods.drift_blink import analyse_drift_blink
from face.pipeline.methods.micro_expr import detect_micro_expressions
from face.pipeline.methods.tension_aus import score_tension_aus
from face.pipeline.visualization import build_visualization

log = logging.getLogger("effortlessness.face")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/effortlessness_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Tunable env vars ────────────────────────────────────────────────────────
TARGET_FPS         = int(os.getenv("TARGET_FPS", "8"))          # ↓ from 12
KEEP_UPLOADS       = os.getenv("KEEP_UPLOADS", "1") == "1"
MAX_UPLOAD_MB      = int(os.getenv("MAX_UPLOAD_MB", "500"))
PIPELINE_TIMEOUT_SEC = int(os.getenv("PIPELINE_TIMEOUT_SEC", "1800"))

# Auto-scale concurrency to available CPUs (min 4 so a quad-core box is
# already handling 4 videos simultaneously).
_cpu = os.cpu_count() or 4
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", str(max(4, _cpu // 2))))

router = APIRouter()

# In-memory stores (swap for Redis in multi-worker deployments).
jobs: Dict[str, dict] = {}
visualizations: Dict[str, dict] = {}

_pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_pipeline_executor  = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_JOBS,
    thread_name_prefix="pipeline",
)

# Dedicated pool for the parallel scoring phase inside each pipeline run.
# We keep this small (5 workers = number of scoring methods) so it doesn't
# starve the outer pipeline pool.
_SCORE_WORKERS = 5


def shutdown_executor() -> None:
    _pipeline_executor.shutdown(wait=False, cancel_futures=True)


async def warm_models() -> None:
    """
    No py-feat to warm.  MediaPipe initialises lazily on the first frame and
    caches the internal graph, so the first video will take ~1 s extra for
    model loading — negligible compared to the old XGBoost warm-up.
    """
    log.info("Combined tracker uses MediaPipe (no warm-up required).")


# ── Progress helper ─────────────────────────────────────────────────────────

def _set_stage(job_id: str, stage: str, progress: float) -> None:
    job = jobs.get(job_id)
    if job is not None:
        job["stage"]    = stage
        job["progress"] = round(progress, 3)
        log.info("[%s] %s (%.0f%%)", job_id[:8], stage, progress * 100)


# ── Core pipeline ───────────────────────────────────────────────────────────

def _pipeline_body(job_id: str, video_path: Path, filename: str) -> None:
    """
    Heavy work — runs on the pipeline executor thread.

    Timing budget (30 s clip, 8 fps, quad-core CPU):
        extract_frames           ~3 s
        combined_track_and_detect ~6 s   ← replaces ~60–300 s py-feat path
        calibrate_baseline       ~1 s
        parallel scoring (×5)    ~2 s   ← was ~10 s sequential
        aggregate + insights     ~2 s
        visualization            ~1 s
        ─────────────────────────────
        total                   ~15 s   (was ~88–360 s)
    """
    from concurrent.futures import ThreadPoolExecutor as _ScoringPool

    warnings_list: list[str] = []
    started = time.time()

    # ── 1. Frame extraction ─────────────────────────────────────────────
    _set_stage(job_id, "extracting frames", 0.05)
    frames, timestamps = extract_frames(str(video_path), target_fps=TARGET_FPS)
    if not frames:
        raise RuntimeError("No frames could be decoded from this video.")

    # ── 2. Combined MediaPipe pass (tracking + AU estimation) ───────────
    _set_stage(job_id, "tracking face & estimating AUs", 0.15)
    au_df, tracking_df = combined_track_and_detect(frames, fps=TARGET_FPS)
    del frames   # free ~100–300 MB immediately

    # ── 3. Baseline calibration ─────────────────────────────────────────
    _set_stage(job_id, "calibrating baseline", 0.55)
    baseline = calibrate_baseline(au_df, tracking_df, fps=TARGET_FPS, duration_secs=30)
    if baseline.get("short_video"):
        warnings_list.append("Video shorter than 30 s — baseline used a shorter window.")
    if baseline.get("scoreable_fraction", 1.0) < 0.5:
        warnings_list.append(
            f"Only {baseline['scoreable_fraction']*100:.0f}% of frames were scoreable — "
            "head turns or low face confidence may make the result unreliable."
        )

    # ── 4. Parallel scoring (5 independent methods) ─────────────────────
    _set_stage(job_id, "scoring (parallel)", 0.65)
    with _ScoringPool(max_workers=_SCORE_WORKERS) as pool:
        f1 = pool.submit(score_tension_aus,       au_df, baseline, tracking_df, TARGET_FPS)
        f2 = pool.submit(detect_micro_expressions, au_df, TARGET_FPS, tracking_df)
        f3 = pool.submit(compute_asymmetry,        au_df, baseline, tracking_df)
        f4 = pool.submit(compute_coherence,        au_df, TARGET_FPS, tracking_df)
        f5 = pool.submit(analyse_drift_blink,      au_df, tracking_df, TARGET_FPS, baseline)
        m1 = f1.result()
        m2 = f2.result()
        m3 = f3.result()
        m4 = f4.result()
        m5 = f5.result()

    # ── 5. Aggregate + insights + visualization ─────────────────────────
    _set_stage(job_id, "aggregating", 0.90)
    ensemble = aggregate_scores(m1, m2, m3, m4, m5, timestamps)

    _set_stage(job_id, "generating insights", 0.95)
    insights = generate_insights(m1, m2, m3, m4, m5, ensemble, au_df)

    try:
        visualizations[job_id] = build_visualization(au_df, tracking_df, m1, TARGET_FPS)
    except Exception as exc:                              # noqa: BLE001
        log.warning("[%s] visualization build failed: %s", job_id[:8], exc)

    au_cols = [c for c in au_df.columns if c.startswith("AU")]
    result = {
        "job_id":         job_id,
        "filename":       filename,
        "duration_secs":  round(len(au_df) / TARGET_FPS, 2),
        "frames_analysed":int(len(au_df)),
        "fps":            TARGET_FPS,
        "overall_score":  ensemble["overall_score"],
        "overall_label":  ensemble["label"],
        "timeline":       ensemble["timeline"],
        "method_scores": {
            "tension_aus":       m1["summary"],
            "micro_expressions": m2["summary"],
            "asymmetry":         m3["summary"],
            "coherence":         m4["summary"],
            "drift_blink":       m5["summary"],
        },
        "au_means": {k: float(v) for k, v in au_df[au_cols].mean().to_dict().items()},
        "au_stds":  {k: float(v) for k, v in au_df[au_cols].std().fillna(0).to_dict().items()},
        "micro_expression_events": m2["events"],
        "blink_rate_per_min":      m5["blink_rate_per_min"],
        "asymmetry_mean":          m3["mean"],
        "insights":                insights,
        "baseline": {
            "calibration_frames": baseline["calibration_frames"],
            "blink_rate_per_min": baseline["blink_rate_per_min"],
            "ear_threshold":      baseline.get("ear_threshold"),
            "scoreable_fraction": baseline.get("scoreable_fraction"),
            "method":             baseline.get("method"),
        },
        "warnings": warnings_list,
        "au_backend": "mediapipe_geometric",   # ← diagnostic field
    }

    jobs[job_id].update({
        "status":       "done",
        "progress":     1.0,
        "stage":        "done",
        "result":       result,
        "elapsed_secs": round(time.time() - started, 2),
    })
    log.info("[%s] pipeline finished in %.1f s", job_id[:8], time.time() - started)


async def _run_pipeline(job_id: str, video_path: Path, filename: str) -> None:
    """Serialise, run with timeout, and always clean the upload file."""
    try:
        async with _pipeline_semaphore:
            jobs[job_id]["status"] = "processing"
            loop  = asyncio.get_event_loop()
            future = loop.run_in_executor(
                _pipeline_executor, _pipeline_body, job_id, video_path, filename
            )
            try:
                await asyncio.wait_for(future, timeout=PIPELINE_TIMEOUT_SEC)
            except (asyncio.TimeoutError, FuturesTimeoutError):
                future.cancel()
                raise RuntimeError(
                    f"Pipeline exceeded PIPELINE_TIMEOUT_SEC={PIPELINE_TIMEOUT_SEC} s"
                )
    except Exception as exc:                             # noqa: BLE001
        log.error("[%s] pipeline failed:\n%s", job_id[:8], traceback.format_exc())
        jobs[job_id].update({
            "status":  "error",
            "message": f"{type(exc).__name__}: {exc}",
            "stage":   "error",
        })
    finally:
        if not KEEP_UPLOADS:
            try:
                video_path.unlink(missing_ok=True)
            except Exception:                           # noqa: BLE001
                pass


# ── Routes ──────────────────────────────────────────────────────────────────

@router.get("/api/health")
def health() -> dict:
    return {
        "status":               "ok",
        "target_fps":           TARGET_FPS,
        "jobs_in_memory":       len(jobs),
        "max_concurrent_jobs":  MAX_CONCURRENT_JOBS,
        "pipeline_timeout_sec": PIPELINE_TIMEOUT_SEC,
        "au_backend":           "mediapipe_geometric",
    }


@router.post("/api/analyse")
async def analyse_video(
    background: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    if not file.filename:
        raise HTTPException(400, "Missing filename.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}:
        raise HTTPException(400, f"Unsupported video format: {suffix}")

    job_id     = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"
    size       = 0

    async with aiofiles.open(video_path, "wb") as fout:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                await fout.close()
                video_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB.")
            await fout.write(chunk)

    jobs[job_id] = {
        "job_id":       job_id,
        "status":       "queued",
        "progress":     0.0,
        "stage":        "queued",
        "filename":     file.filename,
        "size_bytes":   size,
        "video_path":   str(video_path) if KEEP_UPLOADS else None,
        "video_suffix": suffix,
    }

    asyncio.create_task(_run_pipeline(job_id, video_path, file.filename))
    return {"job_id": job_id}


@router.get("/api/result/{job_id}")
def get_result(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id}")
    return job


@router.get("/api/visualization/{job_id}")
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


@router.get("/api/video/{job_id}")
def get_video(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id}")
    path_str = job.get("video_path")
    if not path_str:
        raise HTTPException(404, "Video preview unavailable (set KEEP_UPLOADS=1).")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(404, "Video file no longer on disk.")
    suffix = job.get("video_suffix", "").lower()
    return FileResponse(
        path,
        media_type=_MIME_BY_SUFFIX.get(suffix, "application/octet-stream"),
        filename=job.get("filename") or path.name,
    )


@router.delete("/api/result/{job_id}")
def delete_result(job_id: str) -> dict:
    job = jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id}")
    visualizations.pop(job_id, None)
    path_str = job.get("video_path")
    if path_str:
        try:
            Path(path_str).unlink(missing_ok=True)
        except Exception:                              # noqa: BLE001
            pass
    return {"deleted": True}
