"""Pipeline orchestrator — the one entry point that wires every stage.

Every stage records its duration → `AnalysisResult.timings` for perf debugging.
"""
from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from presence.config import Config
from presence.filters.one_euro import one_euro_filter
from presence.metrics.arms import compute_arms_metric
from presence.metrics.fluidity import compute_fluidity_metric
from presence.metrics.head import compute_head_metric
from presence.metrics.shoulder import compute_shoulder_metric
from presence.pose.extractor import N_POSE_LANDMARKS, PoseExtractor
from presence.pose.quality import compute_quality
from presence.scoring.aggregate import VideoFeatures, build_features
from presence.scoring.model import PresenceScore, score_features

logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_path: Path
    content_hash: str
    features: VideoFeatures
    score: PresenceScore
    timings: dict[str, float]
    warnings: list[str] = []


def _filter_pose_df(pose_df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Apply One Euro filter to (x, y, z) per landmark vectorized across all 33."""
    n_lm = N_POSE_LANDMARKS
    # Build (T, 99) array from per-frame group; preserve original frame order
    sorted_df = pose_df.sort(["frame_idx", "landmark_idx"])
    frame_ids = sorted_df.get_column("frame_idx").unique(maintain_order=True).to_list()
    if not frame_ids:
        return pose_df
    T = len(frame_ids)
    coords = np.full((T, n_lm * 3), np.nan, dtype=np.float32)
    timestamps = np.empty(T, dtype=np.float32)

    # Pivot via numpy: group rows of 33 per frame
    arr = sorted_df.select(["frame_idx", "landmark_idx", "x", "y", "z", "timestamp"]).to_numpy()
    frame_to_row = {fid: i for i, fid in enumerate(frame_ids)}
    for row in arr:
        fid = int(row[0])
        lm = int(row[1])
        i = frame_to_row[fid]
        base = lm * 3
        coords[i, base] = row[2]
        coords[i, base + 1] = row[3]
        coords[i, base + 2] = row[4]
        timestamps[i] = row[5]

    # Replace nan with column-wise interpolated values (forward-fill within column)
    if np.isnan(coords).any():
        for c in range(coords.shape[1]):
            col = coords[:, c]
            nan_mask = np.isnan(col)
            if nan_mask.all():
                col[:] = 0.0
                continue
            idx = np.arange(len(col))
            good = ~nan_mask
            col[nan_mask] = np.interp(idx[nan_mask], idx[good], col[good])
            coords[:, c] = col

    filtered = one_euro_filter(
        coords,
        timestamps,
        float(config.filters.one_euro_min_cutoff),
        float(config.filters.one_euro_beta),
    )

    # Rebuild a DataFrame with the same schema as the input
    # Vectorized reshape: (T, 33, 3) → flat columns
    reshaped = filtered.reshape(T, n_lm, 3)
    n_total = T * n_lm
    frame_idx_arr = np.repeat(np.array(frame_ids, dtype=np.int32), n_lm)
    lm_idx_arr = np.tile(np.arange(n_lm, dtype=np.int8), T)
    ts_arr = np.repeat(timestamps, n_lm)
    x_arr = reshaped[:, :, 0].reshape(-1)
    y_arr = reshaped[:, :, 1].reshape(-1)
    z_arr = reshaped[:, :, 2].reshape(-1)
    # Keep visibility from the source (filter doesn't change it)
    vis_lookup = (
        sorted_df.select(["frame_idx", "landmark_idx", "visibility"])
        .to_numpy()
    )
    vis_map = {(int(r[0]), int(r[1])): float(r[2]) for r in vis_lookup}
    vis_arr = np.array(
        [vis_map.get((int(fi), int(li)), 0.0) for fi, li in zip(frame_idx_arr, lm_idx_arr)],
        dtype=np.float32,
    )
    return pl.DataFrame(
        {
            "frame_idx": frame_idx_arr,
            "timestamp": ts_arr,
            "landmark_idx": lm_idx_arr,
            "x": x_arr,
            "y": y_arr,
            "z": z_arr,
            "visibility": vis_arr,
        }
    )


def analyze_video(video_path: Path, config: Config) -> AnalysisResult:
    """End-to-end analysis: pose → filter → metrics → score."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    timings: dict[str, float] = {}
    warnings: list[str] = []

    extractor = PoseExtractor(config)
    try:
        t0 = time.perf_counter()
        pose_df, face_df, info = extractor.extract(video_path, with_face=True)
        timings["pose_extract"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filtered_pose = _filter_pose_df(pose_df, config)
        timings["filter"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        quality_df = compute_quality(filtered_pose, config.pose.min_visibility)
        timings["quality"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        shoulder = compute_shoulder_metric(
            filtered_pose, quality_df, config.quality.max_camera_roll_deg
        )
        timings["metric_shoulder"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        head = compute_head_metric(
            face_df, quality_df, info["width"], info["height"],
            total_frames=info["n_frames"],
        )
        timings["metric_head"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        arms = compute_arms_metric(
            filtered_pose,
            quality_df,
            config.metrics.rolling_window_seconds,
            config.pose.sample_fps,
        )
        timings["metric_arms"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        fluidity = compute_fluidity_metric(
            filtered_pose, quality_df, config.pose.sample_fps
        )
        timings["metric_fluidity"] = time.perf_counter() - t0

        usable_fraction = quality_df.filter(pl.col("frame_usable")).height / max(
            quality_df.height, 1
        )
        if usable_fraction < config.quality.min_usable_frame_fraction:
            warnings.append(
                f"Only {usable_fraction:.0%} of frames detected a person "
                f"(below {config.quality.min_usable_frame_fraction:.0%} threshold)"
            )
        # Per-metric quality warnings so the UI can tell shoulder-cropped from
        # bad-lighting from no-person-at-all.
        if shoulder.quality_score < 0.3:
            warnings.append("Shoulders rarely in frame — shoulder symmetry uncertain.")
        if arms.quality_score < 0.3:
            warnings.append("Arms rarely in frame — gesture metrics uncertain.")
        if fluidity.quality_score < 0.3 and head.quality_score < 0.3:
            warnings.append("Very few trackable landmarks — fluidity uncertain.")

        t0 = time.perf_counter()
        features = build_features(
            shoulder, head, arms, fluidity, info["duration_sec"], info["fps_effective"]
        )
        score = score_features(features, config.scoring)
        timings["score"] = time.perf_counter() - t0
    finally:
        extractor.close()

    return AnalysisResult(
        video_path=video_path,
        content_hash=info["content_hash"],
        features=features,
        score=score,
        timings=timings,
        warnings=warnings,
    )


def _analyze_one_safe(video_path: Path, config: Config) -> dict:
    """Worker entry point — returns a flat dict so polars can build a table."""
    try:
        result = analyze_video(video_path, config)
        row = {
            "video_path": str(result.video_path),
            "content_hash": result.content_hash,
            "score": result.score.score,
            "confidence_low": result.score.confidence_low,
            "confidence_high": result.score.confidence_high,
            "is_calibrated": result.score.is_calibrated,
            "quality_overall": result.score.quality_overall,
            "shoulder_component": result.score.components["shoulder"],
            "head_component": result.score.components["head"],
            "arms_component": result.score.components["arms"],
            "fluidity_component": result.score.components["fluidity"],
            "duration_sec": result.features.duration_sec,
            "total_time": float(sum(result.timings.values())),
            "error": None,
            **result.features.model_dump(),
        }
        return row
    except Exception as e:
        return {
            "video_path": str(video_path),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=5),
        }


def batch_analyze(directory: Path, output: Path, config: Config) -> tuple[int, int]:
    """Parallel batch with joblib (loky backend). Returns (n_ok, n_err)."""
    from joblib import Parallel, delayed

    videos = sorted(
        list(directory.rglob("*.mp4"))
        + list(directory.rglob("*.mov"))
        + list(directory.rglob("*.webm"))
    )
    if not videos:
        output.write_bytes(b"")
        return 0, 0

    n_workers = max(1, int(config.runtime.num_workers))
    results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_analyze_one_safe)(v, config) for v in videos
    )

    ok_rows = [r for r in results if r.get("error") is None]
    err_rows = [r for r in results if r.get("error") is not None]

    if ok_rows:
        df = pl.from_dicts(ok_rows)
        df.write_parquet(output, compression="zstd")
    if err_rows:
        err_path = output.with_name(output.stem + "_errors.parquet")
        pl.from_dicts(err_rows).write_parquet(err_path, compression="zstd")

    return len(ok_rows), len(err_rows)
