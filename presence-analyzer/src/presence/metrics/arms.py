"""Rolling-window arm gesture statistics.

Vectorized:
- elbow angle per frame per arm: arccos(dot / |a||b|), where a = elbow→shoulder,
  b = elbow→wrist. Compiled via numpy ops over (T,) arrays.
- rolling 5s windows for variance / gesture count / amplitude
- scipy.signal.find_peaks for gesture frequency (cheap, called per arm)
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import polars as pl
from pydantic import BaseModel
from scipy.signal import find_peaks

from presence.pose.quality import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


class ArmsMetric(BaseModel):
    elbow_var_median: float       # median over windows, average over arms
    gesture_freq_per_min: float   # gestures/min averaged across arms
    gesture_amplitude_median: float  # max wrist excursion median over windows
    n_frames_used: int
    quality_score: float


def _elbow_angles(
    pose_df: pl.DataFrame, shoulder_idx: int, elbow_idx: int, wrist_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (frame_idx, angle_deg, wrist_x, wrist_y) sorted by frame_idx."""
    sub = pose_df.filter(pl.col("landmark_idx").is_in([shoulder_idx, elbow_idx, wrist_idx]))
    wide = sub.pivot(
        values=["x", "y"], index="frame_idx", on="landmark_idx", aggregate_function="first"
    ).sort("frame_idx")
    cols = wide.columns
    # Polars may use "x_11" or "x_landmark_idx_11" — handle both
    def col(prefix: str, idx: int) -> str:
        for c in cols:
            if c == f"{prefix}_{idx}" or c.endswith(f"_{idx}") and c.startswith(f"{prefix}_"):
                return c
        raise KeyError(f"missing {prefix}_{idx}")

    sx = wide.get_column(col("x", shoulder_idx)).to_numpy()
    sy = wide.get_column(col("y", shoulder_idx)).to_numpy()
    ex = wide.get_column(col("x", elbow_idx)).to_numpy()
    ey = wide.get_column(col("y", elbow_idx)).to_numpy()
    wx = wide.get_column(col("x", wrist_idx)).to_numpy()
    wy = wide.get_column(col("y", wrist_idx)).to_numpy()

    a_x = sx - ex
    a_y = sy - ey
    b_x = wx - ex
    b_y = wy - ey
    dot = a_x * b_x + a_y * b_y
    norm_a = np.sqrt(a_x * a_x + a_y * a_y)
    norm_b = np.sqrt(b_x * b_x + b_y * b_y)
    cos = np.clip(dot / (norm_a * norm_b + 1e-9), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))
    frame_idx = wide.get_column("frame_idx").to_numpy()
    return frame_idx, angle.astype(np.float32), wx.astype(np.float32), wy.astype(np.float32)


def _rolling_stats(
    angle: np.ndarray, wx: np.ndarray, wy: np.ndarray, window_frames: int, fps: float
) -> tuple[float, float, float, int]:
    """Return (var_median, gestures_per_min, amplitude_median, n_windows)."""
    n = len(angle)
    if n < window_frames:
        return float("nan"), float("nan"), float("nan"), 0
    # Stride view via np.lib.stride_tricks
    from numpy.lib.stride_tricks import sliding_window_view

    a_win = sliding_window_view(angle, window_frames)
    wx_win = sliding_window_view(wx, window_frames)
    wy_win = sliding_window_view(wy, window_frames)

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        a_var = np.nanvar(a_win, axis=1)
        amp = np.sqrt(
            (np.nanmax(wx_win, axis=1) - np.nanmin(wx_win, axis=1)) ** 2
            + (np.nanmax(wy_win, axis=1) - np.nanmin(wy_win, axis=1)) ** 2
        )

    # Gesture peaks: detect on entire elbow-angle series (lighter than per-window)
    # Then convert count to per-minute.
    valid = angle[~np.isnan(angle)]
    if len(valid) > 4:
        std = float(np.nanstd(angle))
        prom = max(std * 0.5, 1.0)  # 0.5 sigma or 1°
        peaks, _ = find_peaks(np.nan_to_num(angle, nan=float(np.nanmedian(angle))), prominence=prom, distance=int(max(1, fps * 0.4)))
        gestures = float(len(peaks))
    else:
        gestures = 0.0
    duration_min = (n / fps) / 60.0
    gestures_per_min = gestures / max(duration_min, 1e-6)

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        var_med = float(np.nanmedian(a_var)) if a_var.size else float("nan")
        amp_med = float(np.nanmedian(amp)) if amp.size else float("nan")
    return (var_med, float(gestures_per_min), amp_med, int(len(a_var)))


def compute_arms_metric(
    pose_df: pl.DataFrame,
    quality_df: pl.DataFrame,
    rolling_window_seconds: float,
    sample_fps: float,
) -> ArmsMetric:
    window_frames = max(2, int(round(rolling_window_seconds * sample_fps)))

    left_frames, left_angles, lwx, lwy = _elbow_angles(
        pose_df, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
    )
    right_frames, right_angles, rwx, rwy = _elbow_angles(
        pose_df, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
    )

    # Mask out frames where the arm is not visible
    quality_arr = quality_df.sort("frame_idx").select(
        ["frame_idx", "left_arm_visible", "right_arm_visible"]
    )
    q_idx = quality_arr.get_column("frame_idx").to_numpy()
    q_left = quality_arr.get_column("left_arm_visible").to_numpy()
    q_right = quality_arr.get_column("right_arm_visible").to_numpy()
    qmap_left = dict(zip(q_idx.tolist(), q_left.tolist()))
    qmap_right = dict(zip(q_idx.tolist(), q_right.tolist()))
    mask_l = np.array([qmap_left.get(int(f), False) for f in left_frames])
    mask_r = np.array([qmap_right.get(int(f), False) for f in right_frames])
    left_angles = np.where(mask_l, left_angles, np.nan)
    right_angles = np.where(mask_r, right_angles, np.nan)

    lv, lg, la, nl = _rolling_stats(left_angles, lwx, lwy, window_frames, sample_fps)
    rv, rg, ra, nr = _rolling_stats(right_angles, rwx, rwy, window_frames, sample_fps)

    def _avg(a: float, b: float) -> float:
        vals = [v for v in (a, b) if v == v]  # filter NaN
        return float(np.mean(vals)) if vals else float("nan")

    n_used = int(np.nansum(mask_l) + np.nansum(mask_r))
    n_total = 2 * quality_df.height
    quality = n_used / n_total if n_total > 0 else 0.0

    return ArmsMetric(
        elbow_var_median=_avg(lv, rv),
        gesture_freq_per_min=_avg(lg, rg),
        gesture_amplitude_median=_avg(la, ra),
        n_frames_used=n_used,
        quality_score=float(quality),
    )
