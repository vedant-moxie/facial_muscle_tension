"""SPARC (spectral arc length) smoothness metric, JIT-compiled with numba.

Reference: Balasubramanian et al. 2015, "On the analysis of movement smoothness."
More negative = less smooth. Applied per landmark (nose, left wrist, right wrist),
then averaged.
"""
from __future__ import annotations

import math

import numba
import numpy as np
import polars as pl
from pydantic import BaseModel

from presence.pose.quality import LEFT_WRIST, NOSE, RIGHT_WRIST


class FluidityMetric(BaseModel):
    sparc_nose: float
    sparc_wrist_mean: float
    sparc_overall: float
    n_frames_used: int
    quality_score: float


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@numba.njit(cache=True, fastmath=True)
def _spectrum_arc_length(
    mag: np.ndarray, freqs: np.ndarray, fc: float, amp_th: float
) -> float:
    n_freq = mag.shape[0]
    # Cutoff index
    cut = n_freq
    for i in range(n_freq):
        if freqs[i] > fc:
            cut = i
            break
    # Tighten cutoff to the last index with magnitude above the amplitude threshold
    last_above = 0
    for i in range(cut):
        if mag[i] >= amp_th:
            last_above = i
    cut = min(cut, last_above + 1)
    if cut < 3:
        return float("nan")
    arc = 0.0
    inv_fc = 1.0 / max(fc, 1e-9)
    for i in range(1, cut):
        df = (freqs[i] - freqs[i - 1]) * inv_fc
        dm = mag[i] - mag[i - 1]
        arc += math.sqrt(df * df + dm * dm)
    return -arc


def sparc(
    movement: np.ndarray,
    fs: float,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> float:
    """Compute SPARC of a speed profile (Balasubramanian 2015).

    FFT is delegated to numpy (C implementation); the inner arc-length scan
    is numba-compiled.
    """
    n = int(movement.shape[0])
    if n < 4:
        return float("nan")
    pad_n = _next_pow2(n) * int(padlevel)
    padded = np.zeros(pad_n, dtype=np.float64)
    padded[:n] = movement
    spectrum = np.fft.fft(padded)
    n_freq = pad_n // 2
    mag = np.abs(spectrum[:n_freq])
    mag_max = float(mag.max())
    if mag_max <= 0.0:
        return float("nan")
    mag = (mag / mag_max).astype(np.float64)
    freqs = ((fs / 2.0) * np.arange(n_freq) / max(1, n_freq - 1)).astype(np.float64)
    return float(_spectrum_arc_length(mag, freqs, float(fc), float(amp_th)))


def _speed_profile(
    pose_df: pl.DataFrame, landmark_idx: int, min_visibility: float = 0.5
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (timestamps, speed, visible_fraction) for a single landmark.

    Frames where visibility < min_visibility are dropped (NOT forward-filled).
    Forward-filling positions across long invisibility gaps creates phantom
    motion spikes that destroy SPARC. We instead compute speeds only across
    contiguous visible segments; invisible gaps contribute no speed samples.
    """
    sub = (
        pose_df.filter(pl.col("landmark_idx") == landmark_idx)
        .sort("frame_idx")
        .select(["timestamp", "x", "y", "visibility"])
    )
    if sub.height < 4:
        return np.empty(0), np.empty(0), 0.0
    ts = sub.get_column("timestamp").to_numpy().astype(np.float64)
    x = sub.get_column("x").to_numpy().astype(np.float64)
    y = sub.get_column("y").to_numpy().astype(np.float64)
    vis = sub.get_column("visibility").to_numpy().astype(np.float64)

    good = (vis >= min_visibility) & ~np.isnan(x) & ~np.isnan(y)
    visible_fraction = float(good.mean()) if good.size else 0.0

    # Speeds are only valid between two consecutive visible frames.
    pair_good = good[:-1] & good[1:]
    dt = np.diff(ts)
    dt[dt <= 0] = 1e-6
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    speed = np.sqrt(vx * vx + vy * vy)
    ts_speed = ts[1:][pair_good]
    speed = speed[pair_good]
    return ts_speed.astype(np.float64), speed.astype(np.float64), visible_fraction


def compute_fluidity_metric(
    pose_df: pl.DataFrame,
    quality_df: pl.DataFrame,
    sample_fps: float,
    min_visibility: float = 0.5,
    min_coverage: float = 0.30,
) -> FluidityMetric:
    """SPARC of visible-only speed profiles.

    A landmark contributes to the score only when its visible fraction
    exceeds min_coverage (default 30%); otherwise its SPARC is left NaN and
    excluded from the mean. Quality is the MAX visible fraction across the
    three landmarks — a video that shows the face but no wrists still gets
    a meaningful fluidity score from nose motion.
    """
    ts_nose, speed_nose, vis_nose = _speed_profile(pose_df, NOSE, min_visibility)
    ts_lw, speed_lw, vis_lw = _speed_profile(pose_df, LEFT_WRIST, min_visibility)
    ts_rw, speed_rw, vis_rw = _speed_profile(pose_df, RIGHT_WRIST, min_visibility)

    def _safe_sparc(arr: np.ndarray, visible_fraction: float) -> float:
        if len(arr) < 8 or visible_fraction < min_coverage:
            return float("nan")
        return float(sparc(arr, fs=float(sample_fps)))

    s_nose = _safe_sparc(speed_nose, vis_nose)
    s_lw = _safe_sparc(speed_lw, vis_lw)
    s_rw = _safe_sparc(speed_rw, vis_rw)

    wrists = [v for v in (s_lw, s_rw) if v == v]
    s_wrist_mean = float(np.mean(wrists)) if wrists else float("nan")
    overall_vals = [v for v in (s_nose, s_wrist_mean) if v == v]
    s_overall = float(np.mean(overall_vals)) if overall_vals else float("nan")

    # Quality = best-tracked landmark. If we got *anything* clean, the metric
    # is meaningful — partial tracking shouldn't penalise the score to 0.
    visible_fractions = [vis_nose, vis_lw, vis_rw]
    contributing = [
        vis for vis, s in zip(visible_fractions, (s_nose, s_lw, s_rw)) if s == s
    ]
    quality = float(max(contributing)) if contributing else 0.0
    n_used = int(len(speed_nose) + len(speed_lw) + len(speed_rw))

    return FluidityMetric(
        sparc_nose=s_nose,
        sparc_wrist_mean=s_wrist_mean,
        sparc_overall=s_overall,
        n_frames_used=n_used,
        quality_score=quality,
    )
