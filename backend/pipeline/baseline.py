"""
Per-person calibration.

Previously this module took the first 30 s of the video as "neutral" and
averaged AU intensities over that window. That assumption was wrong in
practice — speakers introduce themselves with emphasis, not stillness.

OpenFace's published trick is what we use now: per-AU **25th percentile
across the whole (scoreable) video** is the person-neutral baseline. This
is robust to busy openings, head turns, and short videos, and matches the
person-calibration approach in OpenFace's dynamic models.

We also derive a per-person adaptive **EAR blink threshold** here so that
downstream blink detection in `methods/drift_blink.py` is no longer
hard-coded at 0.21 (which fails for glasses-wearers, epicanthic folds,
and anyone whose neutral eye aperture differs from the average).

Output dict keys (consumed downstream):

    au_mean[au]                 baseline AU intensity        (25th-percentile)
    au_std[au]                  robust spread (MAD * 1.4826) (≥ 0.05 floor)
    blink_rate_per_min          baseline blink rate          (from calib window)
    ear_threshold               adaptive blink threshold     (per-person)
    ear_mean / ear_std          EAR statistics during quiet frames
    scoreable_fraction          fraction of frames usable
    calibration_frames          how many frames informed AU stats
    short_video (bool)          true if video < duration_secs
    method                      "percentile" (new) | "window" (legacy)
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Percentile for neutral AU intensity. 25th-percentile is OpenFace's choice;
# robust against typical speaker openings, doesn't overfit to a single calm
# moment.
NEUTRAL_PERCENTILE = 25.0

# EAR threshold floor — even with extremely wide neutral eyes we don't want a
# threshold below 0.15 (that would catch nothing).
MIN_EAR_THRESHOLD = 0.15
MAX_EAR_THRESHOLD = 0.27
# How many EAR-std below mean counts as "eyes closing". Literature value ≈ 2.
EAR_K = 2.0


def _mad_scale(series: pd.Series) -> float:
    """1.4826 · MAD ≈ robust std; floors to 0.05 to avoid division blow-ups."""
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return 0.1
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return max(0.05, float(1.4826 * mad))


def _derive_blink_events(
    avg_ear: np.ndarray,
    threshold: float,
    fps: int,
    min_gap_frames: int,
) -> List[int]:
    """Onset-only blink event indices: first frame where EAR drops below
    `threshold`, with a refractory gap so the same blink doesn't double-count."""
    below = avg_ear < threshold
    events: List[int] = []
    last = -min_gap_frames
    in_blink = False
    for i, b in enumerate(below):
        if b and not in_blink and (i - last) >= min_gap_frames:
            events.append(i)
            last = i
            in_blink = True
        elif not b:
            in_blink = False
    return events


def derive_blink_events_from_threshold(
    tracking_df: pd.DataFrame,
    threshold: float,
    fps: int,
) -> List[int]:
    """Public helper used by `methods/drift_blink.py` to recompute blink onsets
    against the adaptive per-person threshold."""
    if "avg_ear" not in tracking_df.columns:
        return []
    avg_ear = tracking_df["avg_ear"].to_numpy(dtype=float)
    min_gap = max(1, int(0.1 * fps))
    return _derive_blink_events(avg_ear, threshold, fps, min_gap)


def calibrate_baseline(
    au_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    fps: int = 12,
    duration_secs: int = 30,
) -> Dict:
    n = min(len(au_df), len(tracking_df))
    target_frames = duration_secs * fps
    short_video = n < target_frames

    # ---- scoreable mask ----
    if "scoreable" in tracking_df.columns:
        score_mask = tracking_df["scoreable"].astype(bool).to_numpy()[:n]
    elif "face_present" in tracking_df.columns:
        score_mask = tracking_df["face_present"].astype(bool).to_numpy()[:n]
    else:
        score_mask = np.ones(n, dtype=bool)

    scoreable_fraction = float(score_mask.mean()) if n else 0.0

    au_cols = [c for c in au_df.columns if c.startswith("AU")]

    # ---- AU baseline: 25th-percentile across all scoreable frames ----
    au_view_full = au_df.iloc[:n][au_cols]
    au_view = au_view_full.loc[score_mask] if score_mask.any() else au_view_full.iloc[:0]
    method = "percentile"

    if au_view.empty:
        # Fall back to first-window mean if we have nothing scoreable.
        log.warning("No scoreable frames — falling back to first-window mean baseline.")
        au_view = au_df.iloc[:min(n, target_frames)][au_cols]
        au_neutral = au_view.mean()
        method = "window"
    else:
        au_neutral = au_view.quantile(NEUTRAL_PERCENTILE / 100.0)

    au_neutral = au_neutral.fillna(0.0)
    au_spread = pd.Series({c: _mad_scale(au_view[c]) for c in au_cols})

    # ---- adaptive EAR threshold ----
    ear_thresh = (MIN_EAR_THRESHOLD + MAX_EAR_THRESHOLD) / 2
    ear_mean = 0.30
    ear_std = 0.03
    if "avg_ear" in tracking_df.columns:
        ear_full = tracking_df["avg_ear"].iloc[:n].to_numpy(dtype=float)
        # Use frames where the face is present — but include blinks naturally
        # so the distribution spans both open and closed states.
        if score_mask.any():
            ear_pop = ear_full[score_mask]
        else:
            ear_pop = ear_full
        if ear_pop.size >= max(fps, 6):
            # Robust upper-mode of the distribution = "eyes open". Use top half.
            top_half = ear_pop[ear_pop >= np.median(ear_pop)]
            ear_mean = float(np.median(top_half))
            ear_std = float(np.std(top_half) or 0.02)
            ear_thresh = float(np.clip(
                ear_mean - EAR_K * ear_std,
                MIN_EAR_THRESHOLD,
                MAX_EAR_THRESHOLD,
            ))

    # ---- blink rate during the (first-N) calibration window for legacy reporting ----
    calib_frames = max(1, target_frames if not short_video else max(1, n // 4))
    avg_ear_calib = (
        tracking_df["avg_ear"].iloc[:calib_frames].to_numpy(dtype=float)
        if "avg_ear" in tracking_df.columns
        else np.array([], dtype=float)
    )
    min_gap = max(1, int(0.1 * fps))
    blink_events = _derive_blink_events(avg_ear_calib, ear_thresh, fps, min_gap)
    blink_rate = (len(blink_events) / (calib_frames / fps)) * 60 if calib_frames else 0.0

    log.info(
        "Baseline: method=%s, AU n=%d (scoreable=%d), EAR thresh=%.3f, blink_rate=%.1f/min",
        method, n, int(score_mask.sum()), ear_thresh, blink_rate,
    )

    return {
        "au_mean": {k: float(v) for k, v in au_neutral.items()},
        "au_std": {k: float(v) for k, v in au_spread.items()},
        "blink_rate_per_min": float(blink_rate),
        "ear_threshold": float(ear_thresh),
        "ear_mean": float(ear_mean),
        "ear_std": float(ear_std),
        "left_ear_mean": (
            float(tracking_df["left_ear"].iloc[:n].mean())
            if "left_ear" in tracking_df.columns else 0.30
        ),
        "right_ear_mean": (
            float(tracking_df["right_ear"].iloc[:n].mean())
            if "right_ear" in tracking_df.columns else 0.30
        ),
        "yaw_mean": (
            float(tracking_df["yaw"].iloc[:n].mean())
            if "yaw" in tracking_df.columns else 0.0
        ),
        "scoreable_fraction": scoreable_fraction,
        "calibration_frames": int(calib_frames),
        "short_video": bool(short_video),
        "method": method,
    }
