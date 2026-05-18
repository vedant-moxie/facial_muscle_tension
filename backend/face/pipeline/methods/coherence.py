"""
Method 4 — Upper/lower face temporal coherence (Tier 2, weight 0.25).

Old behaviour (audit Tier 0): correlate the entire upper-face envelope with
the entire lower-face envelope, then argmax. That is dominated by the
speaker's slow arousal arc — it almost always returns lag ≈ 0 regardless of
expressive timing.

New behaviour:

  1. Detect expressive *events* — peaks on (upper + lower) — with the same
     prominence machinery used for micro-expressions.
  2. Around each event, take a ±400 ms window and cross-correlate
     upper-vs-lower **within that window only**.
  3. Refine the integer-frame argmax to sub-frame resolution via parabolic
     interpolation around the peak.
  4. Report the **median** per-event lag (robust to outlier events).

This isolates the upper-vs-lower timing during the moments that *carry*
expression, which is what the Ekman / Duchenne literature actually
addresses. The score curve and thresholds are unchanged.

A negative lag means the lower face leads (smile-before-eyes → performed).
A positive lag means the upper face leads (pre-tensing → strain).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks

UPPER_FACE = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07"]
LOWER_FACE = ["AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26"]

GENUINE_LAG_MAX_MS = 100
PERFORMED_LAG_MS = 200
WINDOW_MS = 400               # ±400 ms around each event
EVENT_DISTANCE_MS = 600       # min spacing between detected events


def _parabolic_peak(corr: np.ndarray, idx: int) -> float:
    """Sub-sample peak position via 3-point parabolic interpolation."""
    if idx <= 0 or idx >= len(corr) - 1:
        return float(idx)
    y0, y1, y2 = corr[idx - 1], corr[idx], corr[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-9:
        return float(idx)
    return idx + 0.5 * (y0 - y2) / denom


def compute_coherence(
    au_df: pd.DataFrame,
    fps: int = 12,
    tracking_df: Optional[pd.DataFrame] = None,
) -> dict:
    ms_per_frame = 1000.0 / fps
    upper_cols = [c for c in UPPER_FACE if c in au_df.columns]
    lower_cols = [c for c in LOWER_FACE if c in au_df.columns]

    if not upper_cols or not lower_cols or len(au_df) < fps * 2:
        return _empty_summary("insufficient signal for coherence analysis")

    upper = au_df[upper_cols].sum(axis=1).to_numpy(dtype=float)
    lower = au_df[lower_cols].sum(axis=1).to_numpy(dtype=float)

    # Mask un-scoreable frames so head turns can't be "expressive events".
    if (
        tracking_df is not None
        and "scoreable" in tracking_df.columns
        and len(tracking_df) == len(au_df)
    ):
        sc = tracking_df["scoreable"].astype(bool).to_numpy()
    else:
        sc = np.ones(len(au_df), dtype=bool)

    combined = upper + lower
    combined_masked = np.where(sc, combined, np.nan)

    # Z-score the masked signal for peak detection.
    cmu = np.nanmean(combined_masked)
    csd = np.nanstd(combined_masked) or 1.0
    z = np.where(np.isnan(combined_masked), 0.0, (combined_masked - cmu) / csd)

    peaks, _ = find_peaks(
        z,
        height=0.8,
        distance=max(1, int(EVENT_DISTANCE_MS / ms_per_frame)),
    )

    half_w = max(2, int(WINDOW_MS / ms_per_frame))
    per_event_lags_ms: List[float] = []
    per_event_corrs: List[float] = []

    for p in peaks:
        lo = max(0, p - half_w)
        hi = min(len(upper), p + half_w + 1)
        if hi - lo < 5:
            continue
        u = upper[lo:hi].astype(float)
        l = lower[lo:hi].astype(float)
        if u.std() < 1e-3 or l.std() < 1e-3:
            continue
        u = (u - u.mean()) / (u.std() + 1e-6)
        l = (l - l.mean()) / (l.std() + 1e-6)
        corr = correlate(u, l, mode="full")
        lags = np.arange(-(len(u) - 1), len(u))
        max_lag_frames = int(WINDOW_MS / ms_per_frame)
        mask = np.abs(lags) <= max_lag_frames
        corr_w = corr[mask]
        lags_w = lags[mask]
        idx = int(np.argmax(corr_w))
        sub_idx = _parabolic_peak(corr_w, idx)
        lag_frames = float(lags_w[0] + sub_idx)
        per_event_lags_ms.append(lag_frames * ms_per_frame)
        per_event_corrs.append(float(corr_w[idx] / len(u)))

    if not per_event_lags_ms:
        # Fall back to whole-signal correlation just so we return *something*.
        u = (upper - upper.mean()) / (upper.std() + 1e-6)
        l = (lower - lower.mean()) / (lower.std() + 1e-6)
        corr = correlate(u, l, mode="full")
        lags = np.arange(-(len(u) - 1), len(u))
        max_lag_frames = int(1000.0 / ms_per_frame)
        mask = np.abs(lags) <= max_lag_frames
        corr_w = corr[mask]
        lags_w = lags[mask]
        idx = int(np.argmax(corr_w))
        sub_idx = _parabolic_peak(corr_w, idx)
        peak_lag_ms = float(lags_w[0] + sub_idx) * ms_per_frame
        max_corr = float(corr_w[idx] / len(u))
        events_used = 0
    else:
        peak_lag_ms = float(np.median(per_event_lags_ms))
        max_corr = float(np.median(per_event_corrs))
        events_used = len(per_event_lags_ms)

    abs_lag = abs(peak_lag_ms)
    if abs_lag <= GENUINE_LAG_MAX_MS:
        score = 1.0
        interp = f"upper and lower face sync within {abs_lag:.0f} ms — genuine"
    elif abs_lag <= PERFORMED_LAG_MS:
        score = 1.0 - (abs_lag - GENUINE_LAG_MAX_MS) / (PERFORMED_LAG_MS - GENUINE_LAG_MAX_MS)
        interp = f"slight upper/lower mismatch ({abs_lag:.0f} ms) — ambiguous"
    else:
        score = max(0.0, 0.4 - (abs_lag - PERFORMED_LAG_MS) / 1000)
        interp = (
            f"lower face leads by {abs_lag:.0f} ms — smile running ahead of the eyes"
            if peak_lag_ms < 0
            else f"upper face leads by {abs_lag:.0f} ms — pre-tensing"
        )

    return {
        "peak_lag_ms": round(peak_lag_ms, 1),
        "max_correlation": round(max_corr, 3),
        "events_used": events_used,
        "summary": {
            "method": "coherence",
            "tier": 2,
            "weight": 0.25,
            "score": round(float(score), 3),
            "peak_lag_ms": round(peak_lag_ms, 1),
            "max_correlation": round(max_corr, 3),
            "events_used": events_used,
            "method_detail": "per_event_median" if events_used else "whole_signal_fallback",
            "interpretation": interp,
        },
    }


def _empty_summary(msg: str) -> dict:
    return {
        "peak_lag_ms": 0.0,
        "max_correlation": 0.0,
        "events_used": 0,
        "summary": {
            "method": "coherence",
            "tier": 2,
            "weight": 0.25,
            "score": 0.7,
            "peak_lag_ms": 0.0,
            "events_used": 0,
            "method_detail": "empty",
            "interpretation": msg,
        },
    }
