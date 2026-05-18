"""
Method 2 — Tension-burst / micro-expression detection (Tier 3, weight 0.20).

Honest naming caveat: literature-grade micro-expression detection (CASME II,
SAMM, SMIC) is done on 100–200 fps high-speed cameras. At our default 12 fps
a 40 ms event is sub-frame — so what we actually detect here is *tension
bursts*: scipy.find_peaks on the AU4+AU7+AU17+AU23 envelope above a rolling
baseline. We still report the event timestamps because they're useful even
if they aren't strict FACS micro-expressions.

== Why this got recalibrated (May 2026) ==

The original detector was too sensitive at 12 fps and produced 30+ "events"
on routine 50-second talking-head videos, clamping the score to 0.00 on
nearly every creator. Three changes:

  1. Stricter peak criterion:
       • height       = 2.5 σ      (was 1.5 σ)
       • prominence   = 1.5 σ      (NEW — peak must stand out above its
                                    neighbouring minima, not just exceed
                                    a flat threshold)
       • distance     = 0.30 s     (was 0.20 s)
       • absolute floor: signal − rolling ≥ 0.50 intensity units, so quiet
                                    regions can't generate fake peaks from
                                    pure noise.

  2. Duration gate at low fps requires ≥ 2 frames above zero deviation
     (a 1-frame blip is almost always py-feat XGB jitter).

  3. Rate-to-score curve softened: linear over 0–10 events/min instead of
     0–6. Spontaneous speech genuinely produces 1–3 tension bursts/min;
     the old curve treated that as "high strain".

If fps ≥ 30, the original literature thresholds (40–200 ms, 1.5 σ) are
restored — that regime IS proper micro-expression territory and the
sensitivity issue doesn't apply.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

log = logging.getLogger(__name__)

MICRO_EXPR_AUS = ["AU04", "AU07", "AU17", "AU23"]
MIN_DURATION_MS = 40
MAX_DURATION_MS = 200
ME_MIN_FPS = 30                # below this we run in tension-burst mode

# Low-fps (tension burst) thresholds:
LOW_FPS_HEIGHT_SIGMA      = 2.5
LOW_FPS_PROMINENCE_SIGMA  = 1.5
LOW_FPS_DISTANCE_S        = 0.30      # min separation between events
LOW_FPS_ABSOLUTE_FLOOR    = 0.50      # signal - rolling at peak must exceed this
LOW_FPS_MIN_FRAMES        = 2         # require sustained burst

# Hi-fps (literature regime) thresholds — restored only when fps ≥ 30.
HI_FPS_HEIGHT_SIGMA       = 1.5
HI_FPS_PROMINENCE_SIGMA   = 0.8
HI_FPS_DISTANCE_S         = 0.20
HI_FPS_ABSOLUTE_FLOOR     = 0.30
HI_FPS_MIN_FRAMES         = 1

# Rate → score (events per minute mapped to 0–1)
SCORE_FLOOR_RATE          = 10.0      # rate at which score reaches 0


def detect_micro_expressions(
    au_df: pd.DataFrame,
    fps: int = 12,
    tracking_df: Optional[pd.DataFrame] = None,
) -> dict:
    cols = [c for c in MICRO_EXPR_AUS if c in au_df.columns]
    if not cols or len(au_df) < fps:
        return {
            "events": [],
            "event_count": 0,
            "summary": {
                "method": "micro_expressions",
                "tier": 3,
                "weight": 0.20,
                "score": 0.8,
                "event_count": 0,
                "fidelity": "insufficient",
                "interpretation": "insufficient signal for micro-expression analysis",
            },
        }

    ms_per_frame = 1000.0 / fps
    signal = au_df[cols].sum(axis=1).to_numpy(dtype=float)

    # NaN-mask un-scoreable frames so they don't influence rolling stats.
    if (
        tracking_df is not None
        and "scoreable" in tracking_df.columns
        and len(tracking_df) == len(au_df)
    ):
        mask = tracking_df["scoreable"].astype(bool).to_numpy()
        signal_masked = np.where(mask, signal, np.nan)
    else:
        mask = np.ones(len(au_df), dtype=bool)
        signal_masked = signal

    rolling = (
        pd.Series(signal_masked).rolling(fps, center=True, min_periods=1).mean().to_numpy()
    )
    deviation = signal_masked - rolling
    deviation = np.where(np.isnan(deviation), 0.0, deviation)
    sigma = float(np.nanstd(deviation) or 1e-3)

    # Pick the right thresholds for the current fps regime.
    if fps >= ME_MIN_FPS:
        height       = sigma * HI_FPS_HEIGHT_SIGMA
        prominence   = sigma * HI_FPS_PROMINENCE_SIGMA
        distance     = max(1, int(fps * HI_FPS_DISTANCE_S))
        abs_floor    = HI_FPS_ABSOLUTE_FLOOR
        min_frames_g = HI_FPS_MIN_FRAMES
        fidelity     = "ok"
        max_dur_ms   = MAX_DURATION_MS
    else:
        height       = sigma * LOW_FPS_HEIGHT_SIGMA
        prominence   = sigma * LOW_FPS_PROMINENCE_SIGMA
        distance     = max(1, int(fps * LOW_FPS_DISTANCE_S))
        abs_floor    = LOW_FPS_ABSOLUTE_FLOOR
        min_frames_g = LOW_FPS_MIN_FRAMES
        fidelity     = "tension_burst"
        max_dur_ms   = max(MAX_DURATION_MS, 4 * ms_per_frame)

    peaks, _ = find_peaks(
        deviation,
        height=height,
        prominence=prominence,
        distance=distance,
    )

    events: List[dict] = []
    for peak in peaks:
        if not mask[peak]:
            continue
        # Absolute floor — protects against high-σ false positives in quiet regions.
        if deviation[peak] < abs_floor:
            continue

        # Walk outwards while we remain above the rolling baseline.
        left = peak
        right = peak
        while left > 0 and deviation[left - 1] > 0:
            left -= 1
        while right < len(deviation) - 1 and deviation[right + 1] > 0:
            right += 1
        dur_frames = right - left + 1
        dur_ms = dur_frames * ms_per_frame
        if dur_frames < min_frames_g:
            continue
        if not (MIN_DURATION_MS <= dur_ms <= max_dur_ms):
            continue

        dominant_au = cols[int(np.argmax([au_df.iloc[peak][c] for c in cols]))]
        ts = float(au_df.iloc[peak]["timestamp"])
        events.append({
            "timestamp": round(ts, 3),
            "timestamp_str": f"{int(ts // 60)}:{int(ts % 60):02d}",
            "duration_ms": round(dur_ms, 1),
            "intensity": round(float(deviation[peak]), 3),
            "dominant_au": dominant_au,
        })

    duration_min = max(1 / 60.0, len(au_df) / fps / 60.0)
    rate = len(events) / duration_min
    score = float(np.clip(1.0 - rate / SCORE_FLOOR_RATE, 0.0, 1.0))

    if len(events) == 0:
        interp_base = "no tension-burst activity detected"
    elif rate < 1.0:
        interp_base = "very low tension-burst rate — relaxed delivery"
    elif rate <= 3.0:
        interp_base = "occasional tension bursts — within normal speech range"
    elif rate <= 6.0:
        interp_base = "elevated tension burst rate"
    else:
        interp_base = "high tension burst activity — likely suppressed effort"

    if fidelity == "tension_burst":
        interp_base += " (low-fps mode — tension bursts, not strict micro-expressions)"

    log.info(
        "micro_expr: σ=%.3f thr=%.3f prom=%.3f → %d peaks → %d events (rate %.1f/min, score %.2f)",
        sigma, height, prominence, len(peaks), len(events), rate, score,
    )

    return {
        "events": events,
        "event_count": len(events),
        "rate_per_min": round(rate, 2),
        "summary": {
            "method": "micro_expressions",
            "tier": 3,
            "weight": 0.20,
            "score": round(score, 3),
            "event_count": len(events),
            "rate_per_min": round(rate, 2),
            "fidelity": fidelity,
            "fps": fps,
            "interpretation": interp_base,
        },
    }
