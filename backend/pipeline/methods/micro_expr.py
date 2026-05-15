"""
Method 2 — Tension-burst / micro-expression detection (Tier 3, weight 0.20).

Honest naming caveat: literature-grade micro-expression detection (CASME II,
SAMM, SMIC) is done on 100–200 fps high-speed cameras. At our default 12 fps
a 40 ms event is sub-frame — so what we actually detect here is *tension
bursts*: scipy.find_peaks on the AU4+AU7+AU17+AU23 envelope above a rolling
baseline. We still report the event timestamps because they're useful even
if they aren't strict FACS micro-expressions.

If the input fps is ≥ 30, the duration gate matches the canonical 40–200 ms
window. Below that, we widen the upper bound proportionally and add a
`fidelity` field to the summary so the UI can warn the user.

Frames flagged un-scoreable by the face tracker are NaN-masked before peak
detection so head-turn artefacts don't generate false events.
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
ME_MIN_FPS = 30      # below this, we widen the duration window and flag fidelity


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

    # NaN-mask un-scoreable frames so rolling stats don't see them.
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

    peaks, _ = find_peaks(
        deviation,
        height=sigma * 1.5,
        distance=max(1, int(fps * 0.2)),
    )

    # Duration window — at low fps widen the upper bound so a 1–2-frame burst
    # is still admissible; honesty-flag the fidelity.
    fidelity = "ok" if fps >= ME_MIN_FPS else "tension_burst"
    max_dur_ms = MAX_DURATION_MS if fps >= ME_MIN_FPS else max(MAX_DURATION_MS, 3 * ms_per_frame)
    min_dur_ms = MIN_DURATION_MS

    events: List[dict] = []
    for peak in peaks:
        if not mask[peak]:
            continue
        left = peak
        right = peak
        while left > 0 and deviation[left - 1] > 0:
            left -= 1
        while right < len(deviation) - 1 and deviation[right + 1] > 0:
            right += 1
        dur_frames = right - left + 1
        dur_ms = dur_frames * ms_per_frame
        if not (min_dur_ms <= dur_ms <= max_dur_ms):
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

    # Rate per minute, then score.
    duration_min = max(1 / 60.0, len(au_df) / fps / 60.0)
    rate = len(events) / duration_min
    score = float(np.clip(1.0 - rate / 6.0, 0.0, 1.0))

    interp_base = (
        "no suppression detected"
        if len(events) == 0
        else "within-normal occasional leaks"
        if rate <= 2
        else "suppressed effort is present"
    )
    if fidelity == "tension_burst":
        interp_base += " (low-fps mode — tension bursts, not strict micro-expressions)"

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
