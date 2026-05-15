"""
Method 5 — Resting drift + blink dynamics (Tier 3, weight 0.15).

Three changes vs. the original:

  • Blink events are derived from the raw `avg_ear` series against the
    **adaptive per-person threshold** computed in `baseline.py`
    (`ear_threshold`). The hard-coded 0.21 only fires as a fallback.

  • The "genuine" blink-rate band is widened to **8–25 /min** to cover real
    activity contexts (conversation = 17, reading = 6–10, screen = 8–10).
    The original 12–20 band only matched conversational speech and unfairly
    flagged anyone reading from notes as "suppressed".

  • Drift variance uses only **scoreable** frames; the "quiet" cutoff is the
    25th percentile of total AU activation (not the median, which is
    definitionally 50% of frames — that's no longer a quiet detector).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from pipeline.baseline import derive_blink_events_from_threshold

# Wider band: conversation = 17, reading = 6-10, screen = 8-10, monologue ≈ 12.
GENUINE_BLINK_MIN = 8.0
GENUINE_BLINK_MAX = 25.0
SUPPRESSION_BLINK = 4.0
HIGH_AROUSAL_BLINK = 35.0

QUIET_PERCENTILE = 25.0


def analyse_drift_blink(
    au_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    fps: int = 12,
    baseline: Optional[dict] = None,
) -> dict:
    duration_secs = max(1.0, len(tracking_df) / fps)

    # ---- adaptive blink events ----
    threshold = (baseline or {}).get("ear_threshold")
    if threshold is None:
        threshold = 0.21
    blink_idx: List[int] = derive_blink_events_from_threshold(tracking_df, threshold, fps)
    blink_count = len(blink_idx)
    blink_rate = (blink_count / duration_secs) * 60

    if len(blink_idx) > 2:
        intervals = np.diff(blink_idx) / fps
        blink_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
    else:
        blink_cv = 0.5

    # ---- drift: AU variance in quiet (≤ 25th-percentile activation) scoreable frames ----
    au_cols = [c for c in au_df.columns if c.startswith("AU")]
    if au_cols:
        total = au_df[au_cols].sum(axis=1).to_numpy(dtype=float)
        if "scoreable" in tracking_df.columns and len(tracking_df) == len(au_df):
            sc = tracking_df["scoreable"].astype(bool).to_numpy()
        else:
            sc = np.ones(len(au_df), dtype=bool)
        if sc.any():
            cutoff = float(np.percentile(total[sc], QUIET_PERCENTILE))
            quiet_mask = (total <= cutoff) & sc
        else:
            quiet_mask = np.zeros_like(sc)
        quiet_block = au_df.loc[quiet_mask, au_cols].to_numpy()
        quiet_variance = float(np.std(quiet_block)) if quiet_block.size else 0.1
    else:
        quiet_variance = 0.1

    # ---- score blink rate ----
    if GENUINE_BLINK_MIN <= blink_rate <= GENUINE_BLINK_MAX:
        blink_score = 1.0
    elif blink_rate < SUPPRESSION_BLINK:
        blink_score = max(0.0, blink_rate / SUPPRESSION_BLINK * 0.5)
    elif blink_rate < GENUINE_BLINK_MIN:
        blink_score = 0.5 + 0.5 * (blink_rate - SUPPRESSION_BLINK) / (GENUINE_BLINK_MIN - SUPPRESSION_BLINK)
    elif blink_rate < HIGH_AROUSAL_BLINK:
        blink_score = max(0.4, 1.0 - (blink_rate - GENUINE_BLINK_MAX) / (HIGH_AROUSAL_BLINK - GENUINE_BLINK_MAX) * 0.6)
    else:
        blink_score = 0.2

    # Penalise robotic regularity (CV < 0.3 — natural conversation is 0.4–0.8).
    regularity_penalty = max(0.0, 0.3 - blink_cv) * 0.5
    blink_score = float(np.clip(blink_score - regularity_penalty, 0.0, 1.0))

    # Drift score: sweet spot at ~0.2–0.4 variance.
    if quiet_variance < 0.05:
        drift_score = quiet_variance / 0.05 * 0.5
    elif quiet_variance <= 0.4:
        drift_score = 1.0
    else:
        drift_score = max(0.3, 1.0 - (quiet_variance - 0.4) / 0.6)

    final = 0.6 * blink_score + 0.4 * drift_score

    regularity_label = "regular (possible suppression)" if blink_cv < 0.3 else "natural"
    drift_label = (
        "over-controlled" if quiet_variance < 0.05
        else "natural" if quiet_variance <= 0.4
        else "jittery"
    )

    return {
        "blink_rate_per_min": round(blink_rate, 1),
        "blink_count": blink_count,
        "blink_cv": round(blink_cv, 3),
        "quiet_variance": round(quiet_variance, 3),
        "ear_threshold": round(float(threshold), 3),
        "summary": {
            "method": "drift_blink",
            "tier": 3,
            "weight": 0.15,
            "score": round(float(final), 3),
            "blink_rate_per_min": round(blink_rate, 1),
            "blink_regularity": regularity_label,
            "drift_state": drift_label,
            "ear_threshold": round(float(threshold), 3),
            "genuine_band": [GENUINE_BLINK_MIN, GENUINE_BLINK_MAX],
            "interpretation": (
                f"{blink_rate:.1f} blinks/min — {regularity_label}; resting face is {drift_label}"
            ),
        },
    }
