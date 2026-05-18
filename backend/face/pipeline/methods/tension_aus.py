"""
Method 1 — Tension AU scoring (Tier 4, weight 0.15).

Combines four "effort" AUs into a per-frame strain index, then subtracts the
per-AU baseline so that a person's resting expression is treated as effortless.

    tension     = Σ w_k · max(0, AU_k − baseline_k)
    effortless  = 1 − clip(tension / SCALE, 0, 1)

Two fixes vs. the original Tier-0 audit:

  • AU43 (sustained eye closure) is now duration-gated: it only contributes
    when the closure persists ≥ 500 ms. Without this gate, every normal blink
    (200–400 ms) drove AU43 → 1 and subtracted from effortlessness. EMG
    literature places sustained closure (a real effort cue) at ≥500 ms.

  • Frames marked `scoreable=False` by the face tracker (face absent, or
    |yaw|>25° / |pitch|>20°) are dropped from the mean. Otherwise turning
    away inflates the score by silently filling zero tension.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TENSION_AUS: Dict[str, float] = {
    "AU04": 0.55,   # corrugator — strongest EMG-validated tension probe
    "AU07": 0.20,   # lid tightener
    "AU10": 0.10,   # upper-lip raiser
    "AU14": 0.10,   # dimpler — known suppression marker
    "AU17": 0.05,   # chin raiser — mild suppression
}
SCALE = 3.0

# AU43 contributes IFF the closure runs ≥ 500 ms.
AU43_MIN_DURATION_MS = 500
AU43_THRESHOLD = 1.0       # py-feat XGB scale: > 1.0 = closed eye
AU43_WEIGHT = 0.15         # weight added to per-frame tension when gate fires


def _au43_gated(au43: np.ndarray, fps: int) -> np.ndarray:
    """Boolean array: True for frames inside a ≥500 ms sustained closure."""
    if au43.size == 0:
        return np.zeros(0, dtype=bool)
    min_frames = max(1, int(AU43_MIN_DURATION_MS * fps / 1000.0))
    closed = au43 > AU43_THRESHOLD
    out = np.zeros_like(closed)
    i = 0
    while i < len(closed):
        if closed[i]:
            j = i
            while j < len(closed) and closed[j]:
                j += 1
            if (j - i) >= min_frames:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def score_tension_aus(
    au_df: pd.DataFrame,
    baseline: dict,
    tracking_df: Optional[pd.DataFrame] = None,
    fps: int = 12,
) -> dict:
    base = baseline.get("au_mean", {})

    deviations = pd.DataFrame(index=au_df.index)
    for au, w in TENSION_AUS.items():
        if au not in au_df.columns:
            deviations[au] = 0.0
            continue
        deviations[au] = (au_df[au] - base.get(au, 0.0)).clip(lower=0)

    tension = sum(w * deviations[au] for au, w in TENSION_AUS.items())

    # Duration-gated AU43 contribution (replaces the old un-gated weight).
    if "AU43" in au_df.columns:
        au43 = au_df["AU43"].to_numpy(dtype=float)
        gated = _au43_gated(au43, fps)
        au43_dev = np.clip(au43 - base.get("AU43", 0.0), 0, None)
        tension = tension + pd.Series(au43_dev * gated * AU43_WEIGHT, index=au_df.index)

    effortless = (1.0 - (tension / SCALE).clip(0, 1)).astype(float)

    # ---- scoreable mask ----
    if (
        tracking_df is not None
        and "scoreable" in tracking_df.columns
        and len(tracking_df) == len(au_df)
    ):
        mask = tracking_df["scoreable"].astype(bool).to_numpy()
    else:
        mask = np.ones(len(au_df), dtype=bool)

    eff_masked = effortless.where(mask)
    mean_score = float(eff_masked.dropna().mean()) if mask.any() else 0.5
    std_score = float(eff_masked.dropna().std() or 0.0)

    frame_scores = pd.DataFrame({
        "timestamp": au_df["timestamp"].astype(float),
        "effortless": effortless.round(4),
        "tension": tension.round(4),
        "scoreable": mask,
    })
    for au in list(TENSION_AUS.keys()) + ["AU43"]:
        frame_scores[au] = au_df.get(au, 0).astype(float).round(3)

    worst = (
        frame_scores[frame_scores["scoreable"]]
        .nsmallest(3, "effortless")[["timestamp", "effortless"]]
        .to_dict(orient="records")
    )

    return {
        "frame_scores": frame_scores.to_dict(orient="records"),
        "summary": {
            "method": "tension_aus",
            "tier": 4,
            "weight": 0.15,
            "score": round(mean_score, 3),
            "std": round(std_score, 3),
            "scoreable_fraction": round(float(mask.mean()), 3),
            "worst_moments": worst,
            "interpretation": (
                "low tension — relaxed face" if mean_score > 0.8
                else "moderate tension AUs present"
                if mean_score > 0.55
                else "sustained brow/lid tension detected"
            ),
        },
    }
