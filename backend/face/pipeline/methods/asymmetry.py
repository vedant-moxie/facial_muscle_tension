"""
Method 3 — Facial asymmetry (Tier 2, weight 0.25).

Theory:
  • Genuine emotion is mildly asymmetric (index 0.10–0.30).
  • Performed / held expressions are over-symmetric (< 0.08).
  • Distressed faces are highly asymmetric (> 0.50).

Previously this method used a broken "AU coefficient-of-variation" proxy that
measured cross-AU spread rather than left/right asymmetry. It now consumes
true bilateral signals derived from MediaPipe FaceMesh landmarks
(`brow_asym`, `cheek_asym`, `mouth_corner_asym`, `eye_open_asym`) emitted by
`face_tracker.py`. If those columns are absent (legacy callers), it still
falls back to the old AU proxy so nothing crashes.

Bilateral signals are roll-corrected (head-tilt removed) and reference-scaled
by inter-ocular distance, so they're geometry-invariant and land in roughly
the same 0–1 band the AU-intensity literature describes.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

PROBE_AUS = ["AU04", "AU06", "AU12", "AU14"]
BILATERAL_COLS = ["brow_asym", "cheek_asym", "mouth_corner_asym", "eye_open_asym"]


def _au_bilateral_branch(au_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Honour explicit _l/_r AU columns if the backbone provides them."""
    bilateral_cols = [c for c in au_df.columns if c.endswith("_l") or c.endswith("_r")]
    if not bilateral_cols:
        return None
    bases = sorted({c[:-2] for c in bilateral_cols})
    asym = np.zeros(len(au_df), dtype=float)
    for base in bases:
        l = au_df.get(f"{base}_l", pd.Series(0.0, index=au_df.index)).astype(float).to_numpy()
        r = au_df.get(f"{base}_r", pd.Series(0.0, index=au_df.index)).astype(float).to_numpy()
        denom = np.abs(l) + np.abs(r) + 1e-6
        asym += np.abs(l - r) / denom
    asym /= max(1, len(bases))
    return pd.DataFrame({"asym": asym, "timestamp": au_df["timestamp"].values})


def _landmark_branch(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Real bilateral signal from FaceMesh-derived left/right deltas.

    Each of the four probes is already |L − R| / (|L| + |R|). Their mean per
    frame is the per-frame asymmetry index used downstream.
    """
    arr = tracking_df[BILATERAL_COLS].to_numpy(dtype=float)
    # Frames with no face contribute zeros which would bias the mean low;
    # mask them out by setting NaN — caller filters NaNs.
    if "face_present" in tracking_df.columns:
        absent = ~tracking_df["face_present"].astype(bool).to_numpy()
        arr[absent] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        asym = np.nanmean(arr, axis=1)
    return pd.DataFrame({
        "asym": asym,
        "timestamp": tracking_df["timestamp"].astype(float).to_numpy(),
    })


def _au_proxy_branch(au_df: pd.DataFrame) -> pd.DataFrame:
    """Last-resort proxy if neither bilateral AU nor tracking_df is available."""
    cols = [c for c in PROBE_AUS if c in au_df.columns]
    if not cols:
        return pd.DataFrame({
            "asym": [0.15] * len(au_df),
            "timestamp": au_df["timestamp"].values,
        })
    arr = au_df[cols].to_numpy(dtype=float)
    mean_per_frame = arr.mean(axis=1)
    std_per_frame = arr.std(axis=1)
    proxy = std_per_frame / (mean_per_frame + 0.5)
    proxy = np.clip(proxy, 0.0, 1.0)
    return pd.DataFrame({"asym": proxy, "timestamp": au_df["timestamp"].values})


def _score_asym(value: float) -> float:
    if not np.isfinite(value):
        return 0.5
    if value < 0.08:
        return max(0.2, value / 0.08)
    if value <= 0.30:
        return 1.0
    if value <= 0.50:
        return max(0.0, 1.0 - (value - 0.30) / 0.20)
    return 0.0


def compute_asymmetry(
    au_df: pd.DataFrame,
    baseline: dict | None = None,
    tracking_df: pd.DataFrame | None = None,
) -> dict:
    # Preference: real bilateral AUs > FaceMesh landmark deltas > AU proxy.
    branch = "au_bilateral"
    df = _au_bilateral_branch(au_df)
    if df is None or df["asym"].isna().all():
        if tracking_df is not None and all(c in tracking_df.columns for c in BILATERAL_COLS):
            df = _landmark_branch(tracking_df)
            branch = "landmark"
        else:
            df = _au_proxy_branch(au_df)
            branch = "au_proxy"

    # Apply scoreable mask if it's available — un-scoreable frames don't count.
    if (
        tracking_df is not None
        and "scoreable" in tracking_df.columns
        and len(tracking_df) == len(df)
    ):
        mask = tracking_df["scoreable"].astype(bool).to_numpy()
        df.loc[~mask, "asym"] = np.nan

    df["score"] = df["asym"].apply(_score_asym)
    valid = df["asym"].dropna()
    mean_asym = float(valid.mean()) if not valid.empty else 0.15
    mean_score = float(df["score"].mean()) if not df["score"].empty else 0.5

    interp = (
        f"over-symmetric ({mean_asym:.2f}) — possible held/performed expression"
        if mean_asym < 0.08
        else f"genuine asymmetry range ({mean_asym:.2f})"
        if mean_asym <= 0.30
        else f"high asymmetry ({mean_asym:.2f}) — strain or distress"
    )

    return {
        "frame_scores": df.round(4).fillna(0).to_dict(orient="records"),
        "mean": round(mean_asym, 4),
        "summary": {
            "method": "asymmetry",
            "tier": 2,
            "weight": 0.25,
            "score": round(mean_score, 3),
            "mean_asymmetry": round(mean_asym, 3),
            "in_genuine_range": bool(0.08 <= mean_asym <= 0.30),
            "branch": branch,
            "interpretation": interp,
        },
    }
