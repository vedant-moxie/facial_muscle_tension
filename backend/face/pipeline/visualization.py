"""
Per-frame visualization payload for the SimulationPanel UI.

Built once after analysis, served from `GET /api/visualization/{job_id}`.
Kept separate from the main `result` JSON because it's large (one row per
analysed frame) and the user only needs it when they actually open the
simulation view.

Schema:
{
  "fps": 12,
  "landmark_groups": { "left_eye": [0..5], ... },
  "frames": [
    {
      "t":        0.083,          # seconds
      "present":  true,
      "scoreable":true,
      "pitch":    1.2,            # degrees
      "yaw":     -0.4,
      "roll":     0.9,
      "ear_l":    0.32,
      "ear_r":    0.31,
      "blink":    false,
      "asym":     {brow, cheek, mouth, eye_open},   # 0–1
      "aus":      {AU01: 0.12, AU04: 0.41, ...},
      "landmarks":[[x,y], ...],                     # 26 pairs, 0–1 normalised
      "effortless": 0.82,
      "tension":    0.21
    }, ...
  ]
}
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from face.pipeline.combined_tracker import VIZ_LANDMARK_GROUPS

log = logging.getLogger(__name__)


def build_visualization(
    au_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    m1: dict,
    fps: int,
) -> Dict[str, Any]:
    n = min(len(au_df), len(tracking_df))
    au_cols = [c for c in au_df.columns if c.startswith("AU")]

    # Build a lookup of effortless/tension from tension_aus frame_scores
    # (which is keyed by timestamp). We can index directly because both
    # DataFrames are 0..n-1.
    fs = m1.get("frame_scores") or []
    effortless = [None] * n
    tension = [None] * n
    for i, row in enumerate(fs[:n]):
        effortless[i] = row.get("effortless")
        tension[i] = row.get("tension")

    frames: List[dict] = []
    for i in range(n):
        tr = tracking_df.iloc[i]
        au = au_df.iloc[i]

        # Pack all AUs at 2-decimal precision — tiny once gzip kicks in.
        aus = {c: round(float(au[c]), 2) for c in au_cols}

        frames.append({
            "t":         round(float(tr["timestamp"]), 3),
            "present":   bool(tr["face_present"]),
            "scoreable": bool(tr["scoreable"]),
            "pitch":     round(float(tr["pitch"]), 1),
            "yaw":       round(float(tr["yaw"]), 1),
            "roll":      round(float(tr["roll"]), 1),
            "ear_l":     round(float(tr["left_ear"]), 3),
            "ear_r":     round(float(tr["right_ear"]), 3),
            "blink":     bool(tr["blink"]),
            "asym": {
                "brow":     round(float(tr["brow_asym"]), 3),
                "cheek":    round(float(tr["cheek_asym"]), 3),
                "mouth":    round(float(tr["mouth_corner_asym"]), 3),
                "eye_open": round(float(tr["eye_open_asym"]), 3),
            },
            "aus":       aus,
            "landmarks": list(tr["landmarks"]) if isinstance(tr["landmarks"], list) else [],
            "effortless": float(effortless[i]) if effortless[i] is not None else None,
            "tension":    float(tension[i]) if tension[i] is not None else None,
        })

    log.info("Visualization payload: %d frames", len(frames))

    return {
        "fps": int(fps),
        "landmark_groups": VIZ_LANDMARK_GROUPS,
        "frames": frames,
    }
