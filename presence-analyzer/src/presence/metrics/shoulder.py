"""Shoulder symmetry metric.

Vectorized via polars expressions:
  shoulder_line_angle = atan2(dy, dx) between left and right shoulder
  de-rolled by per-frame camera_roll_deg
  aggregate: median, IQR of |angle|, n_frames_used
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from pydantic import BaseModel

from presence.pose.quality import LEFT_SHOULDER, RIGHT_SHOULDER


class ShoulderMetric(BaseModel):
    median_deg: float
    iqr_deg: float
    n_frames_used: int
    quality_score: float


def compute_shoulder_metric(
    pose_df: pl.DataFrame, quality_df: pl.DataFrame, max_camera_roll_deg: float
) -> ShoulderMetric:
    ls = (
        pose_df.filter(pl.col("landmark_idx") == LEFT_SHOULDER)
        .select(["frame_idx", pl.col("x").alias("xl"), pl.col("y").alias("yl")])
    )
    rs = (
        pose_df.filter(pl.col("landmark_idx") == RIGHT_SHOULDER)
        .select(["frame_idx", pl.col("x").alias("xr"), pl.col("y").alias("yr")])
    )
    df = ls.join(rs, on="frame_idx").join(quality_df, on="frame_idx")

    usable = df.filter(
        pl.col("shoulder_visible") & (pl.col("camera_roll_deg").abs() < max_camera_roll_deg)
    )

    # angle_raw = atan2(yr - yl, xr - xl), in radians; then convert to degrees,
    # subtract camera_roll_deg to de-roll, take abs.
    angles_df = usable.with_columns(
        [
            (
                (
                    pl.arctan2(
                        pl.col("yr") - pl.col("yl"),
                        pl.col("xr") - pl.col("xl"),
                    )
                    * (180.0 / math.pi)
                )
                - pl.col("camera_roll_deg")
            )
            .abs()
            .alias("angle_deg")
        ]
    )

    n_used = angles_df.height
    n_total = quality_df.filter(pl.col("shoulder_visible")).height
    quality_score = (n_used / n_total) if n_total > 0 else 0.0

    if n_used == 0:
        return ShoulderMetric(
            median_deg=float("nan"),
            iqr_deg=float("nan"),
            n_frames_used=0,
            quality_score=0.0,
        )

    arr = angles_df.get_column("angle_deg").to_numpy()
    median_deg = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    return ShoulderMetric(
        median_deg=median_deg,
        iqr_deg=float(q75 - q25),
        n_frames_used=int(n_used),
        quality_score=float(quality_score),
    )
