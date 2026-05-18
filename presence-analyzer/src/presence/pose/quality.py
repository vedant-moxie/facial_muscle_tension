"""Per-frame quality flags computed vectorized over the pose landmark DataFrame."""
from __future__ import annotations

import math

import polars as pl

# Pose landmark indices (MediaPipe Pose 33-landmark model)
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


def _pivot_landmark(df: pl.DataFrame, idx: int) -> pl.DataFrame:
    return (
        df.filter(pl.col("landmark_idx") == idx)
        .select(["frame_idx", "x", "y", "visibility"])
        .rename({"x": f"x_{idx}", "y": f"y_{idx}", "visibility": f"v_{idx}"})
    )


def compute_quality(pose_df: pl.DataFrame, min_visibility: float) -> pl.DataFrame:
    """Return one row per frame_idx with quality flags + camera_roll_deg."""
    parts = [
        _pivot_landmark(pose_df, idx)
        for idx in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER,
                    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)
    ]
    df = parts[0]
    for part in parts[1:]:
        df = df.join(part, on="frame_idx", how="left")

    df = df.with_columns(
        [
            (pl.col(f"v_{LEFT_SHOULDER}") > min_visibility).alias("_ls_ok"),
            (pl.col(f"v_{RIGHT_SHOULDER}") > min_visibility).alias("_rs_ok"),
            (pl.col(f"v_{NOSE}") > min_visibility).alias("_nose_ok"),
            (pl.col(f"v_{LEFT_EYE}") > min_visibility).alias("_le_ok"),
            (pl.col(f"v_{RIGHT_EYE}") > min_visibility).alias("_re_ok"),
            (pl.col(f"v_{LEFT_ELBOW}") > min_visibility).alias("_lel_ok"),
            (pl.col(f"v_{RIGHT_ELBOW}") > min_visibility).alias("_rel_ok"),
            (pl.col(f"v_{LEFT_WRIST}") > min_visibility).alias("_lw_ok"),
            (pl.col(f"v_{RIGHT_WRIST}") > min_visibility).alias("_rw_ok"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("_ls_ok") & pl.col("_rs_ok")).alias("shoulder_visible"),
            (pl.col("_nose_ok") & pl.col("_le_ok") & pl.col("_re_ok")).alias("face_visible"),
            (pl.col("_ls_ok") & pl.col("_lel_ok") & pl.col("_lw_ok")).alias("left_arm_visible"),
            (pl.col("_rs_ok") & pl.col("_rel_ok") & pl.col("_rw_ok")).alias("right_arm_visible"),
        ]
    )

    # camera_roll_deg: angle of eye line from horizontal, in degrees
    df = df.with_columns(
        [
            (
                pl.arctan2(
                    pl.col(f"y_{RIGHT_EYE}") - pl.col(f"y_{LEFT_EYE}"),
                    pl.col(f"x_{RIGHT_EYE}") - pl.col(f"x_{LEFT_EYE}"),
                )
                * (180.0 / math.pi)
            ).alias("camera_roll_deg")
        ]
    )

    # `frame_usable` is a permissive "is there a person on screen?" flag —
    # any of face / shoulder / either arm is enough. The individual metrics
    # gate themselves on the specific landmarks they need, so a strict
    # composite here just hides good data on body-cropped framings (vertical
    # talking-head Reels with no shoulders, hands-only clips, etc.).
    df = df.with_columns(
        [
            (
                pl.col("face_visible")
                | pl.col("shoulder_visible")
                | pl.col("left_arm_visible")
                | pl.col("right_arm_visible")
            ).alias("frame_usable")
        ]
    )

    return df.select(
        [
            "frame_idx",
            "shoulder_visible",
            "face_visible",
            "left_arm_visible",
            "right_arm_visible",
            "camera_roll_deg",
            "frame_usable",
        ]
    )
