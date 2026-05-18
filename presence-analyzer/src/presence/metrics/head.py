"""Head pose via solvePnP using 6 face landmarks.

We compute pitch/yaw/roll per frame, take baseline pitch as the video median
(self-calibration), then report deviation from baseline. Yaw deviation is
also returned for completeness.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


# 3D model: approximate face geometry in mm, origin at nose tip.
# Coordinates chosen to be consistent with image coords (y down, z into screen).
FACE_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # nose_tip
        [0.0, 63.6, -12.5],       # chin
        [-43.3, -32.7, -26.0],    # left_eye_outer
        [43.3, -32.7, -26.0],     # right_eye_outer
        [-28.9, 28.7, -24.1],     # mouth_left
        [28.9, 28.7, -24.1],      # mouth_right
    ],
    dtype=np.float64,
)
FACE_POINT_NAMES = ["nose_tip", "chin", "left_eye_outer", "right_eye_outer",
                    "mouth_left", "mouth_right"]


class HeadMetric(BaseModel):
    pitch_dev_median_deg: float
    pitch_dev_iqr_deg: float
    yaw_dev_median_deg: float
    yaw_dev_iqr_deg: float
    n_frames_used: int
    quality_score: float


def _rotation_to_euler(rmat: np.ndarray) -> tuple[float, float, float]:
    """Convert rotation matrix to (pitch, yaw, roll) in degrees."""
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    if sy >= 1e-6:
        pitch = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
        yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
        yaw = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        roll = 0.0
    return pitch, yaw, roll


def compute_head_metric(
    face_df: pl.DataFrame | None,
    quality_df: pl.DataFrame,
    image_width: int,
    image_height: int,
    total_frames: int | None = None,
) -> HeadMetric:
    if cv2 is None or face_df is None or face_df.height == 0:
        return HeadMetric(
            pitch_dev_median_deg=float("nan"),
            pitch_dev_iqr_deg=float("nan"),
            yaw_dev_median_deg=float("nan"),
            yaw_dev_iqr_deg=float("nan"),
            n_frames_used=0,
            quality_score=0.0,
        )

    # The face-mesh landmarker runs independently of the pose model — gating
    # this metric on the pose model's NOSE/EYE visibility (face_visible) is
    # wrong and discards good face data on body-cropped framings. We rely on
    # the face mesh's own per-frame detection success, which surfaces as NaN
    # x/y in face_df for frames where detection failed.
    if total_frames is None:
        total_frames = max(quality_df.height, 1)

    # Pivot face_df → wide: one row per frame with columns x_nose_tip, y_nose_tip, etc.
    wide = face_df.pivot(
        values=["x", "y"], index="frame_idx", on="name", aggregate_function="first"
    )

    # Camera intrinsic matrix (assume principal point at image center, focal = width)
    focal = float(image_width)
    cx = image_width / 2.0
    cy = image_height / 2.0
    K = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.zeros((4, 1), dtype=np.float64)

    pitches: list[float] = []
    yaws: list[float] = []
    used = 0

    # Build column-name lookup for the pivot result. Polars produces columns like
    # 'x_name_value', 'y_name_value' OR 'x_nose_tip', 'y_nose_tip' depending on version.
    cols = wide.columns

    def col(prefix: str, name: str) -> str:
        cand_a = f"{prefix}_{name}"
        cand_b = f"{prefix}_name_{name}"
        if cand_a in cols:
            return cand_a
        if cand_b in cols:
            return cand_b
        raise KeyError(f"missing pivot column for {prefix}/{name}; have {cols}")

    x_cols = [col("x", n) for n in FACE_POINT_NAMES]
    y_cols = [col("y", n) for n in FACE_POINT_NAMES]
    xs = wide.select(x_cols).to_numpy()  # (T, 6)
    ys = wide.select(y_cols).to_numpy()  # (T, 6)

    # MediaPipe gives normalized coords [0,1]; project to pixel coords.
    for i in range(wide.height):
        x_row = xs[i]
        y_row = ys[i]
        if np.any(np.isnan(x_row)) or np.any(np.isnan(y_row)):
            continue
        pts_2d = np.stack(
            [x_row * image_width, y_row * image_height], axis=1
        ).astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(
            FACE_MODEL_3D, pts_2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            continue
        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, _roll = _rotation_to_euler(rmat)
        pitches.append(pitch)
        yaws.append(yaw)
        used += 1

    if used == 0:
        return HeadMetric(
            pitch_dev_median_deg=float("nan"),
            pitch_dev_iqr_deg=float("nan"),
            yaw_dev_median_deg=float("nan"),
            yaw_dev_iqr_deg=float("nan"),
            n_frames_used=0,
            quality_score=0.0,
        )

    pitch_arr = np.asarray(pitches)
    yaw_arr = np.asarray(yaws)
    pitch_baseline = float(np.median(pitch_arr))
    yaw_baseline = float(np.median(yaw_arr))
    pitch_dev = np.abs(pitch_arr - pitch_baseline)
    yaw_dev = np.abs(yaw_arr - yaw_baseline)

    pq25, pq75 = np.percentile(pitch_dev, [25, 75])
    yq25, yq75 = np.percentile(yaw_dev, [25, 75])

    return HeadMetric(
        pitch_dev_median_deg=float(np.median(pitch_dev)),
        pitch_dev_iqr_deg=float(pq75 - pq25),
        yaw_dev_median_deg=float(np.median(yaw_dev)),
        yaw_dev_iqr_deg=float(yq75 - yq25),
        n_frames_used=int(used),
        quality_score=float(used / max(total_frames, 1)),
    )
