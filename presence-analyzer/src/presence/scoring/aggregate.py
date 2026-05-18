"""Build the per-video feature vector consumed by the scoring model."""
from __future__ import annotations

from pydantic import BaseModel

from presence.metrics.arms import ArmsMetric
from presence.metrics.fluidity import FluidityMetric
from presence.metrics.head import HeadMetric
from presence.metrics.shoulder import ShoulderMetric


class VideoFeatures(BaseModel):
    shoulder_median_deg: float
    shoulder_iqr_deg: float
    head_pitch_dev_median: float
    head_pitch_dev_iqr: float
    head_yaw_dev_median: float
    head_yaw_dev_iqr: float
    elbow_var_median: float
    gesture_freq_per_min: float
    gesture_amplitude_median: float
    sparc_nose: float
    sparc_wrist_mean: float
    quality_shoulder: float
    quality_head: float
    quality_arms: float
    quality_fluidity: float
    duration_sec: float
    fps_effective: float


def build_features(
    shoulder: ShoulderMetric,
    head: HeadMetric,
    arms: ArmsMetric,
    fluidity: FluidityMetric,
    duration_sec: float,
    fps_effective: float,
) -> VideoFeatures:
    return VideoFeatures(
        shoulder_median_deg=shoulder.median_deg,
        shoulder_iqr_deg=shoulder.iqr_deg,
        head_pitch_dev_median=head.pitch_dev_median_deg,
        head_pitch_dev_iqr=head.pitch_dev_iqr_deg,
        head_yaw_dev_median=head.yaw_dev_median_deg,
        head_yaw_dev_iqr=head.yaw_dev_iqr_deg,
        elbow_var_median=arms.elbow_var_median,
        gesture_freq_per_min=arms.gesture_freq_per_min,
        gesture_amplitude_median=arms.gesture_amplitude_median,
        sparc_nose=fluidity.sparc_nose,
        sparc_wrist_mean=fluidity.sparc_wrist_mean,
        quality_shoulder=shoulder.quality_score,
        quality_head=head.quality_score,
        quality_arms=arms.quality_score,
        quality_fluidity=fluidity.quality_score,
        duration_sec=duration_sec,
        fps_effective=fps_effective,
    )
