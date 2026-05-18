"""Single source of truth for all tunable parameters.

All numbers in the codebase come from here. No magic numbers elsewhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PoseConfig(BaseModel):
    model: Literal["lite", "full", "heavy"] = "heavy"
    min_pose_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_visibility: float = Field(0.5, ge=0.0, le=1.0)
    sample_fps: int = Field(12, ge=1, le=60)


class QualityConfig(BaseModel):
    min_usable_frame_fraction: float = Field(0.60, ge=0.0, le=1.0)
    max_camera_roll_deg: float = Field(15.0, ge=0.0)


class FilterConfig(BaseModel):
    one_euro_min_cutoff: float = Field(1.0, gt=0.0)
    one_euro_beta: float = Field(0.007, ge=0.0)


class MetricsConfig(BaseModel):
    rolling_window_seconds: float = Field(5.0, gt=0.0)


class ScoringConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_path: Path = Path("data/models/scorer_v1.joblib")
    fallback_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "shoulder": 0.20,
            "head": 0.20,
            "arms": 0.30,
            "fluidity": 0.30,
        }
    )

    @field_validator("fallback_weights")
    @classmethod
    def weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        s = sum(v.values())
        if not 0.99 <= s <= 1.01:
            raise ValueError(f"fallback_weights must sum to 1.0, got {s}")
        required = {"shoulder", "head", "arms", "fluidity"}
        if set(v.keys()) != required:
            raise ValueError(f"fallback_weights keys must be {required}, got {set(v.keys())}")
        return v


class CacheConfig(BaseModel):
    dir: Path = Path("data/cache")
    enabled: bool = True


class RuntimeConfig(BaseModel):
    num_workers: int = Field(4, ge=1, le=64)
    use_gpu: Literal["auto", "true", "false"] | bool = "auto"


class Config(BaseModel):
    pose: PoseConfig = Field(default_factory=PoseConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        if path is None:
            path = Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)
