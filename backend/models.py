"""Pydantic response models for the Effortlessness Analyzer API."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class MethodSummary(BaseModel):
    method: str
    tier: int
    weight: float
    score: float


class MicroExpressionEvent(BaseModel):
    timestamp: float
    timestamp_str: str
    duration_ms: float
    intensity: float
    dominant_au: str


class TimelinePoint(BaseModel):
    second: int
    effortless: float
    tension: Optional[float] = None


class Insight(BaseModel):
    type: str = Field(description="good | watch | info")
    text: str


class AnalysisResult(BaseModel):
    job_id: str
    filename: Optional[str] = None
    duration_secs: float
    frames_analysed: int
    fps: int
    overall_score: float
    overall_label: str
    method_scores: Dict[str, Any]
    timeline: List[TimelinePoint]
    au_means: Dict[str, float]
    au_stds: Dict[str, float]
    micro_expression_events: List[MicroExpressionEvent]
    blink_rate_per_min: float
    asymmetry_mean: float
    insights: List[Insight]
    baseline: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    job_id: str
    status: str = Field(description="queued | processing | done | error")
    progress: float = 0.0
    stage: Optional[str] = None
    message: Optional[str] = None
    result: Optional[AnalysisResult] = None
