"""Two-mode scoring: calibrated joblib model OR a transparent fallback.

The fallback maps each raw metric to a 0-100 sub-score via piecewise-linear
"ideal range" mappings derived from the spec's quality targets, then combines
them with the configured weights. Confidence interval widens as overall
quality drops.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from presence.config import ScoringConfig
from presence.scoring.aggregate import VideoFeatures


class PresenceScore(BaseModel):
    score: float
    confidence_low: float
    confidence_high: float
    is_calibrated: bool
    quality_overall: float
    components: dict[str, float]


def _piecewise_score(value: float, ideal_low: float, ideal_high: float, hard_max: float) -> float:
    """Score 100 if value is in [ideal_low, ideal_high], decays to 0 by hard_max."""
    if math.isnan(value):
        return 50.0  # neutral default
    v = abs(value)
    if ideal_low <= v <= ideal_high:
        return 100.0
    if v < ideal_low:
        # closer to 0 than ideal — still good but not perfect
        return 90.0 + 10.0 * (v / max(ideal_low, 1e-6))
    # decay above ideal_high
    decay = (v - ideal_high) / max(hard_max - ideal_high, 1e-6)
    return max(0.0, 100.0 - decay * 100.0)


def _sparc_score(sparc_value: float) -> float:
    """SPARC range typically -2 (smooth) to -10 (jittery). Map linearly."""
    if math.isnan(sparc_value):
        return 50.0
    # -2 → 100, -8 → 0
    return float(np.clip(100.0 - (abs(sparc_value) - 2.0) * (100.0 / 6.0), 0.0, 100.0))


def _component_scores(feats: VideoFeatures) -> dict[str, float]:
    # Tuned to spec: ideal small deviations.
    shoulder = _piecewise_score(feats.shoulder_median_deg, 0.0, 3.0, 15.0)
    head = _piecewise_score(feats.head_pitch_dev_median, 0.0, 5.0, 25.0)
    # Arms reward moderate variance + gesture activity (too still OR too jittery → lower)
    elbow_v = feats.elbow_var_median
    gest = feats.gesture_freq_per_min
    if math.isnan(elbow_v):
        arms_var_score = 50.0
    else:
        # Reward an elbow-angle variance roughly in [5°², 80°²]
        if 5.0 <= elbow_v <= 80.0:
            arms_var_score = 100.0
        elif elbow_v < 5.0:
            arms_var_score = 30.0 + 70.0 * (elbow_v / 5.0)
        else:
            arms_var_score = max(0.0, 100.0 - (elbow_v - 80.0) * (100.0 / 200.0))
    if math.isnan(gest):
        arms_gest_score = 50.0
    else:
        # Reward 4-20 gestures/min
        if 4.0 <= gest <= 20.0:
            arms_gest_score = 100.0
        elif gest < 4.0:
            arms_gest_score = 30.0 + 70.0 * (gest / 4.0)
        else:
            arms_gest_score = max(0.0, 100.0 - (gest - 20.0) * (100.0 / 20.0))
    arms = 0.5 * arms_var_score + 0.5 * arms_gest_score

    sparc_overall_arr = [v for v in (feats.sparc_nose, feats.sparc_wrist_mean)
                         if not math.isnan(v)]
    if not sparc_overall_arr:
        fluidity = 50.0
    else:
        fluidity = float(np.mean([_sparc_score(v) for v in sparc_overall_arr]))

    return {
        "shoulder": float(np.clip(shoulder, 0.0, 100.0)),
        "head": float(np.clip(head, 0.0, 100.0)),
        "arms": float(np.clip(arms, 0.0, 100.0)),
        "fluidity": float(np.clip(fluidity, 0.0, 100.0)),
    }


MIN_COMPONENT_QUALITY = 0.30


def score_features(feats: VideoFeatures, config: ScoringConfig) -> PresenceScore:
    components = _component_scores(feats)
    qualities = {
        "shoulder": feats.quality_shoulder,
        "head": feats.quality_head,
        "arms": feats.quality_arms,
        "fluidity": feats.quality_fluidity,
    }
    # Components below the quality floor are dropped — we'd rather widen the
    # CI than compute a misleading weighted sum that includes a 50-neutral
    # placeholder. `quality_overall` is the *weighted* mean over the kept
    # components, which is a faithful "how much signal did we actually get?".
    retained = {k: q for k, q in qualities.items() if q >= MIN_COMPONENT_QUALITY}
    if retained:
        retained_weight = sum(config.fallback_weights[k] for k in retained)
        quality_overall = float(
            sum(qualities[k] * config.fallback_weights[k] for k in retained)
            / max(retained_weight, 1e-6)
        )
    else:
        quality_overall = float(max(qualities.values()))  # show the best we got

    model_path = Path(config.model_path)
    if model_path.exists():
        try:
            import joblib

            bundle = joblib.load(model_path)
            # bundle: {"model": sklearn_pipeline, "feature_order": [...], "bootstrap_std": float}
            model = bundle["model"]
            feature_order = bundle["feature_order"]
            X = np.array([[getattr(feats, k) for k in feature_order]], dtype=np.float64)
            pred = float(model.predict(X)[0])
            std = float(bundle.get("bootstrap_std", 5.0))
            # widen interval when quality is low
            width_factor = 1.0 + 2.0 * (1.0 - quality_overall)
            half = 1.96 * std * width_factor
            return PresenceScore(
                score=float(np.clip(pred, 0.0, 100.0)),
                confidence_low=float(np.clip(pred - half, 0.0, 100.0)),
                confidence_high=float(np.clip(pred + half, 0.0, 100.0)),
                is_calibrated=True,
                quality_overall=quality_overall,
                components=components,
            )
        except Exception:
            # fall through to fallback if model file is broken
            pass

    weights = config.fallback_weights
    if retained:
        retained_weight = sum(weights[k] for k in retained)
        raw_score = (
            sum(components[k] * weights[k] for k in retained) / max(retained_weight, 1e-6)
        )
    else:
        # No metric cleared the quality floor; fall back to a neutral 50 with
        # a very wide CI so the user knows we couldn't measure them.
        raw_score = 50.0
    # Confidence: tighter when quality is high
    base_half = 4.0
    width_factor = 1.0 + 4.0 * (1.0 - quality_overall)
    half = base_half * width_factor
    return PresenceScore(
        score=float(np.clip(raw_score, 0.0, 100.0)),
        confidence_low=float(np.clip(raw_score - half, 0.0, 100.0)),
        confidence_high=float(np.clip(raw_score + half, 0.0, 100.0)),
        is_calibrated=False,
        quality_overall=quality_overall,
        components=components,
    )
