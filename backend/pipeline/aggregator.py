"""
Weighted ensemble across the five methods.

Weights sum to 1.0. Tier 2 methods (asymmetry, coherence) get the largest
share because they're the hardest to fake; Tier 4 (tension AUs) gets the
smallest because it's the most prone to false-positive on resting brow shape.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

WEIGHTS: Dict[str, float] = {
    "tension_aus":        0.15,   # Tier 4
    "micro_expressions":  0.20,   # Tier 3
    "asymmetry":          0.25,   # Tier 2
    "coherence":          0.25,   # Tier 2
    "drift_blink":        0.15,   # Tier 3
}


def label_score(score: float) -> str:
    if score >= 0.80:
        return "Highly effortless"
    if score >= 0.65:
        return "Effortless"
    if score >= 0.50:
        return "Moderate effort"
    if score >= 0.35:
        return "Noticeable strain"
    return "High strain"


def aggregate_scores(m1, m2, m3, m4, m5, timestamps: List[float]) -> dict:
    scores = {
        "tension_aus":        float(m1["summary"]["score"]),
        "micro_expressions":  float(m2["summary"]["score"]),
        "asymmetry":          float(m3["summary"]["score"]),
        "coherence":          float(m4["summary"]["score"]),
        "drift_blink":        float(m5["summary"]["score"]),
    }
    overall = sum(WEIGHTS[k] * v for k, v in scores.items())

    timeline: List[dict] = []
    fs = pd.DataFrame(m1.get("frame_scores", []))
    if not fs.empty:
        fs["second"] = fs["timestamp"].astype(float).round().astype(int)
        agg = fs.groupby("second").agg(
            effortless=("effortless", "mean"),
            tension=("tension", "mean"),
        ).reset_index()
        for _, row in agg.iterrows():
            timeline.append({
                "second": int(row["second"]),
                "effortless": round(float(row["effortless"]), 3),
                "tension": round(float(row["tension"]), 3),
            })

    return {
        "overall_score": round(float(overall), 3),
        "label": label_score(overall),
        "method_scores": scores,
        "weights": WEIGHTS,
        "timeline": timeline,
    }
