"""
Per-creator composite score aggregator.

Per reel:
    sonic_rebellion_reel = w_nov * novelty + w_voc * vocal_edge + w_gen * genre_edge

Creator:
    creator_score = mean across reels (robust to single outlier)

In addition to the raw scores we surface:
  • z-scores per sub-score across the creator's reels (so the dashboard
    can flag "reel 7 is unusually high on vocal edge" relative to the
    creator's own baseline).
  • plain-English insights describing the strongest within-creator
    deviations and the dominant genre family.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from audio.config import WEIGHT_GENRE, WEIGHT_NOVELTY, WEIGHT_VOCAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _label(score: float) -> str:
    if score >= 0.75: return "Sonic Rebel"
    if score >= 0.55: return "Edge-leaning"
    if score >= 0.35: return "Balanced"
    return "Mainstream"


def _zscores(values: List[float]) -> List[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    arr = np.asarray(values, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd < 1e-6:
        return [0.0] * len(values)
    return [(v - mu) / sd for v in values]


def _insight_chip(text: str, kind: str) -> Dict:
    return {"type": kind, "text": text}


def _build_insights(
    per_reel: List[Dict],
    creator_score: float,
    sub_means: Dict[str, float],
) -> List[Dict]:
    insights: List[Dict] = []
    n = len(per_reel)

    # Headline
    if creator_score >= 0.55:
        insights.append(_insight_chip(
            f"Overall {creator_score:.2f} ({_label(creator_score)}) — "
            f"this creator's audio choices skew away from mainstream defaults.",
            "good",
        ))
    elif creator_score >= 0.35:
        insights.append(_insight_chip(
            f"Overall {creator_score:.2f} ({_label(creator_score)}) — "
            f"some edge signals but anchored in mainstream territory.",
            "info",
        ))
    else:
        insights.append(_insight_chip(
            f"Overall {creator_score:.2f} ({_label(creator_score)}) — "
            f"consistently mainstream audio across the sampled reels.",
            "info",
        ))

    # Dominant genre family
    genres = [r["genre_details"].get("genre_label", "?") for r in per_reel if r.get("genre_details")]
    if genres:
        top, count = Counter(genres).most_common(1)[0]
        if count >= max(2, n // 2):
            insights.append(_insight_chip(
                f"Dominant genre across reels: {top} ({count}/{n}).",
                "info",
            ))
        else:
            insights.append(_insight_chip(
                f"Genre is varied — top family {top} only {count}/{n}.",
                "good",
            ))

    # Outliers — biggest sub-score z-scores
    nov_z = _zscores([r["novelty"]    for r in per_reel])
    voc_z = _zscores([r["vocal_edge"] for r in per_reel])
    gen_z = _zscores([r["genre_edge"] for r in per_reel])
    for i, r in enumerate(per_reel):
        r["z_scores"] = {
            "novelty":    round(float(nov_z[i]), 2),
            "vocal_edge": round(float(voc_z[i]), 2),
            "genre_edge": round(float(gen_z[i]), 2),
        }

    candidates: List[Dict] = []
    if n >= 3:
        # Highest-zscore reel per dimension
        for label, zs, key in (
            ("novelty",    nov_z, "novelty"),
            ("vocal edge", voc_z, "vocal_edge"),
            ("genre edge", gen_z, "genre_edge"),
        ):
            i_max = int(np.argmax(zs))
            if zs[i_max] >= 1.0:
                candidates.append(_insight_chip(
                    f"Reel #{i_max + 1} is unusually high on {label} "
                    f"(z = {zs[i_max]:+.1f} vs. creator mean of {sub_means[key]:.2f}).",
                    "good",
                ))
            i_min = int(np.argmin(zs))
            if zs[i_min] <= -1.0:
                candidates.append(_insight_chip(
                    f"Reel #{i_min + 1} is unusually low on {label} "
                    f"(z = {zs[i_min]:+.1f} vs. creator mean of {sub_means[key]:.2f}).",
                    "watch",
                ))
    insights.extend(candidates[:3])

    # Calibration hint
    if per_reel and per_reel[0]["novelty_details"].get("basis") == "lite_hybrid":
        insights.append(_insight_chip(
            "Novelty in lite-mode: combined against a canonical-pop template "
            "and within-creator diversity. Install msclap + a trending-pool "
            "index for trending-similarity novelty.",
            "info",
        ))

    return insights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def aggregate_reel_scores(
    novelty_results:    List[Dict],
    vocal_edge_scores:  List[Dict],
    genre_edge_scores:  List[Dict],
    reel_filenames:     List[str] | None = None,
) -> Dict:
    n = min(len(novelty_results), len(vocal_edge_scores), len(genre_edge_scores))
    per_reel: List[Dict] = []

    for i in range(n):
        nov = float(novelty_results[i]["score"])
        voc = float(vocal_edge_scores[i]["score"])
        gen = float(genre_edge_scores[i]["score"])
        reel_score = WEIGHT_NOVELTY * nov + WEIGHT_VOCAL * voc + WEIGHT_GENRE * gen
        per_reel.append({
            "reel_index":      i,
            "filename":        (reel_filenames[i] if reel_filenames and i < len(reel_filenames) else None),
            "novelty":         round(nov, 3),
            "vocal_edge":      round(voc, 3),
            "genre_edge":      round(gen, 3),
            "reel_score":      round(reel_score, 3),
            "novelty_details": novelty_results[i],
            "vocal_details":   vocal_edge_scores[i],
            "genre_details":   genre_edge_scores[i],
        })

    reel_scores = [r["reel_score"] for r in per_reel] or [0.0]
    creator_score = float(np.mean(reel_scores))

    sub_means = {
        "novelty":    round(float(np.mean([r["novelty"]    for r in per_reel] or [0.0])), 3),
        "vocal_edge": round(float(np.mean([r["vocal_edge"] for r in per_reel] or [0.0])), 3),
        "genre_edge": round(float(np.mean([r["genre_edge"] for r in per_reel] or [0.0])), 3),
    }

    # Within-creator dispersion — how much variety across reels.
    sub_stds = {
        "novelty":    round(float(np.std([r["novelty"]    for r in per_reel] or [0.0])), 3),
        "vocal_edge": round(float(np.std([r["vocal_edge"] for r in per_reel] or [0.0])), 3),
        "genre_edge": round(float(np.std([r["genre_edge"] for r in per_reel] or [0.0])), 3),
    }

    insights = _build_insights(per_reel, creator_score, sub_means)

    return {
        "creator_score":  round(creator_score, 3),
        "score_label":    _label(creator_score),
        "weights": {
            "novelty":    WEIGHT_NOVELTY,
            "vocal_edge": WEIGHT_VOCAL,
            "genre_edge": WEIGHT_GENRE,
        },
        "sub_score_means": sub_means,
        "sub_score_stds":  sub_stds,
        "per_reel":        per_reel,
        "insights":        insights,
        "n_reels_scored":  n,
    }
