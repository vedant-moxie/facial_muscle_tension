"""
Genre + tempo edge scorer.

genre_edge_score ∈ [0, 1]
  0 = mainstream genre + mainstream tempo
  1 = niche genre + unusual tempo

Two components combine with a 0.65/0.35 weight:

  • genre_score   ∈ [0,1] — coolness-tier lookup of the predicted genre
  • bpm_score     ∈ [0,1] — how far BPM lands outside the 90–135 mainstream band

== Backends ==

Primary (when installed):
  • Genre via Essentia MusiCNN + Discogs 400 classifier
  • BPM   via madmom RNN beat tracker

Lite fallback (always works with librosa):
  • Genre via MFCC + spectral-centroid heuristic → mapped to a small set
    of coarse genre families ("electronic", "speech", "ambient",
    "percussive", "pop"). Crude but consistent across reels.
  • BPM   via librosa.beat.beat_track.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from audio.config import (
    COOLNESS_TIERS_PATH, EDGE_BPM_HIGH, EDGE_BPM_LOW,
    MAINSTREAM_BPM_HIGH, MAINSTREAM_BPM_LOW, SAMPLE_RATE,
)

log = logging.getLogger(__name__)

try:
    import librosa                                  # type: ignore
    import soundfile as sf                          # type: ignore
    _HAS_LIBROSA = True
except Exception:                                   # noqa: BLE001
    _HAS_LIBROSA = False

try:
    import essentia.standard as es                  # type: ignore
    _HAS_ESSENTIA = True
except Exception:                                   # noqa: BLE001
    _HAS_ESSENTIA = False

try:
    import madmom                                   # type: ignore
    _HAS_MADMOM = True
except Exception:                                   # noqa: BLE001
    _HAS_MADMOM = False


_coolness_tiers: Optional[Dict[str, float]] = None


def _get_coolness_tiers() -> Dict[str, float]:
    global _coolness_tiers
    if _coolness_tiers is None:
        with open(COOLNESS_TIERS_PATH) as f:
            data = json.load(f)
        _coolness_tiers = {k: float(v) for k, v in data.items() if not k.startswith("_")}
    return _coolness_tiers


# ---------------------------------------------------------------------------
# Genre classification
# ---------------------------------------------------------------------------
def _classify_genre_essentia(y: np.ndarray, sr: int) -> Tuple[str, float]:
    """Essentia Discogs-Effnet → 400-class genre. Requires .pb models on disk."""
    tmp = _tmp_wav(y, sr)
    try:
        audio = es.MonoLoader(filename=tmp, sampleRate=sr)()
        # NOTE: caller must have downloaded these models into data/models/.
        embedding_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename="data/models/discogs-effnet-bs64-1.pb",
            output="PartitionedCall:1",
        )
        classifier_model = es.TensorflowPredict2D(
            graphFilename="data/models/genre_discogs400-discogs-effnet-1.pb",
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        embeddings = embedding_model(audio)
        predictions = classifier_model(embeddings)
        mean_preds = np.mean(predictions, axis=0)
        with open("data/models/genre_discogs400_labels.json") as f:
            labels = json.load(f)
        top_idx = int(np.argmax(mean_preds))
        return labels[top_idx], float(mean_preds[top_idx])
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _classify_genre_librosa(y: np.ndarray, sr: int) -> Tuple[str, float]:
    """
    Lite genre classifier — feature-range template matcher.

    For each of 10 genre families, we hold a dict of `feature → (low, high)`
    bounds derived from MIR-literature averages (FMA-medium, GTZAN). The
    classifier:
      1. extracts the same 10 features from this clip
      2. scores each genre's template by `0.6 * fraction_of_bounds_satisfied
         + 0.4 * mean_distance_to_bound_midpoints`
      3. returns the top-scoring genre + its score

    This is intentionally more discriminating than the previous 5-rule
    cascade: it can return `lo-fi`, `indie`, `jazz`, `hip hop`,
    `experimental` and `rock` in addition to pop / electronic / ambient /
    speech — which dramatically lifts the genre-edge sub-score's
    informativeness on real reels.
    """
    if not _HAS_LIBROSA:
        return "no music", 0.0

    from audio.pipeline.lite_features import raw_features, score_genre_templates

    feats = raw_features(y, sr)
    if "silence" in feats:
        return "no music", 0.8
    scores = score_genre_templates(feats)
    if not scores:
        return "pop", 0.25
    top_genre, top_score = max(scores.items(), key=lambda kv: kv[1])
    return top_genre, float(top_score)


def _classify_genre(y: np.ndarray, sr: int) -> Tuple[str, float, str]:
    """Returns (label, confidence, backend)."""
    if _HAS_ESSENTIA:
        try:
            label, conf = _classify_genre_essentia(y, sr)
            return label, conf, "essentia"
        except Exception as exc:                    # noqa: BLE001
            log.warning("Essentia genre failed (%s) — falling back to librosa heuristic", exc)
    label, conf = _classify_genre_librosa(y, sr)
    return label, conf, "librosa_heuristic"


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------
def _tmp_wav(y: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, y, sr, subtype="PCM_16")
    return tmp.name


def _bpm_madmom(y: np.ndarray, sr: int) -> Optional[float]:
    tmp = _tmp_wav(y, sr)
    try:
        proc = madmom.features.beats.RNNBeatProcessor()
        act = proc(tmp)
        beat_proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
        beats = beat_proc(act)
        if len(beats) < 2:
            return None
        intervals = np.diff(beats)
        median_interval = float(np.median(intervals))
        if median_interval < 0.01:
            return None
        return round(60.0 / median_interval, 1)
    except Exception as exc:                        # noqa: BLE001
        log.warning("madmom BPM failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _bpm_librosa(y: np.ndarray, sr: int) -> Optional[float]:
    if not _HAS_LIBROSA:
        return None
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(np.atleast_1d(tempo)[0])
        if len(beats) < 2 or not np.isfinite(tempo) or tempo <= 0:
            return None
        return round(float(tempo), 1)
    except Exception as exc:                        # noqa: BLE001
        log.warning("librosa BPM failed: %s", exc)
        return None


def _estimate_bpm(y: np.ndarray, sr: int) -> Tuple[Optional[float], str]:
    if _HAS_MADMOM:
        v = _bpm_madmom(y, sr)
        if v is not None:
            return v, "madmom"
    return _bpm_librosa(y, sr), "librosa"


def _bpm_edge_score(bpm: Optional[float]) -> float:
    if bpm is None:
        return 1.0      # no detectable pulse = an edge in itself
    if MAINSTREAM_BPM_LOW <= bpm <= MAINSTREAM_BPM_HIGH:
        dist = 0.0
    elif bpm < EDGE_BPM_LOW:
        dist = (EDGE_BPM_LOW - bpm) / EDGE_BPM_LOW
    elif bpm > EDGE_BPM_HIGH:
        dist = (bpm - EDGE_BPM_HIGH) / EDGE_BPM_HIGH
    elif bpm < MAINSTREAM_BPM_LOW:
        dist = (MAINSTREAM_BPM_LOW - bpm) / (MAINSTREAM_BPM_LOW - EDGE_BPM_LOW)
    else:
        dist = (bpm - MAINSTREAM_BPM_HIGH) / (EDGE_BPM_HIGH - MAINSTREAM_BPM_HIGH)
    return float(np.clip(dist, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _normalise_label(raw: str) -> str:
    """`'Electronic --- House'` → `'electronic'` (top-level genre)."""
    head = raw.split(" --- ")[0]
    return head.strip().lower()


def score_genre_edge(
    accompaniment_stem: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> Dict:
    tiers = _get_coolness_tiers()

    genre_label, genre_conf, genre_backend = _classify_genre(accompaniment_stem, sr)
    bpm, bpm_backend = _estimate_bpm(accompaniment_stem, sr)

    key = _normalise_label(genre_label)
    genre_score = float(tiers.get(key, 0.5))
    bpm_score = _bpm_edge_score(bpm)

    score = 0.65 * genre_score + 0.35 * bpm_score

    return {
        "score":              float(np.clip(score, 0.0, 1.0)),
        "genre_label":        genre_label,
        "genre_confidence":   round(genre_conf, 3),
        "genre_prestige":     round(genre_score, 3),
        "bpm":                bpm,
        "bpm_edge_score":     round(bpm_score, 3),
        "genre_backend":      genre_backend,
        "bpm_backend":        bpm_backend,
    }


def score_genre_edge_batch(
    accompaniment_stems: List[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> List[Dict]:
    return [score_genre_edge(a, sr) for a in accompaniment_stems]
