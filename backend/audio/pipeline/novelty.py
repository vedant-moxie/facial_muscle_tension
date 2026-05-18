"""
Novelty scorer.

`novelty_score ∈ [0, 1]`
  0 = identical to trending content (or to other reels in this batch)
  1 = completely unlike trending content AND unlike the creator's other reels

== Three-tier cascade ==

  1. **Chromaprint fingerprint** vs. a pre-built hash set (if pyacoustid
     + a trending pool are installed).
  2. **CLAP audio embedding** vs. a FAISS index of trending embeddings
     (if msclap + faiss + the index are installed).
  3. **Lite-mode hybrid** (always works with just librosa) =
        0.6 * distance_to_canonical_pop_template
      + 0.4 * within_creator_diversity_rank

The big improvement over the previous build: even with no trending pool,
we still produce a meaningful, differentiating score, because (a) we hold
a CANONICAL_POP_TEMPLATE feature vector (literature-derived) every clip
is compared against, and (b) we measure how much each reel deviates from
the average of the creator's OWN reels. A repetitive influencer scores
low; a creator who varies their content scores high.
"""
from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from audio.config import (
    AUDIO_DATA_DIR, CLAP_MODEL,
    SAMPLE_RATE, TRENDING_HASHES_PATH, TRENDING_INDEX_PATH,
)
from audio.pipeline.lite_features import (
    CANONICAL_POP_TEMPLATE, cosine_similarity, extract_lite_features,
)

log = logging.getLogger(__name__)

# ---- detect optional deps ----
try:
    import acoustid                                 # type: ignore
    _HAS_CHROMAPRINT = True
except Exception:                                   # noqa: BLE001
    _HAS_CHROMAPRINT = False

try:
    import faiss                                    # type: ignore
    _HAS_FAISS = True
except Exception:                                   # noqa: BLE001
    _HAS_FAISS = False

try:
    from msclap import CLAP                         # type: ignore
    _HAS_CLAP = True
except Exception:                                   # noqa: BLE001
    _HAS_CLAP = False

try:
    import librosa                                  # type: ignore
    import soundfile as sf                          # type: ignore
    _HAS_LIBROSA = True
except Exception:                                   # noqa: BLE001
    _HAS_LIBROSA = False


_clap_model = None
_faiss_index = None
_trending_hashes: Optional[set] = None


# ---------------------------------------------------------------------------
# Lazy loaders for optional backends
# ---------------------------------------------------------------------------
def _load_clap():
    global _clap_model
    if _clap_model is None and _HAS_CLAP:
        log.info("Loading CLAP (%s) …", CLAP_MODEL)
        _clap_model = CLAP(version=CLAP_MODEL, use_cuda=False)
    return _clap_model


def _load_faiss_index():
    global _faiss_index
    if _faiss_index is None and _HAS_FAISS and Path(str(TRENDING_INDEX_PATH)).exists():
        _faiss_index = faiss.read_index(str(TRENDING_INDEX_PATH))
    return _faiss_index


def _load_trending_hashes() -> set:
    global _trending_hashes
    if _trending_hashes is None:
        path = Path(str(TRENDING_HASHES_PATH))
        if path.exists():
            with open(path, "rb") as f:
                _trending_hashes = set(pickle.load(f))
        else:
            _trending_hashes = set()
    return _trending_hashes


# ---------------------------------------------------------------------------
# Stage 1 — Chromaprint
# ---------------------------------------------------------------------------
def _fingerprint(y: np.ndarray, sr: int) -> Optional[str]:
    if not (_HAS_CHROMAPRINT and _HAS_LIBROSA):
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        sf.write(tmp.name, y, sr, subtype="PCM_16")
        _, fp = acoustid.fingerprint_file(tmp.name)
        if isinstance(fp, bytes):
            return fp.decode("ascii", errors="ignore")
        return fp
    except Exception:                               # noqa: BLE001
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Stage 2 — CLAP
# ---------------------------------------------------------------------------
def _clap_embed(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    clap = _load_clap()
    if clap is None:
        return None
    try:
        embed = clap.get_audio_embeddings_from_array([y], resample=True)
        vec = np.asarray(embed[0]).astype(np.float32)
        n = float(np.linalg.norm(vec) or 1.0)
        return vec / n
    except Exception as exc:                        # noqa: BLE001
        log.warning("CLAP embed failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Lite-mode novelty
# ---------------------------------------------------------------------------
def _within_creator_diversity(features: List[np.ndarray]) -> List[float]:
    """
    For each clip, mean cosine-distance to every OTHER clip in the batch.

    High value = this clip is unlike the creator's other reels.
    Low value  = creator is repetitive across reels.

    Returns one float per clip in [0, 1].
    """
    n = len(features)
    if n <= 1:
        return [0.5] * n
    out: List[float] = []
    for i in range(n):
        sims = []
        for j in range(n):
            if j == i:
                continue
            sims.append(cosine_similarity(features[i], features[j]))
        mean_sim = float(np.mean(sims))
        out.append(float(np.clip(1.0 - mean_sim, 0.0, 1.0)))
    return out


def _lite_novelty(
    features: List[np.ndarray],
    canonical_template: np.ndarray = CANONICAL_POP_TEMPLATE,
    canonical_weight: float = 0.6,
) -> List[Dict]:
    diversity = _within_creator_diversity(features)
    out: List[Dict] = []
    for f, div in zip(features, diversity):
        sim_canon = cosine_similarity(f, canonical_template)
        dist_canon = float(np.clip(1.0 - sim_canon, 0.0, 1.0))
        # 0.6 canonical distance + 0.4 within-creator diversity.
        score = canonical_weight * dist_canon + (1.0 - canonical_weight) * div
        score = float(np.clip(score, 0.0, 1.0))
        out.append({
            "score":             score,
            "basis":             "lite_hybrid",
            "fp_hit":            False,
            "max_sim":           None,
            "canonical_distance": round(dist_canon, 3),
            "within_creator_diversity": round(div, 3),
        })
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def score_novelty_batch(
    waveforms: List[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> List[Dict]:
    """
    Returns a list of dicts:
        {
          "score":     float in [0,1],
          "basis":     "chromaprint" | "clap" | "lite_hybrid",
          "fp_hit":    bool,
          "max_sim":   float | None,
          "canonical_distance"?:        float (lite only),
          "within_creator_diversity"?:  float (lite only),
        }
    """
    hashes = _load_trending_hashes()
    index = _load_faiss_index()

    n = len(waveforms)
    results: List[Optional[Dict]] = [None] * n
    embed_indices: List[int] = []

    # --- Stage 1: chromaprint gate ---
    for i, y in enumerate(waveforms):
        fp = _fingerprint(y, sr)
        if fp and fp in hashes:
            results[i] = {
                "score": 0.0, "basis": "chromaprint",
                "fp_hit": True, "max_sim": 1.0,
            }
        else:
            embed_indices.append(i)

    # --- Stage 2: CLAP for misses (if available) ---
    if embed_indices and _HAS_CLAP and index is not None:
        for i in embed_indices:
            vec = _clap_embed(waveforms[i], sr)
            if vec is not None and vec.shape[0] == index.d:
                D, _ = index.search(vec.reshape(1, -1).astype(np.float32), k=1)
                max_sim = float(D[0, 0])
                results[i] = {
                    "score":   float(np.clip(1.0 - max_sim, 0.0, 1.0)),
                    "basis":   "clap",
                    "fp_hit":  False,
                    "max_sim": max_sim,
                }

    # --- Stage 3: lite hybrid for everything still unresolved ---
    unresolved = [i for i in range(n) if results[i] is None]
    if unresolved:
        features = [extract_lite_features(waveforms[i], sr) for i in unresolved]
        lite_results = _lite_novelty(features)
        for slot, lite_res in zip(unresolved, lite_results):
            results[slot] = lite_res

    return results  # type: ignore[return-value]
