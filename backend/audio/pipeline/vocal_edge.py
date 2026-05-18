"""
Vocal edge scorer — how far the voice diverges from the mainstream
influencer norm.

Features (computed on the vocal stem from `separator.py`):

  1. F0 mean — distance from gender-appropriate norm pitch
  2. F0 variance — flatness (deadpan) vs wildness
  3. Words-per-minute — extreme pace is an edge signal
  4. Energy sigma — consistent vs dynamic delivery

vocal_edge_score ∈ [0, 1]
  0 = sounds exactly like a mainstream influencer
  1 = maximally deviant delivery

== Backends ==

Primary (when installed):
  • F0  via CREPE (robust to compressed Instagram AAC)
  • WPM via Whisper tiny (multilingual-capable, accurate word timestamps)

Lite fallback (always works with just librosa):
  • F0  via librosa.pyin (probabilistic YIN — good enough on clean stems)
  • WPM via voice-activity onsets — counts syllabic peaks in the RMS
    envelope and estimates words ≈ syllables / 1.8. Coarser than Whisper
    but tracks the right direction (fast / slow / no speech).
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np

from audio.config import (
    NORM_ENERGY_SIGMA, NORM_F0_FEMALE_HZ, NORM_F0_MALE_HZ,
    NORM_F0_VARIANCE, NORM_WPM, SAMPLE_RATE, WHISPER_MODEL,
)

log = logging.getLogger(__name__)

try:
    import librosa                                  # type: ignore
    import soundfile as sf                          # type: ignore
    _HAS_LIBROSA = True
except Exception:                                   # noqa: BLE001
    _HAS_LIBROSA = False

try:
    import crepe                                    # type: ignore
    _HAS_CREPE = True
except Exception:                                   # noqa: BLE001
    _HAS_CREPE = False

try:
    import whisper                                  # type: ignore
    _HAS_WHISPER = True
except Exception:                                   # noqa: BLE001
    _HAS_WHISPER = False


_whisper_model = None


def _ensure_librosa() -> None:
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required for vocal_edge (`pip install librosa`).")


def _tmp_wav(y: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, y, sr, subtype="PCM_16")
    return tmp.name


# ---------------------------------------------------------------------------
# F0 extraction
# ---------------------------------------------------------------------------
def _f0_crepe(y: np.ndarray, sr: int) -> np.ndarray:
    """CREPE pitch estimator. Returns confident F0 frames in Hz."""
    _, freq, conf, _ = crepe.predict(y, sr, viterbi=True, verbose=0)
    return freq[conf > 0.5]


def _f0_librosa(y: np.ndarray, sr: int) -> np.ndarray:
    """librosa.pyin — probabilistic YIN. Robust enough on cleanish vocals."""
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=2048,
    )
    if f0 is None:
        return np.array([], dtype=np.float32)
    mask = (voiced_flag if voiced_flag is not None else ~np.isnan(f0))
    confident = f0[mask & ~np.isnan(f0)]
    return confident.astype(np.float32)


def _extract_f0(y: np.ndarray, sr: int) -> np.ndarray:
    _ensure_librosa()
    if _HAS_CREPE:
        try:
            return _f0_crepe(y, sr)
        except Exception as exc:                    # noqa: BLE001
            log.warning("CREPE failed (%s) — falling back to librosa.pyin", exc)
    return _f0_librosa(y, sr)


# ---------------------------------------------------------------------------
# Speaking rate
# ---------------------------------------------------------------------------
def _get_whisper():
    global _whisper_model
    if _whisper_model is None and _HAS_WHISPER:
        log.info("Loading Whisper (%s) …", WHISPER_MODEL)
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def _wpm_whisper(y: np.ndarray, sr: int) -> Optional[float]:
    model = _get_whisper()
    if model is None:
        return None
    tmp = _tmp_wav(y, sr)
    try:
        result = model.transcribe(tmp, word_timestamps=True, verbose=False)
        words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
        if len(words) < 2:
            return None
        duration = max(0.0, words[-1]["end"] - words[0]["start"])
        if duration < 1.0:
            return None
        return (len(words) / duration) * 60.0
    except Exception as exc:                        # noqa: BLE001
        log.warning("Whisper transcribe failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _wpm_envelope(y: np.ndarray, sr: int) -> Optional[float]:
    """
    Lite WPM estimate: detect syllabic peaks in the RMS envelope, convert
    syllables → words via a coarse 1.8 ratio (English mean syllables/word).

    This intentionally tracks DIRECTION (fast / slow / silent), not exact
    word counts. Score-wise that's all we need because we normalise to a
    deviation from NORM_WPM.
    """
    _ensure_librosa()
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    # Voice-activity gate: require energy ≥ 1.5× the 30th-percentile floor.
    floor = float(np.percentile(rms, 30) or 1e-6)
    if not np.any(rms > floor * 1.5):
        return None

    # Use librosa.util.peak_pick to find syllable-like onsets.
    syllable_peaks = librosa.util.peak_pick(
        rms, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=floor * 0.5, wait=2,
    )
    if len(syllable_peaks) < 2:
        return None

    hop_s = 256 / sr
    times = syllable_peaks * hop_s
    duration = times[-1] - times[0]
    if duration < 1.0:
        return None

    syllables = len(syllable_peaks)
    words = syllables / 1.8
    return (words / duration) * 60.0


def _speaking_rate_wpm(y: np.ndarray, sr: int) -> Optional[float]:
    if _HAS_WHISPER:
        v = _wpm_whisper(y, sr)
        if v is not None:
            return v
    return _wpm_envelope(y, sr)


# ---------------------------------------------------------------------------
# Energy stats
# ---------------------------------------------------------------------------
def _energy_sigma(y: np.ndarray, sr: int) -> float:
    _ensure_librosa()
    rms = librosa.feature.rms(
        y=y, frame_length=int(sr * 0.5), hop_length=int(sr * 0.25),
    )[0]
    return float(np.std(rms))


# ---------------------------------------------------------------------------
# Additional edge signals
# ---------------------------------------------------------------------------
def _spectral_flatness_mean(y: np.ndarray, sr: int) -> float:
    """
    Mean spectral flatness. ≈ 0 for tonal, ≈ 1 for noisy.
    High values on a vocal stem suggest heavy processing — vocoder,
    distortion, autotune harmonics, or whisper-style breath delivery.
    All four are "edge" markers vs. clean natural pop vocals (~0.10).
    """
    _ensure_librosa()
    flat = librosa.feature.spectral_flatness(y=y)[0]
    return float(np.mean(flat))


def _f0_semitone_range(f0_frames: np.ndarray) -> float:
    """
    Vocal dynamic range in semitones (p10 → p90 of the F0 distribution).
    A monotone delivery sits below ~5 semitones; expressive radio voice
    is ~8–12; very theatrical / sing-songy can exceed 15.
    """
    if len(f0_frames) < 6:
        return 0.0
    lo = float(np.percentile(f0_frames, 10))
    hi = float(np.percentile(f0_frames, 90))
    if lo <= 0 or hi <= 0:
        return 0.0
    return 12.0 * float(np.log2(hi / lo))


def _f0_uptalk_slope(f0_frames: np.ndarray) -> float:
    """
    Mean slope of F0 contour over the last third minus the first third.
    Positive = pitch rises at end of utterances ("uptalk" / valley-girl
    inflection). Negative = pitch falls (authoritative). Returned as a
    signed semitone delta.
    """
    if len(f0_frames) < 6:
        return 0.0
    n = len(f0_frames)
    first = float(np.median(f0_frames[: n // 3]))
    last  = float(np.median(f0_frames[2 * n // 3 :]))
    if first <= 0 or last <= 0:
        return 0.0
    return 12.0 * float(np.log2(last / first))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _normalise_deviation(value: float, norm: float, scale: float) -> float:
    return float(np.clip(abs(value - norm) / scale, 0.0, 1.0))


def score_vocal_edge(
    vocal_stem: np.ndarray,
    sr: int = SAMPLE_RATE,
    gender_hint: str = "unknown",
) -> Dict:
    """Return per-clip vocal-edge dict (see module docstring)."""
    f0_frames = _extract_f0(vocal_stem, sr)
    wpm       = _speaking_rate_wpm(vocal_stem, sr)
    e_sigma   = _energy_sigma(vocal_stem, sr)
    flat_mean = _spectral_flatness_mean(vocal_stem, sr)
    f0_range  = _f0_semitone_range(f0_frames)
    uptalk    = _f0_uptalk_slope(f0_frames)

    if len(f0_frames) > 0:
        f0_mean = float(np.mean(f0_frames))
        f0_var  = float(np.var(f0_frames))
        if gender_hint == "unknown":
            norm_f0 = NORM_F0_FEMALE_HZ if f0_mean > 155 else NORM_F0_MALE_HZ
        else:
            norm_f0 = NORM_F0_FEMALE_HZ if gender_hint == "female" else NORM_F0_MALE_HZ
        feat_f0_mean = _normalise_deviation(f0_mean, norm_f0, norm_f0 * 0.5)
        feat_f0_var  = _normalise_deviation(f0_var, NORM_F0_VARIANCE, NORM_F0_VARIANCE * 2)
    else:
        f0_mean, f0_var = 0.0, 0.0
        feat_f0_mean, feat_f0_var = 0.0, 0.0

    feat_wpm     = _normalise_deviation(wpm, NORM_WPM, 80.0) if wpm else 0.0
    feat_energy  = _normalise_deviation(e_sigma, NORM_ENERGY_SIGMA, NORM_ENERGY_SIGMA * 3)
    # Flatness: pop vocals sit around ~0.10. Heavy effects push >0.25; we
    # cap the deviation scale at 0.20 so 0.30+ saturates at 1.0.
    feat_flatness = _normalise_deviation(flat_mean, 0.10, 0.20)
    # F0 range: ~7 semitones is conversational. Score how FAR from that.
    feat_f0_range = _normalise_deviation(f0_range, 7.0, 8.0) if f0_range > 0 else 0.0
    # Uptalk: ±2 semitones is normal. Beyond that is stylised.
    feat_uptalk   = float(np.clip(abs(uptalk) / 6.0, 0.0, 1.0)) if abs(uptalk) > 1e-3 else 0.0

    # Rebalanced weights — F0 still dominates, but three new edge signals get
    # 30% combined so a creator with autotuned vocals or sing-song delivery
    # actually scores higher even when their mean F0 is "normal".
    score = (
        0.22 * feat_f0_mean   +
        0.18 * feat_f0_var    +
        0.15 * feat_wpm       +
        0.15 * feat_energy    +
        0.12 * feat_flatness  +
        0.10 * feat_f0_range  +
        0.08 * feat_uptalk
    )

    return {
        "score":             float(np.clip(score, 0.0, 1.0)),
        "f0_mean_hz":        round(float(f0_mean), 1),
        "f0_variance":       round(float(f0_var), 1),
        "f0_range_semis":    round(float(f0_range), 2),
        "f0_uptalk_semis":   round(float(uptalk), 2),
        "wpm":               round(float(wpm), 1) if wpm else None,
        "energy_sigma":      round(float(e_sigma), 4),
        "spectral_flatness": round(float(flat_mean), 4),
        "f0_backend":        "crepe" if _HAS_CREPE else "librosa_pyin",
        "wpm_backend":       "whisper" if _HAS_WHISPER else "envelope",
        "feature_breakdown": {
            "f0_mean":   round(feat_f0_mean, 3),
            "f0_var":    round(feat_f0_var, 3),
            "wpm":       round(feat_wpm, 3),
            "energy":    round(feat_energy, 3),
            "flatness":  round(feat_flatness, 3),
            "f0_range":  round(feat_f0_range, 3),
            "uptalk":    round(feat_uptalk, 3),
        },
    }


def score_vocal_edge_batch(
    vocal_stems: List[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> List[Dict]:
    return [score_vocal_edge(v, sr) for v in vocal_stems]
