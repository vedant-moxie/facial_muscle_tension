"""
Audio extraction.

  1. ffmpeg decode video → mono PCM WAV at SAMPLE_RATE
  2. Slide a window of CLIP_DURATION_S seconds across the signal, pick
     the one with the highest mean RMS (most musically dense — avoids
     silent intros/outros).
  3. Return numpy float32 waveform.

Uses librosa for I/O (no subprocess hop needed) when ffmpeg is available
via the system installation. Falls back to a raw ffmpeg subprocess if
librosa can't decode the input directly (e.g., exotic container types).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np

from audio.config import CLIP_DURATION_S, SAMPLE_RATE

log = logging.getLogger(__name__)

try:
    import librosa                                  # type: ignore
    import soundfile as _sf                         # noqa: F401
    _HAS_LIBROSA = True
except Exception:                                   # noqa: BLE001
    _HAS_LIBROSA = False


def _ensure_librosa() -> None:
    if not _HAS_LIBROSA:
        raise ImportError(
            "librosa + soundfile required for audio extraction. "
            "Install with `pip install librosa soundfile`."
        )


def _ffmpeg_decode_to_wav(input_path: str) -> str:
    """Force-decode anything to a 16-bit mono PCM WAV at SAMPLE_RATE."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH — install it (brew install ffmpeg).")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        tmp.name,
    ]
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return tmp.name


def _decode(input_path: str) -> Tuple[np.ndarray, int]:
    """Decode any input → (mono float32 waveform, SAMPLE_RATE)."""
    _ensure_librosa()
    # librosa.load handles MP4/MOV/etc. via audioread + ffmpeg under the hood.
    try:
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        return y.astype(np.float32, copy=False), int(sr)
    except Exception as exc:                         # noqa: BLE001
        log.warning("librosa.load failed for %s (%s) — falling back to ffmpeg subprocess", input_path, exc)
        wav = _ffmpeg_decode_to_wav(input_path)
        try:
            y, sr = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
            return y.astype(np.float32, copy=False), int(sr)
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass


def _peak_energy_window(y: np.ndarray, sr: int) -> np.ndarray:
    """Return the CLIP_DURATION_S-second slice with the highest mean RMS."""
    window_samples = int(CLIP_DURATION_S * sr)
    if len(y) <= window_samples:
        return np.pad(y, (0, window_samples - len(y))).astype(np.float32, copy=False)
    hop = max(1, sr // 2)
    frame_rms = librosa.feature.rms(
        y=y, frame_length=window_samples, hop_length=hop, center=False,
    )[0]
    if len(frame_rms) == 0:
        return y[:window_samples].astype(np.float32, copy=False)
    best_frame = int(np.argmax(frame_rms))
    start = best_frame * hop
    end = start + window_samples
    if end > len(y):
        start = len(y) - window_samples
        end = len(y)
    return y[start:end].astype(np.float32, copy=False)


def extract_audio(input_path: str) -> Tuple[np.ndarray, int]:
    """Decode the input and return its peak-energy `CLIP_DURATION_S` window."""
    y, sr = _decode(input_path)
    clip = _peak_energy_window(y, sr)
    log.info("extracted %.1fs clip from %s (raw len=%.1fs)",
             len(clip) / sr, os.path.basename(input_path), len(y) / sr)
    return clip, sr


def batch_extract(input_paths: List[str]) -> List[Tuple[np.ndarray, int]]:
    """Extract one peak-energy clip per input. Bad files yield a zero clip."""
    results: List[Tuple[np.ndarray, int]] = []
    silence_len = int(CLIP_DURATION_S * SAMPLE_RATE)
    for p in input_paths:
        try:
            results.append(extract_audio(p))
        except Exception as exc:                    # noqa: BLE001
            log.warning("extract failed for %s — filling silence: %s", p, exc)
            results.append((np.zeros(silence_len, dtype=np.float32), SAMPLE_RATE))
    return results
