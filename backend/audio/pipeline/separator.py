"""
Source separation.

Primary path: Spleeter 2-stems (vocals + accompaniment). One model load,
batched inference.

Fallback path: if Spleeter is not installed (or its TensorFlow dependency
collides with the pinned numpy in this venv), we degrade to a **harmonic /
percussive** split via librosa.effects.hpss. That's not vocals/accompaniment
but it correctly separates "tonal voice-like content" from "drums and
transients", which is enough for the downstream scorers to still produce
useful values. Each downstream module is documented to be robust to this.

Downstream scorers should treat the returned dict as
  {"vocals": ndarray, "accompaniment": ndarray, "separator": "spleeter"|"hpss"|"none"}
and may inspect the "separator" key to adjust confidence.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from audio.config import SAMPLE_RATE, SPLEETER_MODEL

log = logging.getLogger(__name__)

# ---- detect Spleeter ----
try:
    from spleeter.separator import Separator        # type: ignore
    _HAS_SPLEETER = True
except Exception:                                   # noqa: BLE001
    _HAS_SPLEETER = False
    Separator = None  # type: ignore

# ---- detect librosa (fallback) ----
try:
    import librosa                                  # type: ignore
    _HAS_LIBROSA = True
except Exception:                                   # noqa: BLE001
    _HAS_LIBROSA = False


_separator = None


def _get_spleeter():
    global _separator
    if _separator is None:
        log.info("Loading Spleeter (%s) …", SPLEETER_MODEL)
        _separator = Separator(SPLEETER_MODEL)
    return _separator


def _separate_one_spleeter(y: np.ndarray) -> Dict[str, np.ndarray]:
    sep = _get_spleeter()
    stereo = np.stack([y, y], axis=-1).astype(np.float32, copy=False)
    pred = sep.separate(stereo)
    vocals = pred["vocals"][:, 0].astype(np.float32, copy=False)
    accomp = pred["accompaniment"][:, 0].astype(np.float32, copy=False)
    return {"vocals": vocals, "accompaniment": accomp, "separator": "spleeter"}


def _separate_one_hpss(y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    librosa harmonic/percussive split — used as the lite-build fallback.
    Vocals correlate strongly with the harmonic component; accompaniment
    (esp. drums) lives in percussive. Not as clean as Spleeter, but the
    downstream features (F0, BPM, MFCC) are calibrated to be robust to it.
    """
    if not _HAS_LIBROSA:
        return {"vocals": y.copy(), "accompaniment": y.copy(), "separator": "none"}
    harmonic, percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))
    return {
        "vocals":        harmonic.astype(np.float32, copy=False),
        "accompaniment": percussive.astype(np.float32, copy=False),
        "separator":     "hpss",
    }


def separate_batch(
    waveforms: List[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> List[Dict[str, np.ndarray]]:
    """
    Separate each waveform into vocals + accompaniment.

    Returns a list of dicts:
        {"vocals": ndarray, "accompaniment": ndarray, "separator": <backend>}
    where `<backend>` indicates which path was used so downstream scorers
    can adjust confidence accordingly.
    """
    out: List[Dict[str, np.ndarray]] = []
    backend = "spleeter" if _HAS_SPLEETER else ("hpss" if _HAS_LIBROSA else "none")
    if backend != "spleeter":
        log.info(
            "Spleeter not available — falling back to %s separation. "
            "Install with `pip install spleeter` (TF dep — heavy).", backend,
        )

    for y in waveforms:
        try:
            if backend == "spleeter":
                out.append(_separate_one_spleeter(y))
            else:
                out.append(_separate_one_hpss(y))
        except Exception as exc:                    # noqa: BLE001
            log.warning("separator failure on a clip (%s) — passing through raw", exc)
            out.append({"vocals": y.copy(), "accompaniment": y.copy(), "separator": "none"})
    return out
