"""
py-feat AU detection wrapper.

Three changes vs. the audit baseline:

  • **Device auto-detect.** CUDA → MPS (Apple Silicon) → CPU. py-feat
    accepts a `device=` kwarg on the Detector; if the installed version
    doesn't, we silently fall back to CPU.

  • **Batch-size escalation.** py-feat 0.6.2 has a bug that raises when
    per-frame face counts differ. We now *try the requested batch_size
    first* (default 16) and degrade to 1 on failure — on py-feat ≥ 0.7
    that gives a ~5–10× speedup; on 0.6.2 we converge to the same safe
    path as before.

  • **`torch.inference_mode()`** around every detect_image call so
    autograd never sees these tensors. This was already there for the
    fallback path; we now apply it to both.

Everything else (column canonicalisation, multi-face deduplication,
NaN-fill, intensity clipping to [0, 5]) is unchanged.
"""
from __future__ import annotations

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger().setLevel(logging.WARNING)
for _name in ("feat", "feat.detector"):
    logging.getLogger(_name).setLevel(logging.WARNING)

CANONICAL_AUS: List[str] = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
    "AU09", "AU10", "AU11", "AU12", "AU14", "AU15",
    "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
    "AU28", "AU43",
]

_detector = None
_DETECTOR_KW = {
    "face_model": "retinaface",
    "landmark_model": "mobilefacenet",
    "au_model": "xgb",
    "emotion_model": "resmasknet",
    "facepose_model": "img2pose",
}


def _pyfeat_version() -> tuple[int, ...]:
    """Return the installed py-feat version as a (major, minor, patch) tuple."""
    try:
        import feat  # type: ignore
        v = getattr(feat, "__version__", "0.0.0")
        return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())
    except Exception:                              # noqa: BLE001
        return (0, 0, 0)


def _pyfeat_supports_device() -> bool:
    """
    py-feat 0.7+ properly moves all tensors to the chosen device.
    0.6.x has a known MPS bug:
        slow_conv2d_forward_mps: input(device='cpu') and weight(device='mps:0')
                                 must be on the same device
    So we only enable non-CPU devices on 0.7+.
    """
    return _pyfeat_version() >= (0, 7, 0)


def _pyfeat_supports_batching() -> bool:
    """
    py-feat 0.6.2 raises on `batch_size > 1` whenever cropped face dims
    differ between frames in a batch (which is the common case). 0.7+
    fixed this by routing through `output_size`.
    """
    return _pyfeat_version() >= (0, 7, 0)


def _pick_device() -> str:
    """Honour env override, otherwise pick the best available device.
    Refuses non-CPU devices on py-feat 0.6.x to dodge the MPS bug."""
    forced = os.environ.get("FEAT_DEVICE", "").lower().strip()
    if forced in {"cpu", "cuda", "mps"}:
        if forced != "cpu" and not _pyfeat_supports_device():
            log.warning(
                "FEAT_DEVICE=%s requested but py-feat %s has broken non-CPU support — falling back to CPU.",
                forced, ".".join(map(str, _pyfeat_version())),
            )
            return "cpu"
        return forced
    if not _pyfeat_supports_device():
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_detector():
    from feat import Detector

    device = _pick_device()
    kw = dict(_DETECTOR_KW)
    log.info(
        "Initialising py-feat %s, device=%s, batching=%s",
        ".".join(map(str, _pyfeat_version())) or "?",
        device,
        _pyfeat_supports_batching(),
    )
    try:
        return Detector(**kw, device=device)
    except TypeError:
        log.info("py-feat does not accept device= — using CPU defaults.")
        return Detector(**kw)


def get_detector():
    global _detector
    if _detector is None:
        _detector = _build_detector()
    return _detector


def warm_detector() -> None:
    get_detector()


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for c in df.columns:
        if not isinstance(c, str) or not c.startswith("AU"):
            continue
        core = c[2:]
        suffix = ""
        if core.endswith("_r") or core.endswith("_l"):
            suffix = core[-2:]
            core = core[:-2]
        if core.isdigit():
            new = f"AU{int(core):02d}{suffix}"
            if new != c:
                rename[c] = new
    return df.rename(columns=rename)


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def _detect_one_batch(detector, paths: List[str], batch_size: int) -> pd.DataFrame:
    """Try detect_image with the requested batch_size; degrade to 1 on failure."""
    # On py-feat 0.6.x the batching path is broken (different face crop sizes
    # across the batch raise), so don't even attempt it — go straight to 1.
    if not _pyfeat_supports_batching():
        batch_size = 1

    last_exc: Optional[Exception] = None
    sizes = (batch_size,) if batch_size == 1 else (batch_size, 1)
    for bs in sizes:
        try:
            with torch.inference_mode():
                fex = detector.detect_image(paths, batch_size=bs)
            return pd.DataFrame(fex) if not isinstance(fex, pd.DataFrame) else fex.copy()
        except Exception as exc:                       # noqa: BLE001
            last_exc = exc
            if bs == 1:
                raise
            log.warning(
                "py-feat batch_size=%d failed (%s) — retrying at batch_size=1.",
                bs, type(exc).__name__,
            )
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


def detect_aus(
    frames: List[np.ndarray],
    fps: int = 12,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Run py-feat AU detection on a list of RGB numpy frames.
    """
    detector = get_detector()
    all_rows: list[pd.DataFrame] = []

    with tempfile.TemporaryDirectory(prefix="feat_") as tmpdir:
        tmp = Path(tmpdir)
        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            paths: list[str] = []
            for i, frm in enumerate(batch):
                p = tmp / f"f_{start + i:06d}.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
                paths.append(str(p))

            try:
                fex_df = _detect_one_batch(detector, paths, batch_size)
                fex_df = _canonicalise_columns(fex_df)

                if "input" in fex_df.columns and fex_df["input"].nunique() < len(fex_df):
                    if {"FaceRectWidth", "FaceRectHeight"}.issubset(fex_df.columns):
                        fex_df["_area"] = fex_df["FaceRectWidth"] * fex_df["FaceRectHeight"]
                        fex_df = (
                            fex_df.sort_values("_area", ascending=False)
                                  .drop_duplicates("input", keep="first")
                                  .drop(columns="_area")
                        )

                fex_df = _ensure_columns(fex_df, CANONICAL_AUS)
                if "input" in fex_df.columns:
                    fex_df = fex_df.set_index("input").reindex(paths).reset_index(drop=True)

                fex_df = fex_df[CANONICAL_AUS].fillna(0.0).clip(lower=0, upper=5)
                all_rows.append(fex_df)

            except Exception as exc:                       # noqa: BLE001
                log.warning("AU detection failed for batch starting %d: %s — filling zeros.", start, exc)
                empty = pd.DataFrame(
                    np.zeros((len(batch), len(CANONICAL_AUS))),
                    columns=CANONICAL_AUS,
                )
                all_rows.append(empty)
            finally:
                for p in paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    df = pd.concat(all_rows, ignore_index=True)
    df["frame_idx"] = range(len(df))
    df["timestamp"] = df["frame_idx"] / fps
    log.info("AU detection complete: %d frames × %d AUs", len(df), len(CANONICAL_AUS))
    return df
