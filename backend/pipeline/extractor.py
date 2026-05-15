"""
Frame extraction.

Two paths, picked at runtime:

  1. **PyAV** (preferred). Uses libav's `best_effort_timestamp` so
     variable-frame-rate video (every phone / WebM screen recording) gets
     the right per-frame timestamp. Decoding is single-pass through
     `container.decode(video=0)`, which is much cheaper than OpenCV's
     "decode-all-then-skip" pattern.

  2. **OpenCV** (fallback). Same as before — works everywhere PyAV isn't
     installed, but uses index/source_fps timestamps (wrong for VFR).

Frames are RGB uint8, optionally downscaled so the long edge ≤ MAX_DIM.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

MAX_FRAMES = 6000
MAX_DIM = 720

try:
    import av                                  # PyAV
    _HAS_AV = True
except Exception:                              # noqa: BLE001
    _HAS_AV = False


def _resize_if_needed(frame: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _extract_pyav(
    video_path: str,
    target_fps: int,
    max_frames: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """PyAV path — VFR-correct timestamps via best_effort_timestamp."""
    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    next_sample_t = 0.0
    sample_dt = 1.0 / float(target_fps)

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:                       # noqa: BLE001
            pass
        time_base = stream.time_base

        source_fps = None
        try:
            r = stream.average_rate or stream.base_rate
            if r is not None:
                source_fps = float(r)
        except Exception:                       # noqa: BLE001
            source_fps = None

        log.info("PyAV decoding %s — avg_fps=%s, target=%d", video_path, source_fps, target_fps)

        for packet in container.demux(stream):
            for frame in packet.decode():
                pts = frame.pts if frame.pts is not None else frame.best_effort_timestamp
                if pts is None:
                    # Fall back to a counter — better than crashing.
                    ts = len(frames) / float(target_fps)
                else:
                    ts = float(pts * time_base)
                if ts + 1e-6 < next_sample_t:
                    continue
                next_sample_t = ts + sample_dt

                rgb = frame.to_ndarray(format="rgb24")
                rgb = _resize_if_needed(rgb)
                frames.append(rgb)
                timestamps.append(round(ts, 3))
                if len(frames) >= max_frames:
                    log.warning("Hit MAX_FRAMES=%d — truncating video.", max_frames)
                    return frames, timestamps

    return frames, timestamps


def _extract_opencv(
    video_path: str,
    target_fps: int,
    max_frames: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """OpenCV fallback path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if source_fps <= 0:
        source_fps = 30.0

    step = max(1, round(source_fps / target_fps))
    effective_fps = source_fps / step
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    log.info(
        "OpenCV decoding %s — source=%.2f fps, step=%d, target≈%.2f fps, total≈%d frames",
        video_path, source_fps, step, effective_fps, total,
    )

    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            # Use CAP_PROP_POS_MSEC when available (more accurate than idx/fps).
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms > 0:
                ts = pos_ms / 1000.0
            else:
                ts = idx / source_fps
            frame = _resize_if_needed(frame)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(round(ts, 3))
            if len(frames) >= max_frames:
                log.warning("Hit MAX_FRAMES=%d — truncating video.", max_frames)
                break
        idx += 1

    cap.release()
    return frames, timestamps


def extract_frames(
    video_path: str,
    target_fps: int = 12,
    max_frames: int = MAX_FRAMES,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Decode `video_path` at ~target_fps. Returns (RGB frames, timestamps_secs).
    """
    if _HAS_AV:
        try:
            frames, ts = _extract_pyav(video_path, target_fps, max_frames)
            log.info("Extracted %d frames via PyAV", len(frames))
            return frames, ts
        except Exception as exc:                  # noqa: BLE001
            log.warning("PyAV decode failed (%s) — falling back to OpenCV.", exc)

    frames, ts = _extract_opencv(video_path, target_fps, max_frames)
    log.info("Extracted %d frames via OpenCV", len(frames))
    return frames, ts
