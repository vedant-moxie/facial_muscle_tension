"""Fast video I/O using decord (or eva-decord on Apple Silicon).

Performance principles:
- decord with native bridge gives raw numpy output and uses internal threading
- batches of ≤64 frames keep allocations bounded
- content hash uses first MB + last MB + filesize (deterministic, sub-100ms)
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import decord  # type: ignore[import-not-found]
    decord.bridge.set_bridge("native")
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "decord/eva-decord is required. On Apple Silicon, eva-decord is installed."
    ) from e


MAX_BATCH = 64
_HASH_CHUNK = 1024 * 1024  # 1 MB head + tail


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


def compute_content_hash(path: Path) -> str:
    """SHA-256 of (first MB || last MB || filesize). Fast and deterministic."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        head = f.read(min(_HASH_CHUNK, size))
        h.update(head)
        if size > _HASH_CHUNK:
            tail_start = max(_HASH_CHUNK, size - _HASH_CHUNK)
            f.seek(tail_start)
            h.update(f.read(_HASH_CHUNK))
    h.update(size.to_bytes(8, "little"))
    return h.hexdigest()


def _ffmpeg_version_ok() -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True, timeout=5
        ).stdout
        # parse "ffmpeg version N.M..." — accept >= 4
        first = out.splitlines()[0]
        parts = first.split()
        if len(parts) < 3:
            return False
        ver = parts[2].split(".")
        return int(ver[0]) >= 4
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError):
        return False


def standardize_video(input_path: Path, output_path: Path, target_fps: int) -> VideoMetadata:
    """Re-encode video to constant frame rate. Skip if already at target fps."""
    if not _ffmpeg_version_ok():
        raise RuntimeError("ffmpeg >= 4 is required on PATH")

    meta = probe(input_path)
    if abs(meta.fps - target_fps) < 0.05 and output_path.exists():
        return probe(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-r",
        str(target_fps),
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    os.replace(tmp, output_path)
    return probe(output_path)


def probe(path: Path) -> VideoMetadata:
    vr = decord.VideoReader(str(path))
    fps = float(vr.get_avg_fps())
    n = len(vr)
    if n == 0 or fps <= 0:
        raise ValueError(f"Invalid video at {path}: fps={fps} frames={n}")
    h, w = vr[0].shape[:2]
    return VideoMetadata(
        path=path,
        fps=fps,
        frame_count=n,
        duration_sec=n / fps,
        width=int(w),
        height=int(h),
    )


class VideoReader:
    """Wrapper over decord with batched access + frame-index sampling."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._vr = decord.VideoReader(str(self.path))
        self.fps = float(self._vr.get_avg_fps())
        self.frame_count = len(self._vr)
        if self.frame_count == 0 or self.fps <= 0:
            raise ValueError(f"Invalid video: {self.path}")
        h, w = self._vr[0].shape[:2]
        self.height = int(h)
        self.width = int(w)
        self.duration_sec = self.frame_count / self.fps

    def get_frame_indices(self, target_fps: int) -> np.ndarray:
        """Return frame indices to sample at the target rate (uniform stride)."""
        if target_fps >= self.fps:
            return np.arange(self.frame_count, dtype=np.int32)
        stride = self.fps / target_fps
        n_target = int(self.duration_sec * target_fps)
        indices = np.floor(np.arange(n_target) * stride).astype(np.int32)
        indices = indices[indices < self.frame_count]
        return indices

    def read_batch(self, indices: np.ndarray) -> np.ndarray:
        """Read frames at the given indices as (N, H, W, 3) uint8."""
        if len(indices) == 0:
            return np.empty((0, self.height, self.width, 3), dtype=np.uint8)
        # Iterate in batches of MAX_BATCH to keep peak memory bounded.
        out = np.empty((len(indices), self.height, self.width, 3), dtype=np.uint8)
        for start in range(0, len(indices), MAX_BATCH):
            chunk = indices[start : start + MAX_BATCH]
            frames = self._vr.get_batch(chunk.tolist())
            # decord returns NDArray; convert to numpy without copying when possible
            if hasattr(frames, "asnumpy"):
                arr = frames.asnumpy()
            else:
                arr = np.asarray(frames)
            out[start : start + len(chunk)] = arr
        return out

    def close(self) -> None:
        self._vr = None  # type: ignore[assignment]

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:  # type: ignore[no-untyped-def]
        self.close()
