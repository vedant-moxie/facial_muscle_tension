"""Pose-result cache backed by Parquet on disk.

Cache key: content_hash + pose_model + sample_fps + mediapipe_version.
Atomic writes (.tmp + os.rename) for cross-process safety.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import polars as pl

try:
    import mediapipe as mp
    _MP_VERSION = getattr(mp, "__version__", "unknown")
except ImportError:  # pragma: no cover
    _MP_VERSION = "unknown"


def cache_key(content_hash: str, model: str, sample_fps: int) -> str:
    return f"{content_hash[:32]}__m-{model}__fps-{sample_fps}__mp-{_MP_VERSION}"


class PoseCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.pose_dir = self.root / "poses"
        self.face_dir = self.root / "faces"
        self.std_dir = self.root / "standardized"
        for d in (self.pose_dir, self.face_dir, self.std_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str) -> Path:
        if kind == "pose":
            return self.pose_dir / f"{key}.parquet"
        if kind == "face":
            return self.face_dir / f"{key}.parquet"
        raise ValueError(kind)

    def get(
        self, content_hash: str, model: str, sample_fps: int, kind: str = "pose"
    ) -> Optional[pl.DataFrame]:
        path = self._path(kind, cache_key(content_hash, model, sample_fps))
        if not path.exists():
            return None
        try:
            return pl.read_parquet(path)
        except Exception:
            return None

    def put(
        self,
        content_hash: str,
        model: str,
        sample_fps: int,
        df: pl.DataFrame,
        kind: str = "pose",
    ) -> None:
        path = self._path(kind, cache_key(content_hash, model, sample_fps))
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.write_parquet(tmp, compression="zstd")
        os.replace(tmp, path)

    def invalidate(
        self, content_hash: str, model: str = "heavy", sample_fps: int = 12
    ) -> None:
        for kind in ("pose", "face"):
            path = self._path(kind, cache_key(content_hash, model, sample_fps))
            if path.exists():
                path.unlink()

    def standardized_path(self, content_hash: str, target_fps: int) -> Path:
        return self.std_dir / f"{content_hash[:32]}__fps-{target_fps}.mp4"
