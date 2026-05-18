"""MediaPipe pose extraction with caching and pre-allocated numpy fills.

Performance:
- VIDEO running_mode uses internal temporal tracking → 2-3× faster than IMAGE
- One landmarker instance, lazy-loaded
- Pre-allocate (N_frames * 33, ...) arrays; fill by index; convert to polars once
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from presence.config import Config
from presence.io.cache import PoseCache
from presence.io.video import VideoReader, compute_content_hash, standardize_video

logger = logging.getLogger(__name__)

# MediaPipe Tasks API model URLs (from Google)
_POSE_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

N_POSE_LANDMARKS = 33
# 6 face landmarks for PnP head pose: nose tip, chin, left eye outer, right eye outer,
# left mouth corner, right mouth corner. Indices in the MediaPipe face mesh.
FACE_PNP_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}


def _model_dir() -> Path:
    d = Path(os.environ.get("PRESENCE_MODEL_DIR", Path.home() / ".cache" / "presence" / "models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_model(url: str, name: str, timeout: float = 60.0) -> Path:
    path = _model_dir() / name
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    logger.info("Downloading %s …", name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to download MediaPipe model {name}: {e}") from e
    os.replace(tmp, path)
    return path


class PoseExtractor:
    """Pose + face landmark extraction with disk-cached results."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cache = PoseCache(config.cache.dir)
        self._pose_landmarker = None
        self._face_landmarker = None

    def _load_pose(self):  # type: ignore[no-untyped-def]
        if self._pose_landmarker is not None:
            return self._pose_landmarker
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_name = self.config.pose.model
        model_url = _POSE_MODEL_URLS[model_name]
        model_path = _ensure_model(model_url, f"pose_landmarker_{model_name}.task")
        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.pose.min_pose_confidence,
            min_pose_presence_confidence=self.config.pose.min_pose_confidence,
            min_tracking_confidence=self.config.pose.min_pose_confidence,
        )
        self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._mp = mp
        return self._pose_landmarker

    def _load_face(self):  # type: ignore[no-untyped-def]
        if self._face_landmarker is not None:
            return self._face_landmarker
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_model(_FACE_MODEL_URL, "face_landmarker.task")
        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.config.pose.min_pose_confidence,
            min_face_presence_confidence=self.config.pose.min_pose_confidence,
            min_tracking_confidence=self.config.pose.min_pose_confidence,
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        return self._face_landmarker

    def extract(
        self, video_path: Path, with_face: bool = True
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, dict]:
        """Return (pose_df, face_df or None, info).

        pose_df schema: frame_idx int32, timestamp float32, landmark_idx int8,
                       x float32, y float32, z float32, visibility float32
        face_df schema: frame_idx int32, timestamp float32, name str, x float32, y float32
        info: {"content_hash", "duration_sec", "fps_effective", "n_frames"}
        """
        content_hash = compute_content_hash(video_path)
        sample_fps = self.config.pose.sample_fps
        model = self.config.pose.model

        pose_cached = self.cache.get(content_hash, model, sample_fps, kind="pose")
        face_cached = self.cache.get(content_hash, model, sample_fps, kind="face") if with_face else None

        with VideoReader(video_path) as vr:
            duration_sec = vr.duration_sec
            indices = vr.get_frame_indices(sample_fps)
            n_frames = len(indices)
            timestamps_s = (indices.astype(np.float64) / vr.fps).astype(np.float32)
            info = {
                "content_hash": content_hash,
                "duration_sec": float(duration_sec),
                "fps_effective": float(sample_fps),
                "n_frames": int(n_frames),
                "src_fps": float(vr.fps),
                "width": vr.width,
                "height": vr.height,
            }

            pose_df = pose_cached
            face_df = face_cached

            if pose_df is None or (with_face and face_df is None):
                frames = vr.read_batch(indices)
                # Run pose and face landmarkers in parallel — MediaPipe releases
                # the GIL inside C++, so threads actually overlap.
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=2) as pool:
                    pose_future = (
                        pool.submit(self._run_pose, frames, indices, timestamps_s)
                        if pose_df is None
                        else None
                    )
                    face_future = (
                        pool.submit(self._run_face, frames, indices, timestamps_s)
                        if with_face and face_df is None
                        else None
                    )
                    if pose_future is not None:
                        pose_df = pose_future.result()
                        self.cache.put(content_hash, model, sample_fps, pose_df, kind="pose")
                    if face_future is not None:
                        face_df = face_future.result()
                        self.cache.put(content_hash, model, sample_fps, face_df, kind="face")

        return pose_df, face_df, info

    def _run_pose(
        self, frames: np.ndarray, indices: np.ndarray, timestamps_s: np.ndarray
    ) -> pl.DataFrame:
        import mediapipe as mp

        landmarker = self._load_pose()
        T = len(frames)
        total = T * N_POSE_LANDMARKS
        # Pre-allocate flat arrays
        frame_idx_arr = np.empty(total, dtype=np.int32)
        ts_arr = np.empty(total, dtype=np.float32)
        lm_idx_arr = np.empty(total, dtype=np.int8)
        x_arr = np.full(total, np.nan, dtype=np.float32)
        y_arr = np.full(total, np.nan, dtype=np.float32)
        z_arr = np.full(total, np.nan, dtype=np.float32)
        vis_arr = np.zeros(total, dtype=np.float32)

        # Pre-fill indices that are constant regardless of detection success
        lm_range = np.arange(N_POSE_LANDMARKS, dtype=np.int8)
        for i in range(T):
            base = i * N_POSE_LANDMARKS
            frame_idx_arr[base : base + N_POSE_LANDMARKS] = int(indices[i])
            ts_arr[base : base + N_POSE_LANDMARKS] = timestamps_s[i]
            lm_idx_arr[base : base + N_POSE_LANDMARKS] = lm_range

        # MediaPipe timestamps must be strictly monotonically increasing ms ints
        ts_ms_used = -1
        for i in range(T):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[i])
            ts_ms = int(round(float(timestamps_s[i]) * 1000.0))
            if ts_ms <= ts_ms_used:
                ts_ms = ts_ms_used + 1
            ts_ms_used = ts_ms
            result = landmarker.detect_for_video(img, ts_ms)
            if not result.pose_landmarks:
                continue
            lms = result.pose_landmarks[0]
            base = i * N_POSE_LANDMARKS
            for j, lm in enumerate(lms):
                x_arr[base + j] = lm.x
                y_arr[base + j] = lm.y
                z_arr[base + j] = lm.z
                vis_arr[base + j] = lm.visibility

        return pl.DataFrame(
            {
                "frame_idx": frame_idx_arr,
                "timestamp": ts_arr,
                "landmark_idx": lm_idx_arr,
                "x": x_arr,
                "y": y_arr,
                "z": z_arr,
                "visibility": vis_arr,
            }
        )

    def _run_face(
        self, frames: np.ndarray, indices: np.ndarray, timestamps_s: np.ndarray
    ) -> pl.DataFrame:
        import mediapipe as mp

        landmarker = self._load_face()
        T = len(frames)
        names = list(FACE_PNP_INDICES.keys())
        idxs = list(FACE_PNP_INDICES.values())
        n_pts = len(names)
        total = T * n_pts

        frame_idx_arr = np.empty(total, dtype=np.int32)
        ts_arr = np.empty(total, dtype=np.float32)
        name_arr = np.empty(total, dtype=object)
        x_arr = np.full(total, np.nan, dtype=np.float32)
        y_arr = np.full(total, np.nan, dtype=np.float32)

        for i in range(T):
            base = i * n_pts
            frame_idx_arr[base : base + n_pts] = int(indices[i])
            ts_arr[base : base + n_pts] = timestamps_s[i]
            name_arr[base : base + n_pts] = names

        ts_ms_used = -1
        for i in range(T):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[i])
            ts_ms = int(round(float(timestamps_s[i]) * 1000.0))
            if ts_ms <= ts_ms_used:
                ts_ms = ts_ms_used + 1
            ts_ms_used = ts_ms
            result = landmarker.detect_for_video(img, ts_ms)
            if not result.face_landmarks:
                continue
            lms = result.face_landmarks[0]
            base = i * n_pts
            for j, idx in enumerate(idxs):
                if idx < len(lms):
                    x_arr[base + j] = lms[idx].x
                    y_arr[base + j] = lms[idx].y

        return pl.DataFrame(
            {
                "frame_idx": frame_idx_arr,
                "timestamp": ts_arr,
                "name": name_arr.astype(str),
                "x": x_arr,
                "y": y_arr,
            }
        )

    def close(self) -> None:
        for landmarker in (self._pose_landmarker, self._face_landmarker):
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass
        self._pose_landmarker = None
        self._face_landmarker = None
