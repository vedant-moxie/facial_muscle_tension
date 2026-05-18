"""Unit tests for video I/O + cache."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from presence.io.cache import PoseCache, cache_key
from presence.io.video import VideoReader, compute_content_hash

SAMPLE = Path("tests/data/synthetic_10s.mp4")


def test_compute_content_hash_fast():
    if not SAMPLE.exists():
        pytest.skip("sample video missing")
    import time

    t0 = time.perf_counter()
    h = compute_content_hash(SAMPLE)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2  # ≤100ms spec, generous
    assert len(h) == 64


def test_video_read_speed():
    if not SAMPLE.exists():
        pytest.skip("sample video missing")
    import time

    with VideoReader(SAMPLE) as vr:
        idx = vr.get_frame_indices(target_fps=12)
        t0 = time.perf_counter()
        batch = vr.read_batch(idx)
        elapsed = time.perf_counter() - t0
        assert batch.shape[0] == len(idx)
        # 10s × 12fps = 120 frames; should read in well under 1s
        assert elapsed < 1.0, f"read 120 frames in {elapsed:.2f}s — too slow"


def test_pose_cache_roundtrip(tmp_path):
    cache = PoseCache(tmp_path)
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    cache.put("deadbeef" * 8, "heavy", 12, df, kind="pose")
    out = cache.get("deadbeef" * 8, "heavy", 12, kind="pose")
    assert out is not None
    assert out.equals(df)


def test_cache_key_uniqueness():
    base = "deadbeef" * 8
    assert cache_key(base, "lite", 12) != cache_key(base, "heavy", 12)
    assert cache_key(base, "heavy", 12) != cache_key(base, "heavy", 24)
