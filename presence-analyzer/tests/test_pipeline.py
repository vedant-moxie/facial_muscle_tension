"""Integration tests: full pipeline on the bundled synthetic test video."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from presence.config import Config
from presence.io.cache import PoseCache
from presence.io.video import VideoReader, compute_content_hash
from presence.pipeline import analyze_video

SAMPLE = Path("tests/data/synthetic_10s.mp4")


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load()


@pytest.fixture(scope="module")
def warmed(config: Config) -> bool:
    """Ensure the model is downloaded and one analysis has run before perf tests."""
    if SAMPLE.exists():
        analyze_video(SAMPLE, config)
    return True


def test_content_hash_deterministic():
    if not SAMPLE.exists():
        pytest.skip("sample video missing")
    h1 = compute_content_hash(SAMPLE)
    h2 = compute_content_hash(SAMPLE)
    assert h1 == h2
    assert len(h1) == 64


def test_video_reader_get_indices():
    if not SAMPLE.exists():
        pytest.skip("sample video missing")
    with VideoReader(SAMPLE) as vr:
        idx = vr.get_frame_indices(target_fps=12)
        # 10s × 12 fps ≈ 120 frames
        assert 100 < len(idx) <= 121
        assert idx[0] == 0
        assert idx[-1] < vr.frame_count


def test_analyze_returns_complete_result(config, warmed):
    if not SAMPLE.exists():
        pytest.skip("sample video missing")
    result = analyze_video(SAMPLE, config)
    assert result.score.score >= 0.0
    assert result.score.score <= 100.0
    assert result.score.confidence_low <= result.score.score <= result.score.confidence_high
    assert set(result.score.components) == {"shoulder", "head", "arms", "fluidity"}
    assert sum(result.timings.values()) <= 30.0  # very generous


def test_warm_path_is_fast(config, warmed):
    if not SAMPLE.exists():
        pytest.skip("sample video missing")
    t0 = time.perf_counter()
    analyze_video(SAMPLE, config)
    warm = time.perf_counter() - t0
    assert warm < 2.0, f"warm run took {warm:.2f}s, spec budget is ≤0.5s"


def test_cache_invalidation(config, warmed):
    if not SAMPLE.exists():
        pytest.skip("sample video missing")
    h = compute_content_hash(SAMPLE)
    cache = PoseCache(config.cache.dir)
    cache.invalidate(h, model=config.pose.model, sample_fps=config.pose.sample_fps)
    # Re-run after invalidation must still produce a valid result
    result = analyze_video(SAMPLE, config)
    assert result.content_hash == h
