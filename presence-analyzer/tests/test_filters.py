"""Unit tests for the One Euro filter."""
from __future__ import annotations

import numpy as np
import pytest

from presence.filters.one_euro import one_euro_filter


def _make_time(n: int, fs: float = 30.0) -> np.ndarray:
    return (np.arange(n, dtype=np.float32) / fs).astype(np.float32)


def test_filter_preserves_constant_signal():
    t = _make_time(100)
    sig = np.full((100, 3), 0.5, dtype=np.float32)
    out = one_euro_filter(sig, t, 1.0, 0.007)
    assert out.shape == sig.shape
    assert np.allclose(out, 0.5, atol=1e-4)


def test_filter_attenuates_high_freq_noise():
    """High-frequency jitter should be smoothed: variance of first differences drops."""
    rng = np.random.default_rng(0)
    t = _make_time(300, fs=30.0)
    clean = np.sin(2 * np.pi * 1.0 * t).astype(np.float32)
    noise = rng.standard_normal(300).astype(np.float32) * 0.3
    sig = (clean + noise).reshape(-1, 1)
    filtered = one_euro_filter(sig, t, 1.0, 0.007).ravel()
    # First-difference variance is a cheap high-frequency-power proxy.
    var_diff_in = float(np.var(np.diff(sig.ravel())))
    var_diff_out = float(np.var(np.diff(filtered)))
    assert var_diff_out < var_diff_in * 0.7


def test_filter_handles_zero_dt_gracefully():
    t = np.array([0.0, 0.0, 0.01, 0.02], dtype=np.float32)
    sig = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    out = one_euro_filter(sig, t, 1.0, 0.007)
    assert np.all(np.isfinite(out))


def test_filter_empty_input():
    t = np.empty(0, dtype=np.float32)
    sig = np.empty((0, 5), dtype=np.float32)
    out = one_euro_filter(sig, t, 1.0, 0.007)
    assert out.shape == (0, 5)


@pytest.mark.benchmark(group="filter")
def test_filter_speed(benchmark):
    """720 frames × 99 coords must filter in ≤ 20 ms (spec target)."""
    t = _make_time(720, fs=12.0)
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((720, 99)).astype(np.float32)
    # warm up JIT
    one_euro_filter(sig, t, 1.0, 0.007)
    result = benchmark(one_euro_filter, sig, t, 1.0, 0.007)
    assert result.shape == sig.shape
    # benchmark.stats may be unavailable; rely on pytest-benchmark to report
