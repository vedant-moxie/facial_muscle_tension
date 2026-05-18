"""Unit tests for metric primitives on synthetic data."""
from __future__ import annotations

import numpy as np

from presence.metrics.fluidity import sparc


def test_sparc_jittery_more_negative_than_smooth():
    fs = 50.0
    t = np.arange(int(5 * fs)) / fs
    smooth = np.sin(2 * np.pi * 0.5 * t)
    rng = np.random.default_rng(0)
    jittery = smooth + rng.standard_normal(len(t)) * 0.5

    s_smooth = sparc(smooth, fs=fs)
    s_jittery = sparc(jittery, fs=fs)
    assert s_smooth > s_jittery  # less negative = smoother


def test_sparc_handles_short_input():
    assert np.isnan(sparc(np.array([1.0, 2.0]), fs=30.0))


def test_sparc_handles_constant_input():
    # Constant signal → no movement → undefined arc-length; should be NaN
    out = sparc(np.zeros(100), fs=30.0)
    assert np.isnan(out)
