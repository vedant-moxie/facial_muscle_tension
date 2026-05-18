"""Vectorized One Euro filter (Casiez et al. 2012), JIT-compiled with numba.

Per-timestep state dependency along the time axis means we can't vectorize
across time. But we CAN vectorize across N landmark dimensions — the inner
loop is over time, the inner work is a numpy array op of shape (N,).
"""
from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(cache=True, fastmath=True)
def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


@numba.njit(cache=True, fastmath=True)
def one_euro_filter(
    values: np.ndarray,        # (T, N) float32 — N can be 99 (33 lm × xyz), etc.
    timestamps: np.ndarray,    # (T,) float32 in seconds
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    T, N = values.shape
    out = np.empty((T, N), dtype=np.float32)
    if T == 0:
        return out
    # Work in float32 throughout to avoid dtype unification issues.
    x_hat_prev = np.empty(N, dtype=np.float32)
    dx_hat_prev = np.zeros(N, dtype=np.float32)
    for j in range(N):
        x_hat_prev[j] = values[0, j]
        out[0, j] = values[0, j]
    t_prev = np.float32(timestamps[0])
    for i in range(1, T):
        t_e = np.float32(timestamps[i] - t_prev)
        if t_e <= np.float32(0.0):
            t_e = np.float32(1e-6)
        t_prev = np.float32(timestamps[i])
        a_d = np.float32(_smoothing_factor(float(t_e), float(d_cutoff)))
        for j in range(N):
            dx_j = np.float32((values[i, j] - x_hat_prev[j]) / t_e)
            dx_hat_j = np.float32(a_d * dx_j + (np.float32(1.0) - a_d) * dx_hat_prev[j])
            cutoff_j = float(min_cutoff) + float(beta) * abs(float(dx_hat_j))
            a_j = np.float32(_smoothing_factor(float(t_e), cutoff_j))
            new_val = np.float32(a_j * values[i, j] + (np.float32(1.0) - a_j) * x_hat_prev[j])
            out[i, j] = new_val
            x_hat_prev[j] = new_val
            dx_hat_prev[j] = dx_hat_j
    return out


def warmup_filter() -> None:
    """Trigger JIT compile once so first user call is fast.

    With cache=True numba persists the compile across runs, so this is only
    expensive on the very first invocation per machine.
    """
    t = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    v = np.zeros((4, 2), dtype=np.float32)
    one_euro_filter(v, t, 1.0, 0.007)
