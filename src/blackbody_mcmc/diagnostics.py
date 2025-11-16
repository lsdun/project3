"""Convergence diagnostics for MCMC chains"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def autocorrelation(chain: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Estimates autocorrelation function for a 1D chain
    Parameters
    ----------
    chain : ndarray
        1D array of samples
    max_lag : int, optional
        Maximum lag to compute
        Defaults to len(chain) // 2
    Returns
    -------
    ndarray
        Autocorrelation for lags 0..max_lag-1
    """
    x = np.asarray(chain, dtype=float)
    x = x - x.mean()
    n = x.size

    if max_lag is None:
        max_lag = n // 2

    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[: max_lag]
    acf /= acf[0]
    return acf










