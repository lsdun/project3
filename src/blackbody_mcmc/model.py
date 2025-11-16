"""Physical and statistical model for blackbody radiation"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import constants as const


def planck_lambda(wavelength: np.ndarray, temperature: float) -> np.ndarray:
    """
    Planck spectral radiance as a function of wavelength
    Parameters
    ----------
    wavelength : ndarray
        Wavelength in meters
    temperature : float
        Temperature in Kelvin
    Returns
    -------
    ndarray
        Spectral radiance
    """
    wl = np.asarray(wavelength, dtype=float)
    T = float(temperature)

    h = const.h
    c = const.c
    k_B = const.k

    a = 2.0 * h * c**2
    b = h * c / (wl * k_B * T)

    # avoids overflow for large b
    intensity = a / (wl**5 * (np.exp(b) - 1.0))
    return intensity













