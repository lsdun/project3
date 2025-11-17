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




def model_intensity(
    wavelength: np.ndarray,
    theta: Sequence[float],
) -> np.ndarray:
    """
    Model intensity including a calibration factor
    Parameters
    ----------
    wavelength : ndarray
        Wavelength in meters
    theta : sequence of float
        Model parameters (T, A), where T is temperature [K], A is amplitude
    Returns
    -------
    ndarray
        Model intensities
    """
    T, A = theta
    return A * planck_lambda(wavelength, T)


def log_prior(theta: Sequence[float]) -> float:
    """
    Log prior probability for parameters
    Parameters
    ----------
    theta : sequence of float
        Parameters (T, A)
    Returns
    -------
    float
        Log prior probability
    """
    T, A = theta
    
    if T <= 500 or T >= 10000:
        return -np.inf
    if A <= 0:
        return -np.inf

    return 0.0


def log_likelihood(
    theta: Sequence[float],
    wavelength: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Gaussian log-likelihood for intensity data
    Parameters
    ----------
    theta : sequence of float
        Parameters (T, A)
    wavelength : ndarray
        Wavelengths [m]
    intensity : ndarray
        Observed intensities
    sigma : ndarray
        Measurement uncertainties
    Returns
    -------
    float
        Log-likelihood value
    """
    model = model_intensity(wavelength, theta)
    resid = (intensity - model) / sigma
    return -0.5 * np.sum(resid**2 + np.log(2.0 * np.pi * sigma**2))








