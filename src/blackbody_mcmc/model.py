"""Physical and statistical model for blackbody radiation"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import constants as const

#Planck function
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

    if np.any(wl <= 0):
        raise ValueError("Wavelengths must be strictly positive.")

    h = const.h
    c = const.c
    k_B = const.k

    a = 2.0 * h * c**2
    b = h * c / (wl * k_B * T)

    # avoids overflow for large b
    denom = np.expm1(b)
    return a / (wl**5 * denom)



#Forward model
def model_intensity(
    wavelength: np.ndarray,
    theta: Sequence[float],
) -> np.ndarray:
    """
    Predicted intensity at given wavelengths.
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

#Priors
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
    
    if not (500.0 < T < 10000.0):
        return -np.inf

    # Amplitude must be positive
    if A <= 0:
        return -np.inf

    # Flat priors → constant log prior
    return 0.0

#Liklihood
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
    theta : (T, A)
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

#Posterior
def log_posterior(
    theta: Sequence[float],
    wavelength: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Log posterior up to a normalisation constant
    Parameters
    ----------
    theta : (T, A)
    wavelength, intensity, sigma : ndarray
        Data and uncertainties
    Returns
    -------
    float
        Log posterior
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, wavelength, intensity, sigma)









