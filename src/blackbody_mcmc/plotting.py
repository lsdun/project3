"""Plotting utilities for blackbody MCMC results"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_data_and_model(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    theta: Sequence[float],
    model_fn,
    outfile: str | Path,
) -> None:
    """Plots data with errorbars and best-fit model curve"""
    wl = np.asarray(wavelength)
    I = np.asarray(intensity)
    s = np.asarray(sigma)

    wl_grid = np.linspace(wl.min(), wl.max(), 500)
    model_grid = model_fn(wl_grid, theta)

    fig, ax = plt.subplots()
    ax.errorbar(wl, I, yerr=s, fmt="o", label="data")
    ax.plot(wl_grid, model_grid, label="best-fit model")

    ax.set_xlabel("Wavelength [m]")
    ax.set_ylabel("Intensity [arb.]")
    ax.legend()
    fig.tight_layout()
    outfile = Path(outfile)
    fig.savefig(outfile)
    plt.close(fig)


def plot_traces(chain: np.ndarray, param_names, outfile: str | Path) -> None:
    """Traces plots for manual MCMC chain"""
    n_steps, n_params = chain.shape
    fig, axes = plt.subplots(n_params, 1, sharex=True, figsize=(6, 2.5 * n_params))

    if n_params == 1:
        axes = [axes]

    x = np.arange(n_steps)

    for i, ax in enumerate(axes):
        ax.plot(x, chain[:, i])
        ax.set_ylabel(param_names[i])
    axes[-1].set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def plot_histograms(chain: np.ndarray, param_names, outfile: str | Path) -> None:
    """Posterior histograms for parameters."""
    _, n_params = chain.shape
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3))

    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.hist(chain[:, i], bins=40, density=True)
        ax.set_xlabel(param_names[i])
        ax.set_ylabel("Posterior density")

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
