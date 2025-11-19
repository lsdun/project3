"""This module contains the emcee for off-the-shelf MCMC"""

from __future__ import annotations

from dataclasses import dataclass
import emcee
import numpy as np


@dataclass
class EmceeResult:
    """Container for emcee result"""

    chain: np.ndarray # (n_walkers, n_steps, n_params)
    log_prob: np.ndarray # (n_walkers, n_steps)
    acceptance_fraction: np.ndarray

def run_emcee(
    log_posterior: Callable[[Sequence[float]], float],
    theta_start: Sequence[float],
    n_walkers: int = 32,
    n_steps: int = 10_000,
    scatter: float = 1e-2,
    random_seed: int | None = None,
) -> EmceeResult:
    """
    Runs emcee ensemble sampler

    Args:
    
    log_posterior (callable):
        returns log postrior for a given theta
    theta_start (sequence of float):
        Central starting position for walkers
    n_walkers (int):
        Number of walkers
    n_steps (int):
        Number of production steps
    scatter (float):
        Relative scatter of initial walker positions
    random_seed (int):
        Seed for RNG

    Returns:
    
    EmceeResult
        Output chain and diagnostics
    """
    theta_start = np.asarray(theta_start, dtype=float)
    n_params = theta_start.size

    if n_walkers < 2 * n_params:
        raise ValueError("n_walkers should be at least 2 * n_params.")

    rng = np.random.default_rng(random_seed)
    pos0 = theta_start + scatter * rng.normal(size=(n_walkers, n_params))

    def log_prob(theta: np.ndarray) -> float:
        return log_posterior(theta)

    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_prob)
    sampler.run_mcmc(pos0, n_steps, progress=False)

    return EmceeResult(
        chain=sampler.get_chain(),  # (n_walkers, n_steps, n_params)
        log_prob=sampler.get_log_prob(),
        acceptance_fraction=sampler.acceptance_fraction,
    )
