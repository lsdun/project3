"""This module contains the emcee for off-the-shelf MCMC"""

from __future__ import annotations
import numpy as np
import emcee
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EmceeResult:
    """Container for emcee result"""

    chain: np.ndarray # (n_walkers, n_steps, n_params)
    log_prob: np.ndarray # (n_walkers, n_steps)
    acceptance_fraction: np.ndarray

def run_emcee(
    log_posterior: Callable[[np.ndarray], float],
    theta_start: np.ndarray,
    n_walkers: int = 32,
    n_steps: int = 10000,
    scatter: Optional[float] = None,  
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
    ndim = len(theta_start)

    p0 = theta_start + 1e-3 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)

    sampler.run_mcmc(p0, n_steps, progress=True)

    chain_raw = sampler.get_chain()
    chain = np.transpose(chain_raw, (1, 0, 2))

    logp_raw = sampler.get_log_prob()
    log_prob = np.transpose(logp_raw, (1, 0))

    return EmceeResult(
        chain=chain,
        log_prob=log_prob,
        acceptance_fraction=np.array(sampler.acceptance_fraction),
    )
