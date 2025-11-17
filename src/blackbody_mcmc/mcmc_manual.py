"""This module contains the Metropolis-Hastings MCMC sampler"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LogPosteriorFn = Callable[[Sequence[float]], float]

@dataclass
class MCMCResult:
    """Container for manual MCMC output"""

    chain: np.ndarray # (n_steps, n_params)
    log_post: np.ndarray # (n_steps,)
    accept_rate: float

def metropolis_step(
    theta_current: np.ndarray,
    logp_current: float,
    log_posterior: LogPosteriorFn,
    proposal_scale: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, bool]:
    """
    Single Metropolis-Hastings step with Gaussian proposal

    Args:
    
    theta_current (ndarray):
        Current parameter vector
    logp_current (float):
        Log posterior evaluated at theta current
    log_posterior (callable):
        returns log postrior for a given theta
    proposal_scale (ndarray):
        Std dev of proposal for each parameter
    rng (Generator):
        Random number generator

    Returns:

    theta_new (ndarray)
        New parameter vector
    logp_new (float)
        Log posterior at new position
    accepted (boolean)
        Whether the proposal was accepted
    """

    # propose a new sample,
    # proposal distribution is Gaussian around the current point
    # add Gaussian noise to each parameter
    proposal = theta_current + rng.normal(scale=proposal_scale, size=theta_current.size)
    # compute log posterior at proposed parameter vector
    logp_proposed = log_posterior(proposal)

    if not np.isfinite(logp_proposed):
        return theta_current, logp_current, False

    # compute acceptance probability a
    # accept move if log(random) < log_posterior_difference
    log_alpha = logp_proposed - logp_current
    if np.log(rng.uniform()) < log_alpha:
        return proposal, logp_proposed, True # accept
    return theta_current, logp_current, False # reject

def run_mcmc(
    theta0: Sequence[float],
    log_posterior: LogPosteriorFn,
    n_steps: int = 50_000,
    proposal_scale: Sequence[float] | float = 0.01,
    random_seed: int | None = None,
) -> MCMCResult:
    """
    Runs single-chain Metropolis-Hastings sampler

    Args:
    theta0 (sequence of float)
        first theta to be used 
    log_posterior (callable)
        returns log postrior for a given theta
    n_steps (int)
        Number of MCMC steps
    proposal_scale (float or sequence):
        Gaussian proposal widths
    random_seed (int)
        Seed for RNG

    Returns:
    
    MCMCResult
        Chain, log-posteriors, acceptance rate
    """
    
    # convert initial parameters to array
    # record number of parameters
    rng = np.random.default_rng(random_seed)
    theta0 = np.asarray(theta0, dtype=float)
    n_params = theta0.size

    if np.isscalar(proposal_scale):
        proposal_scale = np.repeat(float(proposal_scale), n_params)
    proposal_scale = np.asarray(proposal_scale, dtype=float)

    chain = np.zeros((n_steps, n_params), dtype=float) # store vector
    log_post = np.zeros(n_steps, dtype=float) # store log posterior
    accepted = np.zeros(n_steps, dtype=bool) # record whether accepted or not

    # initialize chain at theta0 and compute log posterior once
    theta = theta0.copy()
    logp = log_posterior(theta)

    """
    For each iteration:
    - Propose and possibly accept a new state using metropolis_step
    - Store the new/current state and its log posterior
    - Record whether the proposal was accepted
    - Compute fraction of steps that resulted in accepted proposals (btwn 0 and 1)
    - Return result containing full chain, log-posteriros, and acceptance rate. 
    """
    for i in range(n_steps):
        theta, logp, acc = metropolis_step(
            theta, logp, log_posterior, proposal_scale, rng
        )
        chain[i] = theta
        log_post[i] = logp
        accepted[i] = acc

    accept_rate = accepted.mean()
    return MCMCResult(chain=chain, log_post=log_post, accept_rate=accept_rate)
