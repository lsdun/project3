"""Command-line interface for blackbody_mcmc"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_data
from .diagnostics import autocorrelation, gelman_rubin
from .mcmc_library import run_emcee
from .mcmc_manual import run_mcmc
from .model import log_posterior, model_intensity
from .plotting import plot_data_and_model, plot_histograms, plot_traces


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Fit blackbody data using Bayesian MCMC"
    )
    parser.add_argument(
        "data_file",
        help="CSV file with wavelength,intensity columns",
    )
    parser.add_argument(
        "--n-steps-manual",
        type=int,
        default=50_000,
        help="Number of steps for manual Metropolis sampler",
    )
    parser.add_argument(
        "--n-steps-emcee",
        type=int,
        default=10_000,
        help="Number of steps for emcee sampler",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to store plots and outputs",
    )
    parser.add_argument(
        "--proposal-scale",
        type=float,
        nargs=2,
        metavar=("SIGMA_T", "SIGMA_A"),
        default=(50.0, 0.1),
        help="Proposal std devs for (T, A) in manual sampler",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for blackbody-mcmc script"""
    args = parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    wl, I, sigma = load_data(args.data_file)

    # Central guess: Wien's law-ish guess for temperature
    # lambda_max * T ≈ 2.9e-3 m K  -> rough estimate
    lambda_peak = wl[np.argmax(I)]
    T_guess = 2.9e-3 / lambda_peak
    A_guess = 1e-20  # arbitrary scale
    theta0 = np.array([T_guess, A_guess])

    def lp(theta):
        return log_posterior(theta, wl, I, sigma)

    # ---------- Manual MCMC ----------
    print("Running manual Metropolis MCMC...")
    manual = run_mcmc(
        theta0,
        log_posterior=lp,
        n_steps=args.n_steps_manual,
        proposal_scale=np.asarray(args.proposal_scale),
    )
    burn = int(0.2 * args.n_steps_manual)
    chain_manual = manual.chain[burn:]

    print(f"Manual acceptance rate: {manual.accept_rate:.3f}")

    # ---------- emcee ----------
    print("Running emcee ensemble sampler...")
    emcee_res = run_emcee(
        log_posterior=lp,
        theta_start=theta0,
        n_walkers=32,
        n_steps=args.n_steps_emcee,
    )
    burn_e = int(0.2 * args.n_steps_emcee)
    chain_emcee = emcee_res.chain[:, burn_e:, :].reshape(-1, 2)


















