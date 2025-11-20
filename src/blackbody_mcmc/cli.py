"""Command-line interface for blackbody_mcmc"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_data
from .diagnostics import autocorrelation, gelman_rubin
from .mcmc_library import run_emcee, EmceeResult
from .mcmc_manual import run_mcmc, MCMResult
from .model import log_posterior, model_intensity
from .plotting import plot_data_and_model, plot_histograms, plot_traces


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Fit blackbody data using Bayesian MCMC"
    )
    parser.add_argument(
        "data_file",
        help="CSV file with wavelength,intensity[,sigma] columns"
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
        default=(5.0, 0.01),
        help="Proposal std devs for (T, A) in manual sampler",
    )
    return parser.parse_args()


#Inital parameter guess
def initial_guess(wl: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Return rough initial (T, A) guess using Wien's law."""
    lambda_peak = wl[np.argmax(intensity)]
    T_guess = 2.9e-3 / lambda_peak
    A_guess = 1e-20
    return np.array([T_guess, A_guess])



# Hasting Metropolis sampler wrappers
def run_manual_sampler(theta0: np.ndarray, logp_fn, n_steps: int,
                       proposal_scale: Tuple[float, float], random_seed: int | None = None
                       ) -> Tuple[MCMCResult, np.ndarray]:
    result = run_mcmc(theta0, log_posterior=logp_fn, n_steps=n_steps,
                      proposal_scale=np.asarray(proposal_scale), random_seed=random_seed)
    burn = int(0.2 * n_steps)
    postburn_chain = result.chain[burn:]
    print(f"Manual acceptance rate: {result.accept_rate:.3f}")
    return result, postburn_chain


#emcee sampler wrappers
def run_emcee_sampler(
    theta0: np.ndarray,
    logp_fn,
    n_steps: int,
    n_walkers: int = 32,
) -> tuple[EmceeResult, np.ndarray, int]:
    """
    Run emcee ensemble sampler and return (EmceeResult, flattened_postburn_chain, burn).
    """
    emcee_res = run_emcee(
        log_posterior=logp_fn,
        theta_start=theta0,
        n_walkers=n_walkers,
        n_steps=n_steps,
    )
    burn = int(0.2 * n_steps)
    n_walkers_run, n_steps_run, n_params = emcee_res.chain.shape
    if n_steps_run - burn <= 0:
        flat = np.empty((0, n_params))
    else:
        flat = emcee_res.chain[:, burn:, :].reshape(-1, n_params)
    return emcee_res, flat, burn



#Skip empty mc chains
def summarize_chain(chain: np.ndarray, label: str) -> None:
    if chain.size == 0:
        print(f"empty chain — skipped")
        return
    mean = chain.mean(axis=0)
    p16, p50, p84 = np.percentile(chain, [16, 50, 84], axis=0)
    print(f"Normal distribution values")
    print(f"  mean T = {mean[0]:.2f} K, mean A = {mean[1]:.3e}")
    print(f"  T (16/50/84%) = {p16[0]:.2f}, {p50[0]:.2f}, {p84[0]:.2f} K")
    print(f"  A (16/50/84%) = {p16[1]:.3e}, {p50[1]:.3e}, {p84[1]:.3e}")




# Main script
def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    wl, I, sigma = load_data(args.data_file)

    # Initial guess
    theta0 = initial_guess(wl, I)

    # Posterior wrapper
    logp = lambda th: log_posterior(th, wl, I, sigma)

    # Run samplers
    manual_res, chain_manual = run_manual_sampler(theta0, logp,
                                                  args.n_steps_manual, args.proposal_scale)

    emcee_res, chain_emcee, burn_e = run_emcee_sampler(theta0, logp, args.n_steps_emcee)

    param_names = ["T [K]", "A"]

    # Diagnostics: manual
    if chain_manual.size > 0:
        acf_T = autocorrelation(chain_manual[:, 0])
        tau_int = 0.5 + acf_T[1:].sum()
        print(f"Estimated integrated autocorrelation time (manual, T): {tau_int:.1f} steps")
        try:
            subchains = np.array_split(chain_manual, 4)
            r_hat = gelman_rubin(subchains)
            print(f"R-hat (manual): T={r_hat[0]:.3f}, A={r_hat[1]:.3f}")
        except Exception as e:
           print("Could not compute Gelman-Rubin for manual chain:", e)
    else:
        print("Manual chain empty")

     print("emcee acceptance fractions (per walker):", getattr(emcee_res, "acceptance_fraction", None))

    # Summaries
    summarize_chain(chain_manual, "Manual MCMC")
    summarize_chain(chain_emcee, "emcee")

    # Plots
    print("Creating plots...")
    if chain_emcee.size > 0:
        theta_best = np.median(chain_emcee, axis=0)
    elif chain_manual.size > 0:
        theta_best = np.median(chain_manual, axis=0)
    else:
        theta_best = theta0
        print("Both chains empty — using initial guess for plotting.")

    plot_data_and_model(wl, I, sigma, theta=theta_best,
                        model_fn=model_intensity, outfile=outdir / "data_and_model.png")

    try:
        plot_traces(manual_res.chain, param_names=param_names, outfile=outdir / "manual_traces.png")
    except Exception as e:
        print("Could not create trace plot:", e)

    try:
        plot_histograms(chain_manual, param_names=param_names, outfile=outdir / "manual_posteriors.png")
    except Exception as e:
        print("Could not create manual posterior histograms:", e)

    try:
        if chain_emcee.size > 0:
            plot_histograms(chain_emcee, param_names=param_names, outfile=outdir / "emcee_posteriors.png")
        else:
            print("Skipping emcee posterior histogram (empty chain).")
    except Exception as e:
        print("Could not create emcee posterior histograms:", e)

    print(f"\nDone. Results written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

   
   













