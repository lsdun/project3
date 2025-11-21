Author: Lauren Sdun, Julia Jones, Julia Baumgarten
# Project 3

**Overall layout**

```python
blackbody_mcmc/
    pyproject.toml / setup.cfg
    src/
        blackbody_mcmc/
            __init__.py
            data.py
            model.py
            mcmc_manual.py
            mcmc_library.py
            diagnostics.py
            plotting.py
            cli.py          # main script entry point
    tests/
        test_model.py
        test_mcmc.py
```

**Physics and statistics**

model.py
- planck_lambda(wavelength, T): Compute spectral radiance

- log_prior(theta): e.g. $\theta = (T,A)$ with A overall amplitude
T > 0, maybe uniform or log-uniform between e.g. 1000 K and 5000 K
A > 0 with some broad prior

- log_likelihood(theta, wavelength, intensity, sigma): 

- log_posterior(theta, ...) = log_prior(theta) + log_likelihood(...)


data.py
- load_data(path): Load experimental data and returns arrays lambda, I, sigma_I


**Manual MCMC implementation**

mcmc_manual.py:
- propose(theta_current, proposal_widths): Draw new parameters from e.g. Gaussian random walk
- metropolis_step(theta_current, log_post_current, ...): Call propose, Compute log_post_proposed, Accept/reject with Metropolis–Hastings rule
- run_mcmc(theta0, n_steps, ...): Loop over steps call metropolis_step, Store chain and log-posterior values and acceptance rate, return: chain shape (n_steps, n_params) and log_post and acceptance_rate


**Off-the-shelf MCMC**

mcmc_library.py: log_prob_emcee(theta, ...) just wrapper around log_posterior
run_emcee(initial_positions, n_steps, ...): Use emcee.EnsembleSampler
Return chain (n_walkers, n_steps, n_params) and log_prob
Helper: get_emcee_summary(chain) compute posterior means, medians, credible intervals


**Convergence and Diagnostics**

diagnostics.py:
autocorrelation_time(chain) estimate integrated autocorrelation time or use library helper
acceptance_rate(chain, accepted_mask) or from manual MCMC

Formal numeric diagnostic (per assignment):
Gelman–Rubin R or similar
gelman_rubin(chains) need multiple chains with different starting points
effective_sample_size(chain, tau)


**Plots**

plotting.py:
Data + best-fit curve
- plot_data_and_bestfit(wavelength, intensity, sigma, theta_best, outfile): Scatter (data) + line (Planck curve at posterior mean/median). Maybe overlay multiple posterior samples for uncertainty band


Trace plots
- plot_traces(chain, outfile_prefix): Parameter vs iteration for each parameter manual MCMC and library method

Posterior distributions
- plot_posteriors(chain, outfile_prefix): 1D histograms/KDEs for T and A and corner plots (2D contour for T vs A)

Convergence diagnostics
- plot_autocorrelation(chain, outfile)

Comparison of methods
- plot_method_comparison(...): bar chart of T_est_manual vs T_est_emcee with error bars or something along those lines

**CLI/main**

Command Line Integration options:
--data-file
--n-steps-manual, --n-steps-emcee
--output-dir for plots

```python
def main():
    args = parse_args()
    wl, I, sigma = load_data(args.data_file)
    # run manual MCMC
    manual_results = run_mcmc(...)
    # run emcee
    emcee_results = run_emcee(...)
    # compute diagnostics
    ...
    # make plots
    ...
    # print summary to stdout
```


***Bonus Plot: Joint Posterior***
In order to obtain the joint posterior plot, which creates a 2D hexbin density plot of T vs. A the following command needs to be input into the terminal:


```python
PYTHONPATH=src python - << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from blackbody_mcmc.data import load_data
from blackbody_mcmc.model import log_posterior
from blackbody_mcmc.mcmc_library import run_emcee

# ----- Load data -----
wl, I, sigma = load_data("blackbody.csv")

# ----- Define posterior for emcee -----
def lp(theta):
    return log_posterior(theta, wl, I, sigma)

# Initial guess for (T, A) using Wien's law
lambda_peak = wl[np.argmax(I)]
T_guess = 2.9e-3 / lambda_peak
A_guess = 0.63

# ----- Runs a moderately long emcee chain -----
res = run_emcee(lp, [T_guess, A_guess], n_walkers=32, n_steps=6000)
burn = 1000
chain = res.chain[:, burn:, :].reshape(-1, 2)
T = chain[:, 0]
A = chain[:, 1]

# ----- Plots joint posterior -----
outdir = Path("results")
outdir.mkdir(exist_ok=True)

plt.figure(figsize=(5,4))
plt.hexbin(T, A, gridsize=40)
plt.xlabel("T [K]")
plt.ylabel("A")
plt.colorbar(label="posterior density")
plt.tight_layout()
plt.savefig(outdir / "joint_posterior_TA.png")
plt.close()
print("Saved joint posterior to", outdir / "joint_posterior_TA.png")
EOF
```

