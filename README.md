Author: Lauren Sdun, Julia Jones, Julia Baumgarten
# Project 3
 Markov Chain Monte Carlo (MCMC) methods to estimate the temperature and amplitude parameters of a blackbody radiation model. This package runs MCMC samplers, performs convergence diagnostics, and generates plots to estimate the posterior distribution of the model parameters.
## Installation

1. Clone the repository:

```bash
git clone https://github.com/lsdun/project3.git
cd project3
```

2. Install the package using `pip`:
```bash
pip install e
```

## Running the Simulation

You can run the simulation from the command line using the installed CLI:
```bash
blackbody-mcmc data/blackbody.csv --n-steps-manual 5000 --n-steps-emcee 10000
```

### Arguments
`--n-steps-manual`: Number of steps in chain for manual Metropolis-Hastings sampler 

`--n-steps-emcee`: Number of steps in chain for emcee sampler

## Output:
This package outputs the Gelman Rubin statistic for both MCMC methods as well as the autocorrelation times. 
Additionally, it outputs the normal distribution values of the paremeters. The plots generated are the raw data plotted against the model, trace plots of the evolution of each parameter, and posterior histograms for each MCMC method.








