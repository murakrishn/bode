#~ovn!
"""Gaussian process models

Author:
    Piyush Pandita
    R Murali Krishnan
    
Date:
    09/07/2024
    
"""


import GPy
import emcee
import numpy as np

from ._core import GammaPrior, BetaPrior, JeffreysPrior, Dataset

class BayesianGP(GPy.models.GPRegression):
    """Fully Bayesian probabilistic surrogate model"""
    def __init__(self, dataset: Dataset, kernel=None, noisy=False):
        
        self._Kff = GPy.kern.RBF(dataset.input_dim, ARD=True) if kernel is None else kernel
        super().__init__(dataset.X, dataset.Y, self._Kff)
        
        self.scale_prior = GammaPrior(a=8.0, scale=1.0)
        self.ell_prior = BetaPrior(a=2, b=5)
        self.noisy = noisy
        self.nugget = 1e-3
        if self.noisy:
            self._theta_dim = dataset.input_dim + 2
            self.noise_prior = JeffreysPrior()
        else:
            self._theta_dim = dataset.input_dim + 1
            self.noise_prior = None
        
    def lnprob(self, theta):
        if np.any(theta < 0.0):
            return -np.inf
        return self._get_log_likelihood(theta) + self._get_log_prior(theta)
    
    def _get_log_likelihood(self, theta):
        self.kern.variance = theta[0]
        if self.noisy:
            self.kern.lengthscale = theta[1:-1]
            self.likelihood.variance = theta[-1]**2
        else:
            self.kern.lengthscale = theta[1:]
            self.likelihood.variance = self.nugget**2
        return self.log_likelihood()
    
    def _get_log_prior(self, theta):
        scale_log_prior = self.scale_prior(theta[0])
        ell_log_prior = 0.0
        for j in range(self.input_dim):
            ell_log_prior += self.ell_prior(theta[j+1])
            
        if self.noisy:
            noise_log_prior = self.noise_prior(theta[-1])
            log_priors = scale_log_prior + ell_log_prior + noise_log_prior
        else:
            log_priors = scale_log_prior + ell_log_prior
        return log_priors
    
    
def sample_gp_q_data(model_q_d: BayesianGP, mcmc_chains=10, mcmc_steps=500, 
                     mcmc_burn=100, mcmc_thin=30, mcmc_model_avg=50):
    """Sample posterior of the hyperparameters of a BayesianGP conditioned on data."""
    input_dim = model_q_d.input_dim
    theta_dim = model_q_d._theta_dim
    sampler = emcee.EnsembleSampler(
        mcmc_chains, model_q_d._theta_dim, model_q_d.lnprob)
    init_pos = [np.hstack([model_q_d.scale_prior.sample(size=1), 
                            model_q_d.ell_prior.sample(size=input_dim), 
                            model_q_d.noise_prior.sample(size=1) if model_q_d.noisy else []]) 
                for _ in range(mcmc_chains)]
    sampler.run_mcmc(initial_state=init_pos, nsteps=mcmc_steps)
    print(">... acceptance ratio(s): {0}".format(sampler.acceptance_fraction))
    samples_thin = sampler.chain[:, mcmc_burn::mcmc_thin, :].reshape((-1, theta_dim))
    return sampler, samples_thin

def make_fixed_gp_model(X, Y, kernel, params, noisy=False, nugget=1e-3):
    """Make a GP model with fixed hyperparameters"""
    model = GPy.models.GPRegression(X, Y, kernel)
    model.kern.variance.fix(params[0])
    if noisy:
        model.kern.lengthscale.fix(params[1:-1])
        model.likelihood.variance.fix(params[-1]**2)
    else:
        model.kern.lengthscale.fix(params[1:])
        model.likelihood.variance.fix(nugget**2)
    return model