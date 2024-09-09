#~ovn!
"""Expected Information Gain (EIG) sampler for E[f(x)].

Author:
    Ilias Bilionis
    Piyush Pandita
    R Murali Krishnan
    
Date:
    09/04/2023
    
"""

import numpy as np
from pyDOE2 import lhs

from ._gp import BayesianGP, sample_gp_q_data
from ._core import Dataset

class AnalyticalEIGSampler(object):
    
    @property
    def input_dim(self):
        return self.dataset.input_dim
    
    def __init__(self, X, Y,
                 mcmc_chains=10, mcmc_steps=500, mcmc_burn=100,
                 mcmc_thin=30, mcmc_model_avg=50):
        self.dataset = Dataset(X, Y)
        self.mcmc_chains = mcmc_chains
        self.mcmc_steps = mcmc_steps
        self.mcmc_burn = mcmc_burn
        self.mcmc_thin = mcmc_thin
        self.mcmc_model_avg = mcmc_model_avg
        self.model = BayesianGP(self.dataset)
        self._sampler, self.theta_q_d = sample_gp_q_data(self.model, mcmc_chains, mcmc_steps, mcmc_burn, mcmc_thin, mcmc_model_avg)
        
    def get_mu_sigma(self):
        pass
        
    def step(self, num_designs=1_000, **kwargs):
        """Step through one iteration of the sample."""
        
        X_design = lhs(self.input_dim, num_designs, criterion="center")
        
        import pdb; pdb.set_trace()
        pass