#~ovn!
"""BODE on a synthetic example.

 ::math::
    f(x) = (4(1 - \sin(x + 8\exp(x - 7))) - 10) / 5 + \sigma(x) \epsilon
    \epsilon \sim \mathcal{N}(0, 1)
    
Author:
    Ilias Bilionis
    Piyush Pandita
    R Murali Krishnan
    
Date:
    09/04/2023
    
"""

import os
import sys
import numpy as np
from pyDOE2 import lhs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bode._quadrature import Quadrature
from bode._eig_sampler import AnalyticalEIGSampler

class SynethicProblem(object):
    def __init__(self, n=3, dim=1, n_true=100_000):
        self.sigma = lambda x: 0.5
        
        self.n = n
        self.dim = dim
        self.X_init = lhs(dim, samples=n, criterion='center')
        self.Y_init = np.array([self(x) for x in self.X_init])[:, None]
        
        self.n_true = n_true
        self.X_true = lhs(dim, samples=n_true, criterion='center')
        self.Y_true = np.array([self(x) for x in self.X_true])[:, None]
        
        self.true_mean = np.mean(self.Y_true)

    def __call__(self, x):
        x = 6 * x
        return (4 * (1. - np.sin(x[0] + 8 * np.exp(x[0] - 7.))) \
                -10.) / 5. + self.sigma(x[0]) * np.random.randn()

        
if __name__=='__main__':
    print("~ovn!")
    np.random.seed(1333)
    
    problem = SynethicProblem()
    print('true E[f(x)]: ', problem.true_mean)
    
    quadrature = Quadrature(500, problem.dim)
    
    eig_sampler = AnalyticalEIGSampler(problem.X_init, problem.Y_init)
    
    eig_sampler.step()
    
    # import pdb; pdb.set_trace()
    print("~~Ovn!")
    