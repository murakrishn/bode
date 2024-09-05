#~ovn!
"""Quadratures are important.

Author:
    Ilias Bilionis
    Piyush Pandita
    R Murali Krishnan
    
Date:
    09/04/2023
    
"""

import numpy as np
from pyDOE2 import lhs

class Quadrature(object):
        
        def __init__(self, nq, dim, model='lhs'):
            self.nq = nq
            self.dim = dim
            if model == 'lhs':
                self.quad_points = lhs(dim, nq)
                self.quad_points_weight = np.ones(nq)
            else:
                raise NotImplementedError()