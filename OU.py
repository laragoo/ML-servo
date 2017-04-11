import random
import numpy as np

class OU(object):
    '''
    Ornstein-Uhlenbeck Process: stoachstic process that describes the velocity
    of a massive Brownian particlue under the influence of friction; stationary
    Gauss-Markov process (mean-reverting)
    '''
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)
