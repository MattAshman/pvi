from abc import ABC, abstractmethod
from torch import nn


# =============================================================================
# Base approximating likelihood class
# =============================================================================


class ApproximatingLikelihoodFactor(ABC):
    
    # TODO:
    # - deal with the initialisation t(θ) = 1
    
    def __init__(self, log_coefficient, natural_parameters):
        
        # Set leading coefficient and natural parameters
        self.log_coefficient = log_coefficient
        self.natural_parameters = natural_parameters
    
    
    @abstractmethod
    def __call__(self, thetas):
        """
        Computes the log-value of θ under the factor t, i.e. log t(θ).
        """
        pass
    
    
    @abstractmethod
    def compute_refined_factor(self, q, q_):
        """
        Computes the natural parameters and leading coefficient of the
        approximating likelihood term **t** given by
        
            t(θ) = q(θ) / q_(θ) t_(θ)
            
        where **t_** is the approximating likelihood term corresponding
        to **self**.
        """
        pass
    


# # =============================================================================
# # Base exponential family distribution
# # =============================================================================


# class ExponentialFamilyDistribution(ABC, nn.Module):
    
#     def __init__(self,
#                  parameters=None,
#                  natural_parameters=None
#                  is_trainable=False):
        
#         # Initialise using either standard parameters or natural parameters
#         if not (parameters is None):
#             self.a
    
#     @property
#     @abstractmethod
#     def torch_dist_class(self):
#         raise NotImplementedError
    
    
#     def log_prob(self, thetas):
#         """
#         Computes the log-probability of the distribution log q(θ).
#         """
#         return self.q.log_prob(thetas)
    
    
#     def kl_divergence(self, q_):
#         return self.q.kl_divergence(q_)
    
    
#     def get_natural_parameters(self):
#         return self.natural_parameters
    
    
#     def sample(self):
#         return self.q.sample()
    
    
#     def
        
            