from abc import ABC, abstractmethod
from .base import ExponentialFamilyFactor

import math

import torch


# =============================================================================
# Mean field Gaussian factor
# =============================================================================
    

class MeanFieldGaussian(ExponentialFamilyFactor):
    
    
    def __init__(self, natural_parameters):
        super().__init__(natural_parameters=natural_parameters)
    
    
    def log_h(self, thetas):
        """
        Returns the value of log h(θ) for the MeanFieldFactor class. For
        a mean-field multivariate Gaussian, log h(θ) = 0.
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        return torch.zeros(size=thetas.shape[:1])
    
    
    def npf(self, thetas):
        
        np1 = self.natural_parameters["np1"]
        np2 = self.natural_parameters["np2"]
        
        npf = torch.mv(thetas, np1)
        npf = npf + torch.mv(thetas ** 2, np2)
        
        return npf
    
    
    def np_from_distribution(self, q):
        
        # Exctract loc and scale from torch.distribution
        loc = q.loc.detach()
        scale = q.scale.detach()
        
        assert loc.shape == scale.shape
        assert len(loc.shape) == 1
        
        # Compute natural parameters for multivariate normal
        np = {
            "np1" : loc / scale ** 2,
            "np2" : - 0.5 * scale ** -2
        }
        
        return np
    
    
    def distribution_from_np(self, np):
        
        np1 = np["np1"]
        np2 = np["np2"]
        
        loc = - 0.5 * np1 / np2
        scale = (- 0.5 / np2) ** 0.5
        
        dist = torch.distributions.Normal(loc=loc, scale=scale)
        
        return dist

    
    
# =============================================================================
# Multivariate Gaussian factor
# =============================================================================


class MultivariateGaussian(ExponentialFamilyFactor):
    
    
    def __init__(self, natural_parameters):
        
        super().__init__(natural_parameters)
    
    
    def log_h(self, thetas):
        """
        Returns the value of log h(θ) for the MultivariateGaussian class. For
        a multivariate Gaussian, log h(θ) = 0.
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        return torch.zeros(size=thetas.shape[:1])
    
    
    def npf(self, thetas):
        
        np1 = self.natural_parameters["np1"]
        np2 = self.natural_parameters["np2"]
        
        npf = torch.mv(thetas, np1)
        npf = npf + torch.sum(thetas * torch.mm(thetas, np2), dim=1)
        
        return npf
    
    
    def np_from_distribution(self, q):
        """
        Takes a torch.distribution **q**, assumed to be in the EF
        and extracts its leading coefficient and natural parameters.
        """
        
        # Exctract loc and scale from torch.distribution
        loc = q.loc.detach()
        scale_tril = q.scale_tril.detach()
        
        assert loc.shape[0] == scale_tril.shape[0]
        assert len(loc.shape) == 1
        assert len(scale_tril.shape) == 2
        
        np1 = torch.cholesky_solve(loc[:, None], scale_tril)[:, 0]
        np2 = -0.5 * torch.cholesky_inverse(scale_tril, upper=False)
        
        # Compute natural parameters for multivariate normal
        np = {
            "np1" : np1,
            "np2" : np2
        }
        
        return np
    

    def distribution_from_np(self, np):
        """
        Takes a dictionary of natural parameters **np** and returns a
        torch.distribution defined by these natural parameters.
        """
        
        np1 = np["np1"]
        np2 = np["np2"]
        
        loc = - 0.5 * torch.solve(np1[:, None], np2).solution[:, 0]
        scale_tril = torch.linalg.cholesky(- 2 * np2)
        
        dist = torch.distributions.MultivariateNormal(loc=loc,
                                                      scale_tril=scale_tril)
        
        return dist
        
        