from abc import ABC, abstractmethod
from .base import ExponentialFamilyFactor
from .exponential_family_distributions import *

import math

import torch


# =============================================================================
# Mean field Gaussian factor
# =============================================================================
    

class MeanFieldGaussianFactor(ExponentialFamilyFactor):
    
    
    def __init__(self, natural_parameters):
        super().__init__(natural_parameters=natural_parameters)
        
        self.distribution_class = MeanFieldGaussianDistribution
    
    
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
    
    
    def nat_from_dist(self, q):
        
        loc = q.loc.detach()
        scale = q.scale.detach()
        
        std = {
            "loc" : loc,
            "scale" : scale
        }
        
        return self.distribution_class._nat_from_std(std)
    
    
    def dist_from_nat(self, nat):
        
        std = self.distribution_class._std_from_nat(nat)
        dist = torch.distributions.Normal(**std)
        
        return dist

    
    
# =============================================================================
# Multivariate Gaussian factor
# =============================================================================


class MultivariateGaussianFactor(ExponentialFamilyFactor):
    
    
    def __init__(self, natural_parameters):
        
        super().__init__(natural_parameters)
        
        self.distribution_class = MultivariateGaussianDistribution
    
    
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
    
    
    def nat_from_dist(self, q):
        
        loc = q.loc.detach()
        cov = q.covariance_matrix.detach()
        
        std = {
            "loc" : loc,
            "covariance_matrix" : cov
        }
        
        return self.distribution_class._nat_from_std(std)
    
    
    def dist_from_nat(self, nat):
        
        std = self.dist_class._std_from_nat(nat)
        dist = torch.distributions.MultivariateNormal(**std)
        
        return dist
        