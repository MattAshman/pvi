from abc import ABC, abstractmethod
from .base import ApproximatingLikelihoodFactor

import math

import torch


# =============================================================================
# Base class for approximating likelihood factors of the exponential family
# =============================================================================


class ExponentialFamilyFactor(ApproximatingLikelihoodFactor):
    """
    Base class for exponential family (EF) approximating likelihood factors.
    The exponential family is made up of distributions which can be written
    in the form
    
        p(x | θ) = h(x) exp(ν.T f(x) + A(ν)).
    
    For a list of exponential family distributions see:
        https://en.wikipedia.org/wiki/Category:Exponential_family_distributions
        https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions
    
    
    TODO: Write children classes for the following members of the EF
    
        - MultivariateGaussian
        - Laplace
        - Dirichlet
        - Multinomial
        
    There are many more members but these are especially useful for our
    applications.
    """
    
    def __init__(self, log_coefficient, natural_parameters):
        
        super().__init__(log_coefficient=log_coefficient,
                         natural_parameters=natural_parameters)
      
    
    def compute_refined_factor(self, q1, q2):
        """
        Computes the log-coefficient and natural parameters of the
        approximating likelihood term **t** given by
        
            t(θ) = q1(θ) / q2(θ) t_(θ)
            
        where **t_** is the approximating likelihood term corresponding
        to **self**. Note that the log-coefficient computed here includes
        the normalising constants of the q-distributions as well as the
        coefficient of t_.
        """
        
        # Convert distributions to log-coefficients and natural parameters
        log_coeff1, np1 = self.log_coeff_and_np_from_distribution(q1)
        log_coeff1, np2 = self.log_coeff_and_np_from_distribution(q2)
        
        # Log coefficient and natural parameters of refined factor
        log_coeff = log_coeff1 - log_coeff2 + self.log_coeff
        natural_parameters = {}
        
        # Compute natural parameters of the new t-factor
        for name, natural_parameter in self.natural_parameters.items():
            natural_parameters[name] = np1[name] - np2[name] + np
            
        # Create and return refined t of the same type
        t = type(self).__init__(log_coeff, natural_parameters)
        
        return t
    

    def __call__(self, thetas):
        """
        Returns the value of log t(θ) where
        
            log t(θ) = log c + log h(x) + ν.T f(x)
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        
        # Compute f matrix (shape (N, K)) and ν vector (shape (K,))
        f = self.f(thetas)
        np = self.natural_parameter_vector()
        
        # Compute log h(θ), (shape (N,))
        log_h = self.log_h(thetas)
        
        # Compute inner product ν.T f(x) and log t(θ) (both shape (N,))
        npf = torch.mv(f, np)
        log_t = self.log_coefficient + log_h + npf
        
        return log_t
    
    
    @abstractmethod
    def log_h(self, thetas):
        """
        Returns the value of log h(θ). Input **thetas** is assumed to be
        a torch.tensor of shape (N, D) where N is the batch dimension and
        D is the dimension of the distribution.
        """
        pass
    
    
    @abstractmethod
    def f(self, thetas):
        """
        Computes f(θ) vector for this member of the exponential family.
        """
        pass
    
    
    @abstractmethod
    def natural_parameter_vector(self):
        """
        Rearranges NPs to ν vector for this member of the exponential family.
        """
        pass
    
    
    @abstractmethod
    def log_coeff_and_np_from_distribution(self, q):
        """
        Takes a torch.distribution **q**, assumed to be in the EF
        and extracts its leading coefficient and natural parameters.
        """
        pass
    
    
    @abstractmethod
    def distribution_from_np(self, np):
        """
        Takes a dictionary of natural parameters **np** and returns a
        torch.distribution defined by these natural parameters.
        """
        pass
    


# =============================================================================
# Mean field Gaussian factor
# =============================================================================
    

class MeanFieldGaussian(ExponentialFamilyFactor):
    
    
    def __init__(self, log_coefficient, natural_parameters):
        
        super().__init__(log_coefficient=log_coefficient,
                         natural_parameters=natural_parameters)
    
    
    def log_h(self, thetas):
        """
        Returns the value of log h(θ) for the MeanFieldFactor class. For
        a mean-field multivariate Gaussian, log h(θ) = 0.
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        return torch.zeros(size=thetas.shape[:1])
    
    
    def f(self, thetas):
        """
        Computes f(θ) for the mean-field Gaussian where
        
            f(θ) = [θΤ, θΤ ** 2]Τ
        """
        
        f = torch.cat([thetas, thetas ** 2], dim=1)
        
        return f
    
    
    def natural_parameter_vector(self):
        
        np = self.natural_parameters
        
        return torch.cat([np["np1"], np["np2"]], axis=0)
    
    
    def log_coeff_and_np_from_distribution(self, q):
        
        # Exctract loc and scale from torch.distribution
        loc = q.loc.detach()
        scale = q.scale.detach()
        
        assert loc.shape == scale.shape
        assert len(loc.shape) == 1
        
        # Dimension of distribution
        D = loc.shape[0]
        
        # Compute log coefficient of q
        log_coeff = torch.log(scale).sum()
        log_coeff = log_coeff - 0.5 * D * math.log(2 * math.pi)
        
        # Compute natural parameters for multivariate normal
        np = {
            "np1" : loc / scale ** 2,
            "np2" : - 0.5 * scale ** -2
        }
        
        return log_coeff, np
    
    
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
    
    
    def __init__(self, log_coefficient, natural_parameters):
        
        super().__init__(log_coefficient, natural_parameters)
    
    
    def log_h(self, thetas):
        """
        Returns the value of log h(θ) for the MultivariateGaussian class. For
        a multivariate Gaussian, log h(θ) = 0.
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        return torch.zeros(size=thetas.shape[:1])
    
    
    def f(self, thetas):
        """
        """
        
        N = thetas.shape[0]
        
        thetas_outer = torch.einsum('ni, nj -> nij', thetas, thetas)
        thetas_outer = thetas_outer.view(N, -1)
        
        return torch.cat([thetas, thetas_outer], axis=1)
    
    
    def natural_parameter_vector(self):
        
        np = self.natural_parameters
        
        return torch.cat([np["np1"], np["np2"].view(-1)], axis=0)
    
    
    def log_coeff_and_np_from_distribution(self, q):
        """
        Takes a torch.distribution **q**, assumed to be in the EF
        and extracts its leading coefficient and natural parameters.
        """
        
        
        # Exctract loc and scale from torch.distribution
        loc = q.loc.detach()
        scale_tril = q.scale_tril.detach()
        
        assert loc.shape == scale.shape[0]
        assert len(loc.shape) == 1
        assert len(scale_tril.shape) == 2
        
        # Dimension of distribution
        D = loc.shape[0]
        
        # Compute log coefficient of q
        log_coeff = torch.log(torch.diag(scale_tril)).sum()
        log_coeff = log_coeff - 0.5 * D * math.log(2 * math.pi)
        
        # Compute natural parameters for multivariate normal
        np = {
            "np1" : torch.cholesky_solve(mean[:, None], scale_tril)[:, 0],
            "np2" : - 0.5 * scale ** -2
        }
        
        return log_coeff, np
    

    def distribution_from_np(self, np):
        """
        Takes a dictionary of natural parameters **np** and returns a
        torch.distribution defined by these natural parameters.
        """
        pass