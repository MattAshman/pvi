from abc import ABC, abstractmethod
from torch import nn

import torch


# =============================================================================
# Base class for approximating likelihood factors of the exponential family
# =============================================================================


class ExponentialFamilyFactor(ABC):
    """
    Base class for exponential family (EF) approximating likelihood factors.
    The exponential family is made up of distributions which can be written
    in the form
    
        p(θ | v) = h(θ) exp(ν.T f(θ) + A(ν)).
    
    For a list of exponential family distributions see:
        https://en.wikipedia.org/wiki/Category:Exponential_family_distributions
        https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions
        
    There are many more members but these are especially useful for our
    applications.
    """
    
    def __init__(self, nat_params):
        
        self.nat_params = nat_params
      
    
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
        np1 = self.nat_from_dist(q1.distribution)
        np2 = self.nat_from_dist(q2.distribution)
        
        # Log coefficient and natural parameters of refined factor
        nat_params = {}
        
        # Compute natural parameters of the new t-factor
        for name, np in self.nat_params.items():
            nat_params[name] = np1[name] - np2[name] + np
            
        # Create and return refined t of the same type
        t = type(self)(nat_params)
        
        return t
    

    def __call__(self, thetas):
        """
        Returns the value of log t(θ) (up to a const. independent of θ)
        
            log t(θ) = log h(θ) + ν.T f(θ) + const.
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        
        # Compute inner product ν.T f(θ), log h(θ), log t(θ) (all shape (N,))
        npf = self.npf(thetas)
        log_h = self.log_h(thetas)
        log_t = log_h + npf
        
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
    def npf(self, thetas):
        """
        Rearranges NPs to ν vector for this member of the exponential family.
        """
        pass
    
    
    @abstractmethod
    def nat_from_dist(self, q):
        """
        Takes a torch.distribution **q**, assumed to be in the EF
        and extracts its leading coefficient and natural parameters.
        """
        pass
    
    
    @abstractmethod
    def dist_from_nat(self, np):
        """
        Takes a dictionary of natural parameters **np** and returns a
        torch.distribution defined by these natural parameters.
        """
        pass
    


# =============================================================================
# Base exponential family distribution
# =============================================================================


class ExponentialFamilyDistribution(ABC, nn.Module):
    
    def __init__(self,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False):
        
        super().__init__()
    
        # Specify whether the distribution is trainable wrt its NPs
        self.is_trainable = is_trainable
        
        # Set standard and natural parameters
        self.std_params = std_params
        self.nat_params = nat_params
    
        
    @property
    def std_params(self):
        
        if self.is_trainable:
            return self._std_from_unc(self._unc_params)
        
        elif self._std_params is None:
            return self._std_from_nat(self._nat_params)
        
        else:
            return self._std_params
        
            
    @std_params.setter
    def std_params(self, std_params):
        
        if self.is_trainable:
            self._unc_params = nn.ParameterDict(self._unc_from_std(std_params))
            
        else:
            self._std_params = std_params

        
    @property
    def nat_params(self):
        
        # If _nat_params None or distribution trainable compute nat params
        if (self.is_trainable) or (self._nat_params is None):
            self._nat_params = self._nat_from_std(self.std_params)
        
        return self._nat_params
    
    
    @nat_params.setter
    def nat_params(self, nat_params):
        self._nat_params = nat_params
        
        
    @abstractmethod
    def _std_from_unc(self, unc_params):
        pass
    
    
    @abstractmethod
    def _unc_from_std(self, std_params):
        pass
    
    
    @abstractmethod
    def _nat_from_std(self, std_params):
        pass
    
    
    @abstractmethod
    def _std_from_nat(self, nat_params):
        pass
    
    
    def non_trainable_copy(self):
        
        std_params = self.std_params.detach()
        nat_params = self.nat_params.detach()
        
        return type(self)(std_params, nat_params, is_trainable=False)
        
    
    
    @property
    def distribution(self):
        return self.torch_dist_class(**self.std_params)
    
    
    def kl_divergence(self, other):
        return torch.distributions.kl_divergence(self.distribution,
                                                 other.distribution)
    
    
    def log_prob(self, *args, **kwargs):
        return self.distribution.log_prob(*args, **kwargs)
    
    
    def sample(self, *args, **kwargs):
        return self.distribution.sample(*args, **kwargs)
    
    
    def rsample(self, *args, **kwargs):
        return self.distribution.rsample(*args, **kwargs)
    
    
    @property
    @abstractmethod
    def torch_dist_class(self):
        pass
    