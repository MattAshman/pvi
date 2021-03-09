from abc import ABC, abstractmethod
from torch import nn

# =============================================================================
# Base class for approximating likelihood factors of the exponential family
# =============================================================================


class ExponentialFamilyFactor(ABC):
    """
    Base class for exponential family (EF) approximating likelihood factors.
    The exponential family is made up of distributions which can be written
    in the form
    
        p(x | θ) = h(x) exp(ν.T f(x) + A(ν)).
    
    For a list of exponential family distributions see:
        https://en.wikipedia.org/wiki/Category:Exponential_family_distributions
        https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions
    
    
    TODO: Write children classes for the following members of the EF
    
        - Dirichlet
        - Multinomial
        
    There are many more members but these are especially useful for our
    applications.
    """
    
    def __init__(self, natural_parameters):
        
        self.natural_parameters = natural_parameters
      
    
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
        np1 = self.np_from_distribution(q1)
        np2 = self.np_from_distribution(q2)
        
        # Log coefficient and natural parameters of refined factor
        natural_parameters = {}
        
        # Compute natural parameters of the new t-factor
        for name, np in self.natural_parameters.items():
            natural_parameters[name] = np1[name] - np2[name] + np
            
        # Create and return refined t of the same type
        t = type(self)(natural_parameters)
        
        return t
    

    def __call__(self, thetas):
        """
        Returns the value of log t(θ) where
        
            log t(θ) = log c + log h(x) + ν.T f(x)
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        
        # Compute inner product ν.T f(x), log h(θ), log t(θ) (all shape (N,))
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
    def np_from_distribution(self, q):
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
# Base exponential family distribution
# =============================================================================


class ExponentialFamilyDistribution(ABC, nn.Module):
    
    def __init__(self,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False):
    
        # Specify whether the distribution is trainable wrt its NPs
        self.is_trainable = is_trainable
        
        # Set standard and natural parameters
        self.std_params = std_params
        self.nat_params = nat_params

        
    @property
    def std_params(self):
        
        # If _std_params None or distribution trainable compute std params
        if (self.is_trainable) or (self._std_params is None):
            self._std_params = self._nat_to_std(self._nat_params)
        
        return self._std_params
    
    
    @property.setter
    def std_params(self, std_params):
        self._std_params = std_params
    
        
    @property
    def nat_params(self):
        
        if self.is_trainable:
            return self._nat_from_unc(self._unc_params)
        
        else:
            return self._nat_params
        
            
    @nat_params.setter
    def nat_params(self, nat_params):
        
        if self.is_trainable:
            _unc_params = self._unc_from_nat(nat_params)
            
            # TODO: register trainable parameters here
            self._unc_params = None
            
        else:
            self._nat_params = nat_params
        
        
    @abstractmethod
    def _nat_from_unc(self, unconstrained):
        pass
    
    
    @abstractmethod
    def _unc_from_nat(self, nat_params):
        pass
    
    
    @abstractmethod
    def _std_to_nat(self):
        pass
    
    
    @abstractmethod
    def _nat_to_std(self):
        pass
        
    
    @property
    @abstractmethod
    def torch_dist_class(self):
        pass
    
    
    def kl_divergence(self, q_):
        return self.q.kl_divergence(q_)
    
    
    def log_prob(self, *args, **kwargs):
        return self.q.log_prob(*args, **kwargs)
    
    
    def sample(self, *args, **kwargs):
        return self.q.sample(*args, **kwargs)
    
    
    def rsample(self):
        return self.q.rsample(*args, **kwargs)
    
    