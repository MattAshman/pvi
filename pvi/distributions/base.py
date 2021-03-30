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
        np1 = q1.nat_params
        np2 = q2.nat_params
        
        # Log coefficient and natural parameters of refined factor
        nat_params = {}
        
        # Compute natural parameters of the new t-factor (detach gradients)
        for name, np in self.nat_params.items():
            nat_params[name] = (
                    np1[name].detach().clone() - np2[name].detach().clone()
                    + np.detach().clone())
            
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
        raise NotImplementedError

    @abstractmethod
    def npf(self, thetas):
        """
        Rearranges NPs to ν vector for this member of the exponential family.
        """
        raise NotImplementedError

    @abstractmethod
    def eqlogt(self, q):
        """
        Computes E_q[log t(θ)] = ν.T E_q[f(θ)] + E_q[log h(θ)], ignoring the
        latter term.
        :param q: q(θ).
        :return: ν.T E_q[f(θ)].
        """
        raise NotImplementedError

    @abstractmethod
    def nat_from_dist(self, q):
        """
        Takes a torch.distribution **q**, assumed to be in the EF
        and extracts its leading coefficient and natural parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def dist_from_nat(self, np):
        """
        Takes a dictionary of natural parameters **np** and returns a
        torch.distribution defined by these natural parameters.
        """
        raise NotImplementedError


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

        # Set all to None.
        self._nat_params = None
        self._std_params = None
        self._unc_params = None
        
        # Initialise standard and natural parameters
        if is_trainable:
            # Only initialise std_params (initialises unc_params).
            if std_params is not None:
                self.std_params = std_params
            elif nat_params is not None:
                self.std_params = self._std_from_nat(nat_params)
            else:
                # No intitial parameter values specified.
                raise ValueError("No initial parameterisation specified. "
                                 "Cannot create optimisable parameters.")
        else:
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
        if self.is_trainable or self._nat_params is None:
            self._nat_params = self._nat_from_std(self.std_params)
        
        return self._nat_params

    @nat_params.setter
    def nat_params(self, nat_params):

        self._nat_params = nat_params

    @property
    @abstractmethod
    def mean_params(self):
        raise NotImplementedError

    @abstractmethod
    def _std_from_unc(self, unc_params):
        raise NotImplementedError

    @abstractmethod
    def _unc_from_std(self, std_params):
        raise NotImplementedError

    @abstractmethod
    def _nat_from_std(self, std_params):
        raise NotImplementedError
    
    @abstractmethod
    def _std_from_nat(self, nat_params):
        raise NotImplementedError

    def non_trainable_copy(self):

        if self.is_trainable:
            nat_params = None
            std_params = {k: v.detach().clone()
                          for k, v in self.std_params.items()}

        else:
            if self._std_params is not None:
                std_params = {k: v.detach().clone()
                              for k, v in self.std_params.items()}
            else:
                std_params = None

            if self._nat_params is not None:
                nat_params = {k: v.detach().clone()
                              for k, v in self.nat_params.items()}
            else:
                nat_params = None
        
        return type(self)(std_params, nat_params, is_trainable=False)

    def trainable_copy(self):

        if self.is_trainable:
            nat_params = None
            std_params = {k: v.detach().clone()
                          for k, v in self.std_params.items()}

        else:
            if self._std_params is not None:
                std_params = {k: v.detach().clone()
                              for k, v in self.std_params.items()}
            else:
                std_params = None

            if self._nat_params is not None:
                nat_params = {k: v.detach().clone()
                              for k, v in self.nat_params.items()}
            else:
                nat_params = None

        return type(self)(std_params, nat_params, is_trainable=True)

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
        raise NotImplementedError
