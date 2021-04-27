import torch
from torch import nn

import numpy as np

from pvi.models.base import Model

from abc import ABC, abstractmethod, abstractproperty


# =============================================================================
# Bayesian Neural network base class
# =============================================================================
        
        
class FullyConnectedBNN(Model, nn.Module, ABC):
    
    conjugate_family = None

    
    def __init__(self, **kwargs):
        
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)
        
        
    @abstractproperty
    def shapes(self):
        pass
    
    
    @abstractmethod
    def pred_dist_from_tensor(self, tensor, samples_first):
        pass
    
        
    @property
    def sizes(self):
        return [np.prod(shape) for shape in self.shapes]
    
    @property
    def num_parameters(self):
        return sum(self.sizes)

    
    @staticmethod
    def get_default_hyperparameters():
        
        default_hyperparameters = {
            "optimiser_class"          : torch.optim.Adam,
            "optimiser_params"         : {"lr": 1e-3},
            "reset_optimiser"          : True,
            "epochs"                   : 100,
            "batch_size"               : 1000,
            "num_elbo_samples"         : 10,
            "num_predictive_samples"   : 10,
            "device"                   : 'cpu'
        }
        
        return default_hyperparameters
        
        
    def get_default_nat_params(self):
        
        default_nat_params = {
            "np1" : torch.zeros(size=(self.num_parameters,)),
            "np2" : -0.5 * 1e6 * torch.ones(size=(self.num_parameters,))
        }
        
        return default_nat_params
    

    def forward(self, x, q, model_predict=False):
        
        # Number of θ samples to draw
        num_pred_samples = self.hyperparameters["num_predictive_samples"]
        theta = q.distribution.sample((num_pred_samples,))

        # Collection of output distributions, one for each θ, x pair
        # Distribution assumed to be of shape (S, N, D)
        pred_dist = self.likelihood_forward(x, theta, samples_first=False)
        
        # Predictive is a mixture of predictive distributions with equal weights
        equal_logits = torch.ones(size=(pred_dist.logits.shape[:2]))
        mix_prop = torch.distributions.Categorical(logits=equal_logits)
        pred_dist = torch.distributions.MixtureSameFamily(mix_prop, pred_dist)

        return pred_dist
    

    def likelihood_forward(self, x, theta, samples_first=True):
        
        assert len(x.shape) in [1, 2], "x must be (N, D)."
        assert len(theta.shape) in [1, 2], "theta must be (S, K)."
        
        if len(x.shape) == 1:
            x = x[None, :]
            
        if len(theta.shape) == 1:
            theta = theta[None, :]
        
        # Converts θ-vectors to tensors, shaped as expected by the network
        theta = self.reshape_theta(theta)
        
        # Expand input tensor dimension for broadcasting
        tensor = x[None, :, :]
        
        for i, (W, b) in enumerate(theta):
            
            tensor = torch.einsum('sni, sij -> snj', tensor, W)
            tensor = tensor + b[:, None, :]
            
            if i < len(theta) - 1:
                tensor = torch.nn.ReLU()(tensor)

        return self.pred_dist_from_tensor(tensor, samples_first=samples_first)

    
    def likelihood_log_prob(self, data, theta):
        """
        Compute the log probability of the data under the model's likelihood.
        :param data: The data to compute the log likelihood of.
        :param theta: The latent variables of the model.
        :return: The log likelihood of the data.
        """
        
        device = self.hyperparameters['device']
        
        dist = self.likelihood_forward(data["x"].to(device), theta.to(device))
        return dist.log_prob(data["y"].to(device))
    
    
    def reshape_theta(self, theta):
        
        # Number of Monte Carlo samples of parameters
        S = theta.shape[0]
        
        # Check total number of parameters in network equals size of theta
        assert self.num_parameters == theta.shape[-1]
        
        # Indices to slice the theta tensor at
        slices = np.cumsum([0] + self.sizes)
        
        theta = [torch.reshape(theta[:, s1:s2], [S] + list(shape)) \
                 for shape, s1, s2 in zip(self.shapes, slices[:-1], slices[1:])]
        theta = list(zip(theta[::2], theta[1::2]))
        
        return theta

    
    def conjugate_update(self, *args, **kwargs):
        raise NotImplementedError
        

    def expected_log_likelihood(self, *args, **kwargs):
        raise NotImplementedError
        
        
        
# =============================================================================
# Two layer regression BNN
# =============================================================================
        
        
class TwoLayerRegressionBNN(FullyConnectedBNN):
    
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.input_dim = self.hyperparameters['D']
        self.latent_dim = self.hyperparameters['latent_dim']
        self.output_dim = self.hyperparameters['output_dim']
        
        
    @property
    def shapes(self):
        
        shapes = [(self.input_dim, self.latent_dim),
                  (self.latent_dim,),
                  (self.latent_dim, self.latent_dim),
                  (self.latent_dim,),
                  (self.latent_dim, 2 * self.output_dim),
                  (2 * self.output_dim,)]
        
        return shapes
    
    
    def pred_dist_from_tensor(self, tensor):
        
        loc = tensor[:, :, :self.output_dim]
        scale = torch.exp(tensor[:, :, self.output_dim:])
        
        return torch.distributions.normal.Normal(loc=loc, scale=scale)


# =============================================================================
# Two layer classification BNN
# =============================================================================
        
        
class TwoLayerClassificationBNN(FullyConnectedBNN):
    
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.input_dim = self.hyperparameters['D']
        self.latent_dim = self.hyperparameters['latent_dim']
        self.output_dim = self.hyperparameters['output_dim']
        
        
    @property
    def shapes(self):
        
        shapes = [(self.input_dim, self.latent_dim),
                  (self.latent_dim,),
                  (self.latent_dim, self.latent_dim),
                  (self.latent_dim,),
                  (self.latent_dim, self.output_dim),
                  (self.output_dim,)]
        
        return shapes
    
    
    def pred_dist_from_tensor(self, tensor, samples_first=True):
        
        if not samples_first:
            tensor = torch.transpose(tensor, 0, 1)
        
        return torch.distributions.Categorical(logits=tensor)
    