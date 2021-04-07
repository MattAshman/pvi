import torch

from torch import nn

from pvi.models.base import Model



class FullyConnectedNetwork(nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims,
                 nonlinearity):
        
        super().__init__()
        
        shapes = [input_dim] + hidden_dims + [output_dim]
        shapes = [(s1, s2) for s1, s2 in zip(shapes[:-1], shapes[1:])]
        
        self.W = []
        self.b = []
        self.num_linear = len(hidden_dims) + 1
        
        for shape in shapes:

            W = nn.Parameter(torch.randn(size=shape) / shape[0] ** 0.5)
            b = nn.Parameter(torch.randn(size=shape[1:]))

            self.W.append(W)
            self.b.append(b)
            
        self.W = torch.nn.ParameterList(self.W)
        self.b = torch.nn.ParameterList(self.b)
        
        self.nonlinearity = getattr(nn, nonlinearity)()





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
    
    
    def sizes(self):
        return [np.prod(shape) for shape in self.shapes]

    
    @staticmethod
    def get_default_hyperparameters():
        
        default_hyperparameters = {
            "D"                        : None,
            "optimiser_class"          : torch.optim.Adam,
            "optimiser_params"         : {"lr": 1e-3},
            "reset_optimiser"          : True,
            "epochs"                   : 100,
            "batch_size"               : 1000,
            "num_elbo_samples"         : 10,
            "num_predictive_samples"   : 10
        }
        
        return default_hyperparameters
        
        
    def get_default_nat_params(self):
        
        num_params = sum(self.sizes)
        
        default_nat_params = {
            "np1" : torch.zeros(shape=(num_params,)),
            "np2" : -0.5 * 1e6 * torch.ones(shape=(num_params,))
        }
        
        return default_nat_params
    
    
    @abstractmethod
    def pred_dist_from_tensor(self, tensor):
        pass
    

    def forward(self, x, q):
        
        # Number of θ samples to draw
        num_pred_samples = self.hyperparameters["num_predictive_samples"]
        thetas = q.distribution.sample((num_pred_samples,))

        # Collection of output distributions, one for each θ, x pair
        pred_dist = self.likelihood_forward(x, thetas)
        
        # Predictive is a mixture of predictive distributions with equal weights
        mix_prop = torch.distributions.Categorical(torch.ones(len(thetas),))
        pred_dist = torch.distributions.MixtureSameFamily(mix_prop, pred_dist)

        return pred_dist
    

    def likelihood_forward(self, x, theta):
        
        assert len(x.shape) in [1, 2], "x must be (N, D)."
        assert len(theta.shape) in [1, 2], "theta must be (S, K)."
        
        if len(x.shape) == 1:
            x = x[None, :]
            
        if len(thetas.shape) == 1:
            theta = theta[None, :]
        
        # Converts θ-vectors to tensors, shaped as expected by the network
        theta = self.reshape_theta(theta)
        
        # Expand input tensor dimension for broadcasting
        tensor = x[None, :, :]
        
        for i, (W, b) in enumerate(theta):
            
            tensor = torch.einsum('sni, sij -> snj', tensor, W)
            tensor = tensor + b[:, None, :]
            
            if i < len(thetas) - 2:
                tensor = self.nonlinearity(tensor)

        return self.pred_dist_from_tensor(tensor)
    
    
    def reshape_theta(self, theta):
        
        # Number of Monte Carlo samples of parameters
        S = theta.shape[0]
        
        # Check total number of parameters in network equals size of theta
        assert sum(self.sizes) == theta.shape[-1]
        
        # Indices to slice the theta tensor at
        slices = np.cumsum([0] + self.sizes)
        
        theta = [torch.reshape(theta[:, s1:s2], [S] + shape) \
                 for shape, s1, s2 in zip(self.shapes, *slices)]
        theta = list(zip(theta[::2], theta[1::2]))
        
        return theta

    
    def conjugate_update(self, *args, **kwargs):
        raise NotImplementedError
        

    def expected_log_likelihood(self, *args, **kwargs):
        raise NotImplementedError
        
        
        
# =============================================================================
# Two layer regression BNN
# =============================================================================


# =============================================================================
# Two layer classification BNN
# =============================================================================
        
        
class TwoLayerClassificationBNN(FullyConnectedBNN):
    
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        
    @abstractproperty
    def shapes(self):
        
        shapes = [(self.input_dim, self.latent_dim),
                  (self.latent_dim),
                  (self.latent_dim, self.latent_dim),
                  (self.latent_dim),
                  (self.latent_dim, 2 * self.output_dim),
                  (2 * self.output_dim)]
        
        return shapes
    
    
    @abstractmethod
    def pred_dist_from_tensor(self, tensor):
        
        loc = tensor[:, :, :self.output_dim]
        scale = torch.exp(tensor[:, :, self.output_dim:])
        
        return torch.distributions.normal.Normal(loc=loc, scale=scale)