import torch

from torch import distributions, nn, optim
from torch.utils.data import TensorDataset, DataLoader
from .base import Model
from pvi.likelihoods.logistic_regression import LogisticRegressionLikelihood


class LogisticRegressionModel(Model):
    """
    Logistic regression model with a multivariate Gaussian approximate
    posterior.
    """
    def __init__(self, output_sigma=1., **kwargs):
        super().__init__(LogisticRegressionLikelihood(output_sigma), **kwargs)

        # Set up optimiser.
        if self.hyperparameters["optimiser_class"] is not None:
            self.optimiser = self.hyperparameters["optimiser_class"](
                self.parameters(), **self.hyperparameters["optimiser_params"]
            )

    def get_default_parameters(self):
        return {
            "np1": nn.Parameter(
                torch.tensor([0.]*self.hyperparameters["D"]),
                requires_grad=True),
            "np2": nn.Parameter(
                torch.tensor([1.]*self.hyperparameters["D"]).diag_embed(),
                requires_grad=True)
        }

    def get_default_hyperparameters(self):
        return {
            "D": None,
            "optimiser_class": optim.Adam,
            "optimiser_params": {"lr": 1e-3},
            "reset_optimiser": True,
            "epochs": 100,
            "batch_size": 100,
            "num_elbo_samples": 1,
            "num_predictive_samples": 1,
        }

    def forward(self, x):
        """
        Returns the (approximate) predictive posterior distribution of a
        Bayesian logistic regression model.
        :param x: The input locations to make predictions at.
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        q = self.get_distribution()
        thetas = q.sample((self.hyperparameters["num_predictive_samples"],))
        comp = self.likelihood.forward(x, thetas)
        mix = distributions.Categorical(torch.ones(len(thetas),))

        return distributions.MixtureSameFamily(mix, comp)

    def fit(self, data, t_i):
        """
        Perform local VI
        :param data: The local data to refine the model with.
        :param t_i: The local contribution of the client.
        :return: t_i_new, the new local contribution.
        """
        # Set up optimiser.
        if self.hyperparameters["optimiser_class"] is not None and \
                self.hyperparameters["reset_optimiser"]:
            self.optimiser = self.hyperparameters["optimiser_class"](
                self.parameters(), **self.hyperparameters["optimiser_params"]
            )

        # Cavity, or effective prior, parameters.
        cav_np = {}
        for key in self.parameters.keys():
            cav_np[key] = (self.parameters[key] - t_i[key]).detach()

        # Set up data etc.
        x_full = data["x"]
        y_full = data["y"]
        dataset = TensorDataset(x_full, y_full)
        loader = DataLoader(
            dataset, batch_size=self.hyperparameters["batch_size"],
            shuffle=True)

        # Local optimisation to find new parameters.
        for _ in self.hyperparameters["epochs"]:
            for batch_idx, (x_batch, y_batch) in iter(loader):
                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # TODO: perform some MCMC inference.
                q = self.get_distribution()
                thetas = q.rsample((self.hyperparameters["num_elbo_samples"],))
                ll = self.likelihood.log_prob(batch, thetas).mean()

                # Compute the KL divergence between current approximate
                # posterior and prior.
                q_cav = self.get_distribution(cav_np)
                kl = distributions.kl_divergence(q_cav, q)

                loss = kl - ll
                loss.backward()
                self.optimiser.step()

        # New local contribution.
        t_i_new = {}
        for key in self.parameters.keys():
            t_i_new[key] = (self.parameters[key] - cav_np[key]).detach()

        return t_i_new

    def sample(self, x, num_samples=1):
        """
        Samples the (approximate) predictive posterior distribution of a
        Bayesian logistic regression model.
        :param x: The input locations to make predictions at.
        :param num_samples: The number of samples to take.
        :return: A sample from the (approximate) predictive posterior,
        ∫ p(y | θ, x) q(θ) dθ.
        """
        pp = self.forward(x)

        return pp.sample((num_samples,))

    def get_distribution(self, parameters=None):
        """
        Return a multivariate Gaussian distribution defined by parameters.
        :param parameters: Natural parameters of a multivariate Gaussian
        distribution.
        :return: Multivariate Gaussian distribution defined by the parameters.
        """
        if parameters is None:
            parameters = self.parameters

        prec = -2 * parameters["np2"]
        mu = torch.solve(parameters["np1"], prec)

        return distributions.MultivariateNormal(mu, precision_matrix=prec)
