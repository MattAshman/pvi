import logging
import torch

from torch import distributions, nn, optim
from torch.utils.data import TensorDataset, DataLoader
from .base import Model
from pvi.likelihoods.logistic_regression import LogisticRegressionLikelihood

logger = logging.getLogger(__name__)


class LogisticRegressionModel(Model):
    """
    Logistic regression model with a multivariate Gaussian approximate
    posterior.
    """
    def __init__(self, **kwargs):
        super().__init__(LogisticRegressionLikelihood(), **kwargs)

        # Set up optimiser.
        if self.hyperparameters["optimiser_class"] is not None:
            self.optimiser = self.hyperparameters["optimiser_class"](
                self.parameters(), **self.hyperparameters["optimiser_params"]
            )
        else:
            raise ValueError("Optimiser class not specified.")

        # For logging training performance.
        self._training_curves = []

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.]*(self.hyperparameters["D"]+1)),
            "np2": torch.tensor([-.5]*(
                    self.hyperparameters["D"]+1)).diag_embed()
        }

    @staticmethod
    def get_default_hyperparameters():
        return {
            "D": None,
            "optimiser_class": optim.Adam,
            "optimiser_params": {"lr": 1e-3},
            "reset_optimiser": True,
            "epochs": 100,
            "batch_size": 100,
            "num_elbo_samples": 1,
            "num_predictive_samples": 1,
            "print_epochs": 10,
        }

    def set_parameters(self, nat_params):
        """
        Sets the optimisable parameters. These are different to the natural
        parameters to ensure that the positive definite constraint of the
        covariance matrix is not violated during training.
        :param nat_params: Natural parameters of a multivariate Gaussian
        distribution.
        """
        # Work with Cholesky factor of precision matrix and np1.
        prec = -2. * nat_params["np2"]
        prec_chol = torch.cholesky(prec)
        self.register_parameter(
            "prec_chol", nn.Parameter(prec_chol, requires_grad=True))
        self.register_parameter(
            "np1", nn.Parameter(nat_params["np1"], requires_grad=True))

    def forward(self, x):
        """
        Returns the (approximate) predictive posterior distribution of a
        Bayesian logistic regression model.
        :param x: The input locations to make predictions at.
        :return: ∫ p(y | θ, x) q(θ) dθ ≅ (1/M) Σ_m p(y | θ_m, x) θ_m ~ q(θ).
        """
        q = self.get_distribution()
        thetas = q.sample((self.hyperparameters["num_predictive_samples"],))

        comp = self.likelihood.forward(x, thetas)
        mix = distributions.Categorical(torch.ones(len(thetas),))

        return distributions.MixtureSameFamily(mix, comp)

    def fit(self, data, t_i):
        """
        Perform local VI.
        :param data: The local data to refine the model with.
        :param t_i: The local contribution of the client.
        :return: t_i_new, the new local contribution.
        """
        # Set up optimiser.
        if self.hyperparameters["optimiser_class"] is not None and \
                self.hyperparameters["reset_optimiser"]:
            logging.info("Resetting optimiser")
            self.optimiser = self.hyperparameters["optimiser_class"](
                self.parameters(), **self.hyperparameters["optimiser_params"]
            )

        # Cavity, or effective prior, parameters.
        cav_np = {}
        for key in self.nat_params.keys():
            cav_np[key] = (self.nat_params[key] - t_i[key]).detach()

        # Set up data etc.
        x_full = data["x"]
        y_full = data["y"]
        dataset = TensorDataset(x_full, y_full)
        loader = DataLoader(
            dataset, batch_size=self.hyperparameters["batch_size"],
            shuffle=True)

        # Local optimisation to find new parameters.
        training_curve = {
            "elbo": [],
            "kl": [],
            "ll": [],
        }
        for i in range(self.hyperparameters["epochs"]):
            epoch = {
                "elbo": 0,
                "kl": 0,
                "ll": 0,
            }
            for (x_batch, y_batch) in iter(loader):
                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }
                # TODO: perform some MCMC inference.
                q = self.get_distribution()
                thetas = q.rsample((self.hyperparameters["num_elbo_samples"],))
                ll = self.likelihood.log_prob(batch, thetas).mean(0).sum()

                # Compute the KL divergence between current approximate
                # posterior and prior.
                q_cav = self.get_distribution(cav_np)
                kl = distributions.kl_divergence(q_cav, q)

                loss = kl - ll
                loss.backward()
                self.optimiser.step()

                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item()
                epoch["kl"] += kl.item()
                epoch["ll"] += ll.item()

            # Log progress.
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if i % self.hyperparameters["print_epochs"] == 0:
                logger.debug(
                    "ELBO: {:.3f}, LL: {:.3f}, KL: {:.3f}, Epochs: {}.".format(
                        epoch["elbo"], epoch["ll"], epoch["kl"], i))

        self._training_curves.append(training_curve)

        # New local contribution.
        t_i_new = {}
        for key in self.nat_params.keys():
            t_i_new[key] = (self.nat_params[key] - cav_np[key]).detach()

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

    def get_distribution(self, nat_params=None):
        """
        Return a multivariate Gaussian distribution defined by parameters.
        :param nat_params: Natural parameters of a multivariate Gaussian
        distribution.
        :return: Multivariate Gaussian distribution defined by the parameters.
        """
        if nat_params is None:
            nat_params = self.nat_params

        prec = -2 * nat_params["np2"]
        mu = torch.solve(nat_params["np1"].unsqueeze(-1), prec)[0].squeeze(-1)

        return distributions.MultivariateNormal(mu, precision_matrix=prec)

    @property
    def nat_params(self):
        """
        Returns the natural parameters, based on parameters included in self.
        :return: Natural parameters.
        """
        np1 = self.np1
        np2 = -0.5 * self.prec_chol.matmul(self.prec_chol.T)
        nat_params = {
            "np1": np1,
            "np2": np2,
        }
        return nat_params
