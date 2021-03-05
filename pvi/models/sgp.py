import logging
import torch
import numpy as np

from torch import distributions, nn, optim
from torch.utils.data import TensorDataset, DataLoader
from .base import Model
from pvi.likelihoods.gaussian import HomoGaussian
from pvi.utils.psd_utils import psd_inverse, psd_logdet, add_diagonal

logger = logging.getLogger(__name__)


class SparseGaussianProcessModel(Model):
    """
    Sparse Gaussian process model using closed-form optimisation of q(u).
    """

    def __init__(self, inducing_locations, output_sigma=1., **kwargs):
        super().__init__(HomoGaussian(output_sigma), **kwargs)

        # Construct inducing points and kernel.
        if self.hyperparameters["kernel_class"] is not None:
            self.kernel = self.hyperparameters["kernel_class"](
                **self.hyperparameters["kernel_params"]
            )
        else:
            raise ValueError("Kernel class not specified.")

        # Set up optimiser.
        if self.hyperparameters["optimiser_class"] is not None:
            self.optimiser = self.hyperparameters["optimiser_class"](
                self.parameters(),
                **self.hyperparameters["optimiser_params"]
            )
        else:
            raise ValueError("Optimiser class not specified.")

        # Add private inducing points to parameters.
        self.register_parameter(
            "inducing_locations",
            nn.Parameter(inducing_locations, requires_grad=True))

        # For logging training performance.
        self._training_curves = []

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.] * self.hyperparameters["num_inducing"]),
            "np2": torch.tensor(
                [-.5] * self.hyperparameters["num_inducing"]).diag_embed()
        }

    @staticmethod
    def get_default_hyperparameters():
        return {
            "D": None,
            "num_inducing": 50,
            "optimiser_class": optim.Adam,
            "optimiser_params": {"lr": 1e-3},
            "reset_optimiser": True,
            "epochs": 100,
            "print_epochs": 10
        }

    def set_parameters(self, nat_params):
        """
        Sets the optimisable parameters. These are just the natural parameters,
        since we perform natural parameter updates in the optimisation.
        :param nat_params: Natural parameters of a multivariate Gaussian
        distribution.
        """
        self.register_parameter(
            "np1", nn.Parameter(nat_params["np1"], requires_grad=False))
        self.register_parameter(
            "np2", nn.Parameter(nat_params["np2"], requires_grad=False))

    def forward(self, x):
        """
        Returns the (approximate) predictive posterior distribution of a
        Bayesian logistic regression model.
        :param x: The input locations to make predictions at.
        :return: ∫ p(y | θ, x) q(θ) dθ ≅ (1/M) Σ_m p(y | θ_m, x) θ_m ~ q(θ).
        """
        # Parameters of approximate posterior.
        q = self.get_distribution()
        mu = q.mean
        su = q.covariance_matrix

        # Parameters of prior.
        kxz = self.kernel(x, self.inducing_locations).evaluate()
        kzx = kxz.T
        kzz = add_diagonal(self.kernel(
            self.inducing_locations, self.inducing_locations).evaluate(),
                           1e-4)
        kxx = add_diagonal(self.kernel(x, x).evaluate(), 1e-4)

        # Predictive posterior.
        kzz_inv = psd_inverse(kzz)
        ppmu = kxz.matmul(kzz_inv).matmul(mu)
        ppcov = kxx + kxz.matmul(
            kzz_inv).matmul(su - kzz).matmul(kzz_inv).matmul(kzx)

        return distributions.MultivariateNormal(ppmu, covariance_matrix=ppcov)

    def fit(self, data, t_i):
        """
        Perform local VI.
        :param data: The local data to refine the model with.
        :param t_i: The local contribution of the client.
        :return: t_i_new, the new local contribution.
        """
        # Cavity, or effective prior, parameters.
        cav_np = {}
        for key in self.nat_params.keys():
            cav_np[key] = (self.nat_params[key] - t_i[key]).detach()

        # Set up data etc.
        x = data["x"]
        y = data["y"]
        n = x.shape[0]

        # Set up optimiser.
        if self.hyperparameters["optimiser_class"] is not None and \
                self.hyperparameters["reset_optimiser"]:
            logging.info("Resetting optimiser")
            self.optimiser = self.hyperparameters["optimiser_class"](
                self.parameters(), **self.hyperparameters["optimiser_params"]
            )

        # Local optimisation to find new parameters.
        training_curve = {
            "elbo": [],
        }
        for i in range(self.hyperparameters["epochs"]):
            epoch = {
                "elbo": 0,
            }
            # Compute ELBO by integrating out q(u).
            # Parameters of prior.
            kxz = self.kernel(x, self.inducing_locations).evaluate()
            kzx = kxz.T
            kzz = add_diagonal(self.kernel(
                self.inducing_locations, self.inducing_locations).evaluate(),
                               1e-4)
            kxx = add_diagonal(self.kernel(x, x).evaluate(), 1e-4)

            lzz = kzz.cholesky()

            # Derivation following GPFlow notes.
            sigma = self.likelihood.output_sigma
            a = torch.triangular_solve(kzx, lzz, upper=False)[0]*sigma.pow(-1)
            b = torch.eye(lzz.shape[0]) + a.matmul(a.T)
            lbb = b.cholesky()
            c = (torch.triangular_solve(a.matmul(y), lbb, upper=False)[0]
                 * sigma.pow(-1))

            elbo = 0.5 * (
                    -n * torch.log(2 * np.pi * sigma.pow(2))
                    - psd_logdet(b, chol=lbb)
                    - sigma.pow(-2) * y.reshape(-1).dot(y.reshape(-1))
                    + c.T.matmul(c)
                    - sigma.pow(-2) * kxx.trace()
                    + (a.matmul(a.T)).trace())

            loss = -elbo

            loss.backward()
            self.optimiser.step()

            # Will be very slow if training on GPUs.
            epoch["elbo"] += elbo.item()

            # Log progress.
            training_curve["elbo"].append(epoch["elbo"])

            if i % self.hyperparameters["print_epochs"] == 0:
                logger.debug(
                    "ELBO: {:.3f}, Epochs: {}.".format(epoch["elbo"], i))

            if i == (self.hyperparameters["epochs"] - 1):
                # Update optimum q(u) on final epoch.
                kzz_inv = psd_inverse(chol=lzz)
                prec = (kzz_inv + sigma.pow(-2)
                        * kzz_inv.matmul(kzx.matmul(kxz)).matmul(kzz_inv))
                np2 = nn.Parameter(-0.5 * prec, requires_grad=False)
                np1 = nn.Parameter(
                    sigma.pow(-2) * kzz_inv.matmul(kzx).matmul(y).squeeze(-1),
                    requires_grad=False)
                self.set_parameters({"np1": np1, "np2": np2})

        # Add progress log.
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
        nat_params = {
            "np1": self.np1,
            "np2": self.np2,
        }
        return nat_params


class StochasticSparseGaussianProcessModel(Model):
    """
    Sparse Gaussian process model using stochastic optimisation of q(u).
    """
    def __init__(self, inducing_locations, output_sigma=1., **kwargs):
        super().__init__(HomoGaussian(output_sigma), **kwargs)

        # Construct inducing points and kernel.
        if self.hyperparameters["kernel_class"] is not None:
            self.kernel = self.hyperparameters["kernel_class"](
                **self.hyperparameters["kernel_params"]
            )
        else:
            raise ValueError("Kernel class not specified.")

        # Set up optimiser.
        if self.hyperparameters["optimiser_class"] is not None:
            self.optimiser = self.hyperparameters["optimiser_class"](
                self.parameters(),
                **self.hyperparameters["optimiser_params"]
            )
        else:
            raise ValueError("Optimiser class not specified.")

        # Add private inducing points to parameters.
        self.register_parameter(
            "inducing_locations",
            nn.Parameter(inducing_locations, requires_grad=True))

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.] * self.hyperparameters["num_inducing"]),
            "np2": torch.tensor(
                [-.5] * self.hyperparameters["num_inducing"]).diag_embed()
        }

    @staticmethod
    def get_default_hyperparameters():
        return {
            "D": None,
            "num_inducing": 50,
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
        # Parameters of approximate posterior.
        q = self.get_distribution()
        mu = q.mean
        su = q.covariance_matrix

        # Parameters of prior.
        kzx = self.kernel(x, self.inducing_locations)
        kxz = kzx.T
        kzz = self.kernel(self.inducing_locations, self.inducing_locations)
        kxx = self.kernel(x, x)

        # Predictive posterior.
        kzz_inv = psd_inverse(kzz)
        ppmu = kzx.matmul(kzz_inv).matmul(mu)
        ppcov = kxx - kxz.matmul(
            kzz_inv).matmul(su - kzz).matmul(kzz_inv).matmul(kzx)

        return distributions.MultivariateNormal(ppmu, covariance_matrix=ppcov)

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
                # Compute log-likelihood under q(u).
                q = self.get_distribution()
                mu = q.mean
                su = q.covariance_matrix

                # Parameters of prior.
                kzx = self.kernel(
                    x_batch, self.inducing_locations).unsqueeze(-1)
                kxz = kzx.T
                kzz = self.kernel(
                    self.inducing_locations, self.inducing_locations)
                kxx = self.kernel(x_batch.unsqueeze(-1), x_batch.unsqueeze(-1))

                kzz_inv = psd_inverse(kzz)
                a = kxz.matmul(kzz_inv) # (N, D)
                b = kxx - kxz.matmul(kzz_inv).matmul(kxz)   # (N, 1, 1)

                output_sigma = self.likelihood.output_sigma
                dist = self.likelihood.forward(a.matmul(mu))
                ll1 = dist.log_prob(y_batch)
                ll2 = -output_sigma.pow(-2) * (a.matmul(su).matmul(a.T) + b)
                ll = (ll1 + ll2).sum()

                # Compute the KL divergence between current approximate
                # posterior and prior.
                q = self.get_distribution()
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
