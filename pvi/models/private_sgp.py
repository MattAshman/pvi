import torch

from torch import distributions, nn, optim
from .base import Model
from pvi.utils.psd_utils import psd_inverse, add_diagonal

JITTER = 1e-6


class PrivateSparseGaussianProcessRegression(Model, nn.Module):
    """
    Sparse Gaussian process model with private inducing points.
    """

    conjugate_family = None

    def __init__(self, output_sigma=1., **kwargs):
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        # Construct inducing points and kernel.
        if self.hyperparameters["kernel_class"] is not None:
            self.kernel = self.hyperparameters["kernel_class"](
                **self.hyperparameters["kernel_params"]
            )
        else:
            raise ValueError("Kernel class not specified.")

        self.register_parameter("output_sigma", nn.Parameter(
            torch.tensor(output_sigma), requires_grad=True))

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

    def forward(self, x, q):
        """
        Returns the (approximate) predictive posterior distribution of a
        sparse Gaussian process.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(u | z).
        :return: ∫ p(y | f, x) p(f | u) q(u) df du.
        """
        qu_loc = q.std_params["loc"]
        qu_cov = q.std_params["covariance_matrix"]
        z = q.inducing_locations

        # Parameters of prior.
        kxz = self.kernel(x, z).evaluate()
        kzx = kxz.T
        kzz = add_diagonal(self.kernel(z, z).evaluate(), JITTER)
        kxx = add_diagonal(self.kernel(x, x).evaluate(), JITTER)

        ikzz = psd_inverse(kzz)

        # Predictive posterior.
        qf_mu = kxz.matmul(ikzz).matmul(qu_loc)
        qf_cov = kxx + kxz.matmul(
            ikzz).matmul(qu_cov - kzz).matmul(ikzz).matmul(kzx)

        return distributions.MultivariateNormal(
            qf_mu, covariance_matrix=qf_cov)
    
    def likelihood_forward(self, x, theta=None):
        """
        Returns the model's likelihood p(y | x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model.
        :return: p(y | x).
        """
        return distributions.Normal(x, self.output_sigma)
    
    def conjugate_update(self, data, q, t_i):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t_i: The local factor t(θ).
        :return: q_new, t_i_new, the new global posterior and the new local
        contribution.
        """
        raise NotImplementedError

    def local_free_energy(self, data, q, t):
        """
        Returns the local variational free energy (up to an additive constant)
        of the sparse Gaussian process model under q(a) with local factor t(b).
        :param data: The local data.
        :param q: The current global posterior q(a).
        :param t: The local factor t(b).
        :return: The evidence lower bound.
        """
        # Set up data etc.
        x = data["x"]
        y = data["y"]

        # Copy to approximate posterior, making it non-trainable.
        qa = q.non_trainable_copy()
        za = qa.inducing_locations

        if za is None:
            """
            No inducing points yet. Perform standard VI approach, with 
            
            F(q(u)) = -KL(q(u) || p(u)) + E_q(u)[log p(y | f)].
            
            Just set q(u) = t(b) and Z = Z_b.
            """
            z = t.inducing_locations
            q = type(q)(
                inducing_locations=z,
                nat_params=t.nat_params,
            )

        else:
            """
            Perform modified VI approach, with
            
            q(a, b) ⍺ q(a) p(b | a) t(b)

            F(q(f)) = -KL(q(a, b) || p(a, b)) + E_q(a, b)[log p(y | f)]    
                        + terms not depending on t(b) or hyperparameters.
                        
            Just set q(a, b) ⍺ q(a) p(b | a) t(b) and Z = {Z_a, Z_b} and ignore
            the terms not depending on t(b) or hyperparameters.
            """
            zb = t.inducing_locations
            z = torch.cat([za, zb], axis=0)

            # Compute qcav(a, b) = q(a) p(b | a).
            qcav_cov = torch.empty(len(z), len(z))
            qcav_loc = torch.empty(len(z))

            kbb = add_diagonal(self.kernel(zb, zb).evaluate(), JITTER)
            kaa = add_diagonal(self.kernel(za, za).evaluate(), JITTER)
            kba = self.model.kernel(zb, za).evaluate()
            ikaa = psd_inverse(kaa)

            # Covariance and mean of q(a).
            qa_cov = qa.std_params["covariance_matrix"]
            qa_loc = qa.std_params["loc"]

            # Covariance and mean of qcav(b).
            a = kba.matmul(ikaa)
            qcav_bcov = (kbb - a.matmul(kba.T) + a.matmul(qa_cov).matmul(a.T))
            qcav_bloc = a.matmul(qa_loc)

            # Covariance between a and b.
            qcav_acovb = qa_cov.matmul(ikaa).matmul(kba.T)

            # Compute qcav(a, b) = q(a)p(b|a).
            qcav_cov[:len(za), :len(za)] = qa_cov
            qcav_cov[len(za):, len(za):] = qcav_bcov
            qcav_cov[:len(za), len(za):] = qcav_acovb
            qcav_cov[len(za):, :len(za)] = qcav_acovb.T
            qcav_loc[:len(za)] = qa_loc
            qcav_loc[len(za):] = qcav_bloc

            qcav = type(q)(
                inducing_locations=z,
                std_params={
                    "loc": qcav_loc,
                    "covariance_matrix": qcav_cov,
                }
            )

            # Compute q(a, b) ⍺ qcav(a, b) t(b).
            t_np1 = torch.zeros(len(z))
            t_np1[len(za):] = t.nat_params["np1"]
            t_np2 = torch.zeros(len(z), len(z))
            t_np2[len(za):, len(za):] = t.nat_params["np2"]

            q = type(q)(
                inducing_locations=z,
                nat_params={
                    "np1": t_np1 + qcav.nat_params["np1"],
                    "np2": t_np2 + qcav.nat_params["np2"],
                },
            )

        # Everything is the same from here on in.
        kzz = add_diagonal(self.kernel(z, z).evaluate(), JITTER)
        ikzz = psd_inverse(kzz)
        p = type(q)(
            inducing_locations=z,
            std_params={
                "loc": torch.zeros(len(z)),
                "covariance_matrix": kzz,
            }
        )

        # Compute KL divergence between q and p.
        kl = q.kl_divergence(p).sum()

        # Compute E_q[log p(y | f)]
        # = log N(y; E_q[f], σ^2) - 0.5 / (σ ** 2) Var_q[f]
        kxz = self.kernel(x, z).evaluate()
        kxx = add_diagonal(self.kernel(x, x).evaluate(), JITTER)

        a = kxz.matmul(ikzz)
        b = kxx - a.matmul(kxz.T)

        qf_loc = a.matmul(q.std_params["loc"])
        qf_cov = b + a.matmul(q.std_params["covariance_matrix"]).matmul(a.T)

        sigma = self.model.outputsigma
        dist = self.model.likelihood_forward(qf_loc)
        ll1 = dist.log_prob(y.squeeze())
        ll2 = -0.5 * sigma ** (-2) * qf_cov.diag()
        ll = ll1.sum() + ll2.sum()

        elbo = ll - kl

        return elbo


class PrivateSparseGaussianProcessClassification(Model, nn.Module):
    """
    Sparse Gaussian process model with private inducing points.
    """

    conjugate_family = None

    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        # Construct inducing points and kernel.
        if self.hyperparameters["kernel_class"] is not None:
            self.kernel = self.hyperparameters["kernel_class"](
                **self.hyperparameters["kernel_params"]
            )
        else:
            raise ValueError("Kernel class not specified.")

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
            "optimiser_class": optim.Adam,
            "optimiser_params": {"lr": 1e-3},
            "reset_optimiser": True,
            "epochs": 100,
            "num_elbo_samples": 1,
            "num_predictive_samples": 1,
            "print_epochs": 10,
        }

    def forward(self, x, q):
        """
        Returns the (approximate) predictive posterior distribution of a
        sparse Gaussian process.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(u | z).
        :return: ∫ p(y | f, x) p(f | u) q(u) df du.
        """
        qu_loc = q.std_params["loc"]
        qu_cov = q.std_params["covariance_matrix"]
        z = q.inducing_locations

        # Parameters of prior.
        kxz = self.kernel(x, z).evaluate()
        kzx = kxz.T
        kzz = add_diagonal(self.kernel(z, z).evaluate(), JITTER)
        kxx = add_diagonal(self.kernel(x, x).evaluate(), JITTER)

        ikzz = psd_inverse(kzz)

        # Predictive posterior.
        qf_loc = kxz.matmul(ikzz).matmul(qu_loc)
        qf_cov = kxx + kxz.matmul(
            ikzz).matmul(qu_cov - kzz).matmul(ikzz).matmul(kzx)

        qf = distributions.MultivariateNormal(qf_loc, covariance_matrix=qf_cov)
        fs = qf.sample((self.hyperparameters["num_predictive_samples"],))

        comp = distributions.Bernoulli(logits=fs.T)
        mix = distributions.Categorical(torch.ones(len(fs),))

        return distributions.MixtureSameFamily(mix, comp)

    def likelihood_forward(self, x, theta):
        """
        Returns the model's likelihood p(y | x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model.
        :return: p(y | x).
        """
        return distributions.Bernoulli(logits=theta)

    def conjugate_update(self, data, q, t_i):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t_i: The local factor t(θ).
        :return: q_new, t_i_new, the new global posterior and the new local
        contribution.
        """
        raise NotImplementedError

    def local_free_energy(self, data, q, t):
        """
        Returns the local variational free energy (up to an additive constant)
        of the sparse Gaussian process model under q(a) with local factor t(b).
        :param data: The local data.
        :param q: The current global posterior q(a).
        :param t: The local factor t(b).
        :return: The evidence lower bound.
        """
        # Set up data etc.
        x = data["x"]
        y = data["y"]

        # Copy to approximate posterior, making it non-trainable.
        qa = q.non_trainable_copy()
        za = qa.inducing_locations

        if za is None:
            """
            No inducing points yet. Perform standard VI approach, with 

            F(q(u)) = -KL(q(u) || p(u)) + E_q(u)[log p(y | f)].

            Just set q(u) = t(b) and Z = Z_b.
            """
            z = t.inducing_locations
            q = type(q)(
                inducing_locations=z,
                nat_params=t.nat_params,
            )

        else:
            """
            Perform modified VI approach, with

            q(a, b) ⍺ q(a) p(b | a) t(b)

            F(q(f)) = -KL(q(a, b) || p(a, b)) + E_q(a, b)[log p(y | f)]    
                        + terms not depending on t(b) or hyperparameters.

            Just set q(a, b) ⍺ q(a) p(b | a) t(b) and Z = {Z_a, Z_b} and ignore
            the terms not depending on t(b) or hyperparameters.
            """
            zb = t.inducing_locations
            z = torch.cat([za, zb], axis=0)

            # Compute qcav(a, b) = q(a) p(b | a).
            qcav_cov = torch.empty(len(z), len(z))
            qcav_loc = torch.empty(len(z))

            kbb = add_diagonal(self.kernel(zb, zb).evaluate(), JITTER)
            kaa = add_diagonal(self.kernel(za, za).evaluate(), JITTER)
            kba = self.model.kernel(zb, za).evaluate()
            ikaa = psd_inverse(kaa)

            # Covariance and mean of q(a).
            qa_cov = qa.std_params["covariance_matrix"]
            qa_loc = qa.std_params["loc"]

            # Covariance and mean of qcav(b).
            a = kba.matmul(ikaa)
            qcav_bcov = (kbb - a.matmul(kba.T) + a.matmul(qa_cov).matmul(a.T))
            qcav_bloc = a.matmul(qa_loc)

            # Covariance between a and b.
            qcav_acovb = qa_cov.matmul(ikaa).matmul(kba.T)

            # Compute qcav(a, b) = q(a)p(b|a).
            qcav_cov[:len(za), :len(za)] = qa_cov
            qcav_cov[len(za):, len(za):] = qcav_bcov
            qcav_cov[:len(za), len(za):] = qcav_acovb
            qcav_cov[len(za):, :len(za)] = qcav_acovb.T
            qcav_loc[:len(za)] = qa_loc
            qcav_loc[len(za):] = qcav_bloc

            qcav = type(q)(
                inducing_locations=z,
                std_params={
                    "loc": qcav_loc,
                    "covariance_matrix": qcav_cov,
                }
            )

            # Compute q(a, b) ⍺ qcav(a, b) t(b).
            t_np1 = torch.zeros(len(z))
            t_np1[len(za):] = t.nat_params["np1"]
            t_np2 = torch.zeros(len(z), len(z))
            t_np2[len(za):, len(za):] = t.nat_params["np2"]

            q = type(q)(
                inducing_locations=z,
                nat_params={
                    "np1": t_np1 + qcav.nat_params["np1"],
                    "np2": t_np2 + qcav.nat_params["np2"],
                },
            )

        # Everything is the same from here on in.
        kzz = add_diagonal(self.kernel(z, z).evaluate(), JITTER)
        ikzz = psd_inverse(kzz)
        p = type(q)(
            inducing_locations=z,
            std_params={
                "loc": torch.zeros(len(z)),
                "covariance_matrix": kzz,
            }
        )

        # Compute KL divergence between q and p.
        kl = q.kl_divergence(p).sum()

        # Compute E_q[log p(y | f)]
        kxz = self.kernel(x, z).evaluate()
        kzx = kxz.T
        kxx = add_diagonal(self.kernel(x, x).evaluate(), JITTER)
        # Predictive posterior.
        qu_loc = q.std_params["loc"]
        qu_cov = q.std_params["covariance_matrix"]

        qf_loc = kxz.matmul(ikzz).matmul(qu_loc)
        qf_cov = kxx + kxz.matmul(
            ikzz).matmul(qu_cov - kzz).matmul(ikzz).matmul(kzx)

        qf = distributions.MultivariateNormal(
            qf_loc, covariance_matrix=qf_cov)
        fs = qf.sample((self.hyperparameters["num_elbo_samples"],))

        ll = self.likelihood_log_prob(data, fs).mean(0).sum()

        elbo = ll - kl

        return elbo
