import logging
import torch

from tqdm.auto import tqdm
from torch import distributions, nn
from torch.utils.data import TensorDataset, DataLoader
from pvi.clients.base import ContinualLearningClient
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.models.sgp import SparseGaussianProcessRegression
from pvi.distributions.gp_distributions import \
    MultivariateGaussianDistributionWithZ

logger = logging.getLogger(__name__)

JITTER = 1e-6


class StreamingSGPClient(ContinualLearningClient):
    def __init__(self, data, model, inducing_locations):
        super().__init__(data, model)

        # Private inducing locations Z_b.
        self.inducing_locations = inducing_locations

    def fit(self, q):
        """
        Computes a refined posterior and its associated approximating
        likelihood term. This method is called directly by the server.
        """
        # Compute new posterior (ignored) and approximating likelihood term
        q = self.gradient_based_update(q)

        return q

    def gradient_based_update(self, q):
        """
        The gradient based update in the streaming SGP setting involves
        completely overhalling the current approximate posterior q with new
        inducing points, hence we override the default gradient_based_update
        function.
        :param q: The current approximate posterior, q(a | Z_a).
        :return q_new: The new approximate posterior, q(a, b | Z_a, Z_b).
        """
        hyper = self.model.hyperparameters

        # Cannot update during optimisation.
        self._can_update = False

        # Set up data etc.
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=hyper["batch_size"],
                            shuffle=True)

        # Copy current approximate posterior, ensuring non-trainable.
        qa = q.non_trainable_copy()
        za = qa.inducing_locations

        # Parameterise as q(b | a) N(b; mb + Aa, Sb), initialised as
        # q(b | a) = p(b | a).
        zb = self.inducing_locations
        mb = torch.zeros(len(zb))

        if za is not None:
            # Fixed during optimisation.
            qa_cov = qa.std_params["covariance_matrix"]
            qa_loc = qa.std_params["loc"]

            kaa = add_diagonal(self.model.kernel(za, za).evaluate().detach(),
                               JITTER)
            ikaa = psd_inverse(kaa)
            kbb = add_diagonal(self.model.kernel(zb, zb).evaluate().detach(),
                               JITTER)
            kba = self.model.kernel(zb, za).evaluate().detach()
            # Initialise Sb = Kbb - Kba Kaa^{-1} Kab.
            sb = kbb - kba.matmul(ikaa).matmul(kba.T)
            sb_chol = torch.cholesky(sb)
        else:
            # Initialise Sb = Kbb.
            kbb = add_diagonal(self.model.kernel(zb, zb).evaluate().detach(),
                               JITTER)
            sb = kbb
            sb_chol = torch.cholesky(sb)

        # Variational parameters.
        zb = nn.Parameter(zb, requires_grad=True)
        mb = nn.Parameter(mb, requires_grad=True)
        sb_chol = nn.Parameter(sb_chol, requires_grad=True)
        variational_parameters = nn.ParameterList([zb, mb, sb_chol])

        # Reset optimiser
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, hyper["optimiser"])(
            variational_parameters, **hyper["optimiser_params"])

        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "kl": [],
            "ll": [],
        }

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(hyper["epochs"]), "Epochs")
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "kl": 0,
                "ll": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch
                }

                if za is None:
                    """
                    No inducing points yet. Perform standard VI approach, with 

                    F(q(u)) = -KL(q(u) || p(u)) + E_q(u)[log p(y | f)].

                    Just set q(u) = q(b) and Z = Z_b.
                    """
                    q = MultivariateGaussianDistributionWithZ(
                        inducing_locations=zb,
                        std_params={
                            "loc": mb,
                            "covariance_matrix": sb_chol.matmul(sb_chol.T)
                        }
                    )
                    z = zb

                else:
                    """
                    Perform modified VI approach, with

                    q(a, b) ⍺ q(a) p(b | a) t(b)

                    F(q(f)) = -KL(q(a, b) || p(a, b)) + E_q(a, b)[log p(y | f)]    
                                + terms not depending on t(b) or ε.

                    Just set q(a, b) ⍺ q(a) q(b | a) and Z = {Z_a, Z_b}
                    and ignore the terms not depending on t(b) or ε.
                    """
                    z = torch.cat([za, zb], axis=0)

                    kba = self.model.kernel(zb, za).evaluate().detach()
                    a = kba.matmul(ikaa)

                    q_loc = torch.empty(len(z))
                    q_cov = torch.empty(len(z), len(z))

                    q_loc[:len(za)] = qa_loc
                    q_loc[len(za):] = mb + a.matmul(qa_loc)

                    q_cov[:len(za), :len(za)] = qa_cov
                    q_cov[len(za):, len(za):] = (
                            sb_chol.matmul(sb_chol.T)
                            + a.matmul(qa_cov).matmul(a.T))
                    q_cov[len(za):, :len(za)] = qa_cov.matmul(a.T)
                    q_cov[:len(za), len(za):] = a.matmul(qa_cov)

                    q = type(q)(
                        inducing_locations=z,
                        std_params={
                            "loc": q_loc,
                            "covariance_matrix": q_cov
                        },
                    )

                # Everything is the same from here on in.
                kzz = add_diagonal(self.model.kernel(z, z).evaluate(),
                                   JITTER)
                ikzz = psd_inverse(kzz)
                p = type(q)(
                    inducing_locations=z,
                    std_params={
                        "loc": torch.zeros(len(z)),
                        "covariance_matrix": kzz,
                    }
                )

                # Compute KL(q || p).
                kl = q.kl_divergence(p).sum()

                # Compute E_q[log p(y | f)].
                if (str(type(self.model))
                        == str(SparseGaussianProcessRegression)):
                    """
                    Can compute E_q[log p(y | f)] in closed form:
                    
                    = log N(y; E_q[f], σ^2) - 0.5 / (σ ** 2) Var_q[f].
                    """
                    kxz = self.model.kernel(x, z).evaluate()
                    kxx = add_diagonal(self.model.kernel(x, x).evaluate(),
                                       JITTER)

                    a = kxz.matmul(ikzz)
                    c = kxx - a.matmul(kxz.T)

                    qf_loc = a.matmul(q.std_params["loc"])
                    qf_cov = c + a.matmul(
                        q.std_params["covariance_matrix"]).matmul(a.T)

                    sigma = self.model.outputsigma
                    dist = self.model.likelihood_forward(qf_loc)
                    ll1 = dist.log_prob(y.squeeze())
                    ll2 = -0.5 * sigma ** (-2) * qf_cov.diag()
                    ll = ll1.sum() + ll2.sum()
                else:
                    """
                    Cannot compute in closed form---use MC estimate instead.
                    """
                    # Parameters of prior.
                    kxz = self.model.kernel(x_batch, z).evaluate()
                    kzz = add_diagonal(self.model.kernel(z, z).evaluate(),
                                       JITTER)
                    kxx = add_diagonal(
                        self.model.kernel(x_batch, x_batch).evaluate(),
                        JITTER)
                    ikzz = psd_inverse(kzz)

                    a = kxz.matmul(ikzz)
                    c = kxx - a.matmul(kxz.T)

                    qf_loc = a.matmul(q.std_params["loc"])
                    qf_cov = c + a.matmul(
                        q.std_params["covariance_matrix"]).matmul(a.T)

                    qf = distributions.MultivariateNormal(
                        qf_loc, covariance_matrix=qf_cov)
                    fs = qf.rsample(
                        (self.model.hyperparameters["num_elbo_samples"],))
                    ll = self.model.likelihood_log_prob(
                        batch, fs).mean(0).sum()

                # Normalise values w.r.t. batch size.
                kl /= len(x)
                ll /= len(x_batch)

                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item()
                epoch["kl"] += kl.item()
                epoch["ll"] += ll.item()

                epoch_iter.set_postfix(elbo=-loss.item(), kl=kl.item(),
                                       ll=ll.item())

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if i % hyper["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"Epochs: {i}.")

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Update clients inducing points.
        self.inducing_locations = zb.detach()

        # Create non_trainable_copy to send back to server.
        q_new = q.non_trainable_copy()

        self._can_update = True

        return q_new
