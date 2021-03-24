import logging
import torch

from tqdm.auto import tqdm
from torch import distributions
from torch.utils.data import TensorDataset, DataLoader
from pvi.clients.base import Client
from pvi.utils.psd_utils import psd_inverse, add_diagonal, safe_cholesky
from pvi.models.sgp import SparseGaussianProcessRegression

logger = logging.getLogger(__name__)

JITTER = 1e-4


class StreamingSGPClient(Client):
    def __init__(self, data, model, t):
        super().__init__(data, model, t)

    def fit(self, q):
        """
        Computes a refined posterior and its associated approximating
        likelihood term. This method is called directly by the server.
        """

        # Compute new posterior (ignored) and approximating likelihood term
        q, self.t = self.gradient_based_update(q)

        return q

    def gradient_based_update(self, q):
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

        # TODO: How do we constrain these such that q(a, b) is a valid
        #  distribution? For now just ensure t(b) is a valid distribution,
        #  although note that it does not have to be.
        t = type(q)(
            inducing_locations=self.t.inducing_locations.clone(),
            nat_params={
                "np1": self.t.nat_params["np1"].clone(),
                "np2": self.t.nat_params["np2"].clone(),
            },
            is_trainable=True,
            train_inducing=self.t.train_inducing,
        )
        zb = t.inducing_locations

        # Reset optimiser
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, hyper["optimiser"])(
            t.parameters(), **hyper["optimiser_params"])

        # Copy current approximate posterior, ensuring non-trainable.
        qa = q.non_trainable_copy()
        za = qa.inducing_locations

        if za is not None:
            # Fixed during optimisation.
            qa_cov = qa.std_params["covariance_matrix"]
            qa_loc = qa.std_params["loc"]

            kaa = add_diagonal(self.model.kernel(za, za).evaluate().detach(),
                               JITTER)
            ikaa = psd_inverse(kaa)

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

                    Just set q(u) = t(b) and Z = Z_b.
                    """
                    q = t
                    z = zb

                else:
                    """
                    Perform modified VI approach, with

                    q(a, b) ⍺ q(a) p(b | a) t(b)

                    F(q(f)) = -KL(q(a, b) || p(a, b)) + E_q(a, b)[log p(y | f)]    
                                + terms not depending on t(b) or ε.

                    Just set q(a, b) ⍺ q(a) p(b | a) t(b) and Z = {Z_a, Z_b}
                    and ignore the terms not depending on t(b) or ε.
                    """
                    z = torch.cat([za, zb], axis=0)

                    # Compute qcav(a, b) = q(a) p(b | a).
                    qcav_cov = torch.empty(len(z), len(z))
                    qcav_loc = torch.empty(len(z))

                    kbb = add_diagonal(self.model.kernel(zb, zb).evaluate(),
                                       JITTER)
                    kba = self.model.kernel(zb, za).evaluate()

                    a = kba.matmul(ikaa)
                    qcav_bcov = (kbb - a.matmul(kba.T)
                                 + a.matmul(qa_cov).matmul(a.T))
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

                    # Resize natural parameters of t(b).
                    t_np1 = torch.zeros(len(z))
                    t_np1[len(za):] = t.nat_params["np1"]
                    t_np2 = torch.zeros(len(z), len(z))
                    t_np2[len(za):, len(za):] = t.nat_params["np2"]

                    q = type(q)(
                        inducing_locations=z,
                        nat_params={
                            "np1": t_np1 + qcav.nat_params["np1"],
                            "np2": t_np2 + qcav.nat_params["np2"]
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
                    b = kxx - a.matmul(kxz.T)

                    qf_loc = a.matmul(q.std_params["loc"])
                    qf_cov = b + a.matmul(
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
                    kzx = kxz.T
                    kzz = add_diagonal(self.model.kernel(z, z).evaluate(),
                                       JITTER)
                    kxx = add_diagonal(
                        self.model.kernel(x_batch, x_batch).evaluate(),
                        JITTER)
                    ikzz = psd_inverse(kzz)

                    # Predictive posterior.
                    qu_loc = q.std_params["loc"]
                    qu_cov = q.std_params["covariance_matrix"]

                    qf_loc = kxz.matmul(ikzz).matmul(qu_loc)
                    qf_cov = kxx + kxz.matmul(
                        ikzz).matmul(qu_cov - kzz).matmul(ikzz).matmul(kzx)

                    qf_chol = safe_cholesky(qf_cov, min_eps=1e-8, max_eps=1e-3)

                    qf = distributions.MultivariateNormal(
                        qf_loc, scale_tril=qf_chol)

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

        # Put back into factor form.
        t_new = type(self.t)(
            inducing_locations=t.inducing_locations.detach().clone(),
            nat_params={
                "np1": t.nat_params["np1"].detach().clone(),
                "np2": t.nat_params["np2"].detach().clone()
            },
            train_inducing=self.t.train_inducing,
        )
        q_new = q.non_trainable_copy()

        self._can_update = True

        return q_new, t_new
