import logging
import torch

from collections import defaultdict
from .base import Client
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.utils.gaussian import joint_from_marginal, nat_from_std, std_from_nat
from pvi.distributions import MultivariateGaussianDistributionWithZ
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

JITTER = 1e-4


class FederatedSGPClient(Client):
    def __init__(self, data, model, t, config=None):
        super().__init__(data, model, t, config)

    def gradient_based_update(self, p, init_q=None):
        # TODO: use the collapsed variational lower bound for SGP regression.
        #  i.e. should be closed-form solution.
        # Cannot update during optimisation.
        self._can_update = False

        # Check server has marginalised properly.
        assert torch.allclose(p.inducing_locations.detach(),
                              self.t.inducing_locations), \
            "Inducing location of q and self.t do not match."

        # Copy the approximate posterior, make old posterior non-trainable.
        # Compute q_cav by subtracting old natural parameters. Note that q is
        # current specified at the old inducing locations, Za.
        q_old = p.non_trainable_copy()
        q_cav = p.non_trainable_copy()
        q_cav.nat_params = {k: v - self.t.nat_params[k]
                            for k, v in q_cav.nat_params.items()}

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            q = p.trainable_copy()

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            if "model_optimiser_params" in self.config:
                parameters = [
                    {"params": q.parameters()},
                    {"params": self.model.parameters(),
                     **self.config["model_optimiser_params"]}
                ]
            else:
                parameters = [
                    {"params": q.parameters()},
                    {"params": self.model.parameters()}
                ]
        else:
            parameters = q.parameters()

        # Reset optimiser.
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])
        optimiser.zero_grad()

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

        # Dict for logging optimisation progress
        training_curve = defaultdict(list)

        # Stay fixed throughout optimisation as model hyperparameters are
        # only updated by the server.
        za = q_cav.inducing_locations
        kaa = add_diagonal(self.model.kernel(za, za).detach(), JITTER)
        ikaa = psd_inverse(kaa)

        # Compute joint distributions q(a, b) and qcav(a, b).
        # Zb are new private inducing locations, which are to be optimised.
        zb = q.inducing_locations
        z = torch.cat([za, zb], axis=0)

        kab = self.model.kernel(za, zb)
        kbb = add_diagonal(self.model.kernel(zb, zb), JITTER)
        ikbb = psd_inverse(kbb)

        # q(a, b) = q(b) p(a | b). Remember to order as q(a, b), not q(b, a).
        qab_loc, qab_cov = joint_from_marginal(
            q, kab.T, kbb=kaa, ikaa=ikbb, b_then_a=True)

        # q_cav(a, b) = q_cav(a) p(b | a).
        # qab_cav_loc, qab_cav_cov = joint_from_marginal(
        #     q_cav, kab, kbb=kbb, ikaa=ikaa)
        qab_cav_loc, qab_cav_cov = joint_from_marginal(
            q_old, kab, kbb=kbb, ikaa=ikaa)

        qab_cav_np = nat_from_std(std_params={
            "loc": qab_cav_loc,
            "covariance_matrix": qab_cav_cov,
        })

        qab_cav_np["np1"][:len(za)] -= self.t.nat_params["np1"]
        qab_cav_np["np2"][:len(za), :len(za)] -= self.t.nat_params["np2"]


        qab = type(q)(
            inducing_locations=z,
            std_params={
                "loc": qab_loc,
                "covariance_matrix": qab_cov,
            }
        )

        qab_cav = type(q)(inducing_locations=z, nat_params=qab_cav_np)

        # qab_old = type(q)(
        #     inducing_locations=z,
        #     std_params={
        #         "loc": qab_old_loc,
        #         "covariance_matrix": qab_old_cov,
        #     }
        # )

        # Gradient-based optimisation loop.
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
                          leave=True)
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.)

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and q_old.
                # kl = qab.kl_divergence(qab_old).sum() / len(x)
                kl = qab.kl_divergence(qab_cav, calc_log_ap=False).sum()
                kl /= len(x)

                # Sample θ from q and compute p(y | θ, x) for each θ.
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]).sum()
                ll /= len(x_batch)

                # E_q[log t(θ)].
                # qa = self.model.posterior(za, q, diag=False)
                # qa = MultivariateGaussianDistributionWithZ(
                #     std_params={
                #         "loc": qa.loc,
                #         "covariance_matrix": qa.covariance_matrix,
                #     },
                #     inducing_locations=za,
                # )
                # logt = self.t.eqlogt(qa, self.config["num_elbo_samples"])
                # logt /= len(x)

                loss = kl - ll
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                # Recompute joint distributions q(a, b) and qcav(a, b).
                zb = q.inducing_locations
                z = torch.cat([za, zb], axis=0)

                kab = self.model.kernel(za, zb)
                kbb = add_diagonal(self.model.kernel(zb, zb), JITTER)
                ikbb = psd_inverse(kbb)

                # q(a, b) = q(b) p(a | b).
                qab_loc, qab_cov = joint_from_marginal(
                    q, kab.T, kbb=kaa, ikaa=ikbb, b_then_a=True)

                # q_cav(a, b) = q_cav(a) p(b | a).
                # qab_cav_loc, qab_cav_cov = joint_from_marginal(
                #     q_cav, kab, kbb=kbb, ikaa=ikaa)
                qab_cav_loc, qab_cav_cov = joint_from_marginal(
                    q_old, kab, kbb=kbb, ikaa=ikaa)

                qab_cav_np = nat_from_std(std_params={
                    "loc": qab_cav_loc,
                    "covariance_matrix": qab_cav_cov,
                })

                qab_cav_np["np1"][:len(za)] -= self.t.nat_params["np1"]
                qab_cav_np["np2"][:len(za), :len(za)] -= self.t.nat_params[
                    "np2"]

                qab = type(q)(
                    inducing_locations=z,
                    std_params={
                        "loc": qab_loc,
                        "covariance_matrix": qab_cov,
                    }
                )

                qab_cav = type(q)(inducing_locations=z, nat_params=qab_cav_np)

                # qab_old = type(q)(
                #     inducing_locations=z,
                #     std_params={
                #         "loc": qab_old_loc,
                #         "covariance_matrix": qab_old_cov,
                #     }
                # )

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)
                # epoch["logt"] += logt.item() / len(loader)

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])
            training_curve["logt"].append(epoch["logt"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"log t: {epoch['logt']:.3f}, "
                             f"Epochs: {i}.")

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"], logt=epoch["logt"])

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Create non-trainable copy to send back to server.
        q_new = q.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        # Compute new local contribution from old distributions.
        # t(b) = ∫ (q(a, b) / q_cav(a, b)) da

        # q_old(a, b) = q_old(a) p(b | a).
        qab_old_loc, qab_old_cov = joint_from_marginal(
            q_old, kab, kbb=kbb, ikaa=ikaa)

        qab_old = type(q)(
            inducing_locations=z,
            std_params={
                "loc": qab_old_loc,
                "covariance_matrix": qab_old_cov,
            }
        )

        # Compute t(a, b) = [q(a, b) / q_old(a, b)] ** λ * t_old(a).
        tab_np = {k: (v.detach() - qab_old.nat_params[k].detach())
                  * self.config["damping_factor"]
                  for k, v in qab.nat_params.items()}
        tab_np["np1"][:len(za)] += self.t.nat_params["np1"]
        tab_np["np2"][:len(za), :len(za)] += self.t.nat_params["np2"]

        # Marginalise to get t(b) = ∫ t(a, b) da.
        # TODO: t(a, b) must be a valid distribution to do this.
        tab_std = std_from_nat(tab_np)
        tb_std = {
            "loc": tab_std["loc"][len(za):],
            "covariance_matrix": tab_std["covariance_matrix"][
                                 len(za):, len(za):],
        }

        tb_np = nat_from_std(tb_std)
        t_new = type(self.t)(
            inducing_locations=zb.detach(),
            nat_params=tb_np,
        )

        return q_new, t_new
