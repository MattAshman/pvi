import logging
import torch
import numpy as np

from tqdm.auto import tqdm
from .base import *
from pvi.utils.gaussian import joint_from_marginal, nat_from_std, std_from_nat
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.distributions import MultivariateGaussianDistributionWithZ

logger = logging.getLogger(__name__)

JITTER = 1e-1


class SequentialSGPServer(Server):
    def __init__(self, model, q, clients, config=None,
                 maintain_inducing=False):
        super().__init__(model, q, clients, config)

        self.maintain_inducing = maintain_inducing
        self.client_inducing_idx = []

        if not self.maintain_inducing:
            self.q = self.compute_posterior_union()

        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 25,
            "damping_factor": 1.,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 1e-2},
            "epochs": 100,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients)):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t

                # Send over q marginalised at t_i_old.inducing_locations.
                za_i = t_i_old.inducing_locations
                qa_i_old = self.model.posterior(za_i, self.q, diag=False)
                qa_i_old = MultivariateGaussianDistributionWithZ(
                    std_params={
                        "loc": qa_i_old.loc,
                        "covariance_matrix": qa_i_old.covariance_matrix,
                    },
                    inducing_locations=za_i,
                    is_trainable=False,
                    train_inducing=self.q.train_inducing,   # Should be True.
                )

                _, t_i_new = client.fit(qa_i_old)

                logger.debug(
                    "Received client update. Updating global posterior.")

                # Project onto global inducing locations.
                self.q = self.update_posterior(t_i_old, t_i_new)
                # self.q = self.compute_posterior_union()

                clients_updated += 1
                self.communications += 1

                # Log q after each update.
                self.log["q"].append(self.q.non_trainable_copy())
                self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Update hyperparameters.
        if self.config["train_model"] and \
                self.iterations % self.config["model_update_freq"] == 0:
            self.update_hyperparameters()

        self.log["clients_updated"].append(clients_updated)

    def compute_posterior_union(self):
        """
        Computes q(u) = p(u) Π_m t_m(u_m).
        :return: q(u).
        """
        # Store inducing locations, and indices for each client.
        z_idx = []
        z = []
        i = 0
        for client in self.clients:
            zi = client.t.inducing_locations
            z.append(zi)
            z_idx.append(list(range(i, i + len(zi))))
            i = i + len(zi)

        z = torch.cat(z)

        # Prior distribution p(u).
        kzz = self.model.kernel(z, z).detach()
        std_params = {
            "loc": torch.zeros(z.shape[0]), # Assumes zero mean.
            "covariance_matrix": kzz,
        }

        # Compute q(u) = p(u) Π_m t_m(u_m).
        nat_params = nat_from_std(std_params)
        for client_idx, client in enumerate(self.clients):
            np_i = client.t.nat_params
            for i, idx1 in enumerate(z_idx[client_idx]):
                nat_params["np1"][idx1] += np_i["np1"][i]
                for j, idx2 in enumerate(z_idx[client_idx]):
                    nat_params["np2"][idx1, idx2] += np_i["np2"][i, j]

        q = MultivariateGaussianDistributionWithZ(
            nat_params=nat_params,
            inducing_locations=z,
            is_trainable=False,
            train_inducing=True
        )

        return q

    def update_posterior(self, t_old, t_new):
        """
        Computes the projection
        q(f) = argmin KL[q(f) || qold(f) x t_new(bi) / t_old(ai)].
        :param q_old: Old posterior q(f) = p(f | a) q(a).
        :param t_old: Old approximate factor t_old(ai).
        :param t_new: New approximate factor t_new(bi).
        :return: New posterior q(f) = p(f | b) q(b).
        """

        if not self.maintain_inducing:
            return self.compute_posterior_union()

        else:
            # TODO: use collapsed lower bound.
            # TODO: check this passes trainable inducing locations too.
            q_old = self.q.non_trainable_copy()
            q = self.q.trainable_copy()
            za = q_old.inducing_locations
            zbi = t_new.inducing_locations
            zai = t_old.inducing_locations

            # Perturb Za to get Zb (avoids numerical issues).
            q.inducing_locations = q.inducing_locations + torch.randn_like(
                q.inducing_locations) * JITTER
            zb = q.inducing_locations

            # This remains fixed throughout optimisation.
            kaa = self.model.kernel(za, za).detach()
            ikaa = psd_inverse(kaa)

            optimiser = getattr(torch.optim, self.config["optimiser"])(
                q.parameters(), **self.config["optimiser_params"])
            optimiser.zero_grad()

            # Dict for logging optimisation progress
            training_curve = {
                "elbo": [],
                "kl": [],
                "logt_new": [],
                "logt_old": [],
            }

            # Gradient-based optimisation loop -- loop over epochs
            epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
                              leave=True)
            # for i in range(self.config["epochs"]):
            for _ in epoch_iter:
                epoch = {
                    "elbo": 0,
                    "kl": 0,
                    "logt_new": 0,
                    "logt_old": 0,
                }

                zab = torch.cat([za, zb])
                zaibi = torch.cat([zai, zbi])
                z = torch.cat([za, zb, zai, zbi])

                kab = self.model.kernel(za, zb)
                kbb = self.model.kernel(zb, zb)
                kabab = self.model.kernel(zab, zab)
                ikbb = psd_inverse(kbb)
                ikabab = psd_inverse(kabab)

                kabaibi = self.model.kernel(zab, zaibi)
                kaibiaibi = self.model.kernel(zaibi, zaibi)

                # q(a, b) = q(b) p(a | b).
                # Remember to order as q(a, b), not q(b, a).
                qab_loc, qab_cov = joint_from_marginal(
                    q, kab.T, kbb=kaa, ikaa=ikbb, b_then_a=True)
                qab = type(q)(
                    inducing_locations=zab,
                    std_params={
                        "loc": qab_loc,
                        "covariance_matrix": qab_cov,
                    }
                )
                # q(a, b, ai, bi) = q(a, b) p(ai, bi | a, b).
                qz_loc, qz_cov = joint_from_marginal(
                    qab, kabaibi, kbb=kaibiaibi, ikaa=ikabab
                )

                qz = type(q)(
                    inducing_locations=z,
                    std_params={
                        "loc": qz_loc,
                        "covariance_matrix": qz_cov,
                    }
                )

                # q_old(a, b) = q_old(a) p(b | a).
                qab_old_loc, qab_old_cov = joint_from_marginal(
                    q_old, kab, kbb=kbb, ikaa=ikaa)
                qab_old = type(q)(
                    inducing_locations=zab,
                    std_params={
                        "loc": qab_old_loc,
                        "covariance_matrix": qab_old_cov,
                    }
                )
                # q(a, b, ai, bi) = q(a, b) p(ai, bi | a, b).
                qz_old_loc, qz_old_cov = joint_from_marginal(
                    qab_old, kabaibi, kbb=kaibiaibi, ikaa=ikabab
                )
                qz_old_std = {
                    "loc": qz_old_loc,
                    "covariance_matrix": qz_old_cov,
                }
                qz_old_np = nat_from_std(qz_old_std)

                # Create tilted as q_tilt = q_old(f) * t(bi) / t(ai).
                qz_tilt_np = qz_old_np

                # Add t(bi) (last len(zbi) rows and columns of nat params).
                for i in range(len(zbi)):
                    qz_tilt_np["np1"][-len(zbi)+i] += (
                            t_new.nat_params["np1"][i])
                    for j in range(len(zbi)):
                        qz_tilt_np["np2"][-len(zbi)+i, -len(zbi)+j] += (
                                t_new.nat_params["np2"][i, j])

                # Subtract t(ai) (comes before rows and columns of zbi).
                for i in range(len(zai)):
                    qz_tilt_np["np1"][-len(zaibi)+i] -= (
                        t_old.nat_params["np1"][i])
                    for j in range(len(zai)):
                        qz_tilt_np["np2"][-len(zaibi)+i, -len(zaibi)+j] -= (
                            t_old.nat_params["np2"][i, j])

                qz_tilt = type(q)(inducing_locations=z, nat_params=qz_tilt_np)

                # KL[q(a, b) || q_old(a, b)].
                try:
                    kl = qz.kl_divergence(qz_tilt).sum()
                except:
                    import pdb
                    pdb.set_trace()
                    kl = 0

                loss = kl
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] = -loss.item()
                epoch["kl"] = kl.item()

                # Log progress for current epoch
                training_curve["elbo"].append(epoch["elbo"])
                training_curve["kl"].append(epoch["kl"])
                training_curve["logt_new"].append(epoch["logt_new"])
                training_curve["logt_old"].append(epoch["logt_old"])

                epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"])

            # Log the training curves for this update
            self.log["training_curves"].append(training_curve)

            # Create non-trainable copy to send back to server.
            q_new = q.non_trainable_copy()

            try:
                q_new.std_params["covariance_matrix"].cholesky()
            except:
                import pdb
                pdb.set_trace()
                print("wtf")

            return q_new

        # else:
        #     # TODO: use collapsed lower bound.
        #     # TODO: check this passes trainable inducing locations too.
        #     q_old = self.q.non_trainable_copy()
        #     q = self.q.trainable_copy()
        #     za = q_old.inducing_locations
        #     zbi = t_new.inducing_locations
        #     zai = t_old.inducing_locations
        #
        #     # Perturb Za to get Zb (avoids numerical issues).
        #     q.inducing_locations = q.inducing_locations + torch.randn_like(
        #         q.inducing_locations) * JITTER
        #     zb = q.inducing_locations
        #
        #     # This remains fixed throughout optimisation.
        #     kaa = self.model.kernel(za, za).detach()
        #     ikaa = psd_inverse(kaa)
        #
        #     optimiser = getattr(torch.optim, self.config["optimiser"])(
        #         q.parameters(), **self.config["optimiser_params"])
        #     optimiser.zero_grad()
        #
        #     # Dict for logging optimisation progress
        #     training_curve = {
        #         "elbo": [],
        #         "kl": [],
        #         "logt_new": [],
        #         "logt_old": [],
        #     }
        #
        #     # Gradient-based optimisation loop -- loop over epochs
        #     epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
        #                       leave=True)
        #     # for i in range(self.config["epochs"]):
        #     for _ in epoch_iter:
        #         epoch = {
        #             "elbo": 0,
        #             "kl": 0,
        #             "logt_new": 0,
        #             "logt_old": 0,
        #         }
        #
        #         z = torch.cat([za, zb])
        #
        #         kab = self.model.kernel(za, zb)
        #         kbb = self.model.kernel(zb, zb)
        #         ikbb = psd_inverse(kbb)
        #
        #         # q(a, b) = q(b) p(a | b).
        #         # Remember to order as q(a, b), not q(b, a).
        #         qab_loc, qab_cov = joint_from_marginal(
        #             q, kab.T, kbb=kaa, ikaa=ikbb, b_then_a=True)
        #         qab = type(q)(
        #             inducing_locations=z,
        #             std_params={
        #                 "loc": qab_loc,
        #                 "covariance_matrix": qab_cov,
        #             }
        #         )
        #
        #         # q_old(a, b) = q_old(a) p(b | a).
        #         qab_old_loc, qab_old_cov = joint_from_marginal(
        #             q_old, kab, kbb=kbb, ikaa=ikaa)
        #         qab_old = type(q)(
        #             inducing_locations=z,
        #             std_params={
        #                 "loc": qab_old_loc,
        #                 "covariance_matrix": qab_old_cov,
        #             }
        #         )
        #
        #         # KL[q(a, b) || q_old(a, b)].
        #         kl = qab.kl_divergence(qab_old).sum()
        #
        #         # E_q[log t_new(b)] - E_q[log t_old(a)].
        #         qbi = self.model.posterior(zbi, q, diag=False)
        #         qbi = MultivariateGaussianDistributionWithZ(
        #             std_params={
        #                 "loc": qbi.loc,
        #                 "covariance_matrix": qbi.covariance_matrix,
        #             },
        #             inducing_locations=zbi,
        #         )
        #         # logt_new = t_new.eqlogt(qbi, self.config["num_elbo_samples"])
        #         bis = qbi.rsample((self.config["num_elbo_samples"],))
        #         logt_new = t_new(bis).mean()
        #
        #         qai = self.model.posterior(zai, q, diag=False)
        #         qai = MultivariateGaussianDistributionWithZ(
        #             std_params={
        #                 "loc": qai.loc,
        #                 "covariance_matrix": qai.covariance_matrix,
        #             },
        #             inducing_locations=zai,
        #         )
        #         # logt_old = t_old.eqlogt(qai)
        #         ais = qai.rsample((self.config["num_elbo_samples"],))
        #         logt_old = t_old(ais).mean()
        #
        #         loss = kl + logt_new - logt_old
        #         loss.backward()
        #         optimiser.step()
        #         optimiser.zero_grad()
        #
        #         # Keep track of quantities for current batch
        #         # Will be very slow if training on GPUs.
        #         epoch["elbo"] = -loss.item()
        #         epoch["kl"] = kl.item()
        #         epoch["logt_new"] = logt_new.item()
        #         epoch["logt_old"] = logt_old.item()
        #
        #         # Log progress for current epoch
        #         training_curve["elbo"].append(epoch["elbo"])
        #         training_curve["kl"].append(epoch["kl"])
        #         training_curve["logt_new"].append(epoch["logt_new"])
        #         training_curve["logt_old"].append(epoch["logt_old"])
        #
        #         epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
        #                                logt_new=epoch["logt_new"],
        #                                logt_old=epoch["logt_old"])
        #
        #     # Log the training curves for this update
        #     self.log["training_curves"].append(training_curve)
        #
        #     # Create non-trainable copy to send back to server.
        #     q_new = q.non_trainable_copy()
        #
        #     return q_new

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1