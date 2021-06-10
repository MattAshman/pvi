import logging
import torch
import numpy as np

from tqdm.auto import tqdm
from .base import *
from pvi.utils.gaussian import joint_from_marginal, nat_from_std
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.distributions import MultivariateGaussianDistributionWithZ

logger = logging.getLogger(__name__)

JITTER = 1e-4
MIN_EIGVAL = 1e-3


class SGPServer(Server):
    def __init__(self, model, p, clients, config=None, init_q=None,
                 maintain_inducing=False):
        super().__init__(model, p, clients, config, init_q)

        self.maintain_inducing = maintain_inducing
        self.client_inducing_idx = []

        if not self.maintain_inducing:
            # q(u) = p(u) Π_m t_m(u_m).
            self.q = self.compute_posterior_union()

        # self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 25,
            "damping_factor": 1.,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 1e-2},
            "lr_scheduler": "MultiplicativeLR",
            "lr_scheduler_params": {
                "lr_lambda": lambda epoch: 1.
            },
            "early_stopping": lambda elbo: False,
            "epochs": 100,
        }

    def tick(self):
        raise NotImplementedError

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
        std_q = {
            "loc": torch.zeros(kzz.shape[:-1]).double(),
            "covariance_matrix": kzz,
        }

        # Compute q(u) = p(u) Π_m t_m(u_m).
        np_q = nat_from_std(std_q)

        for client_idx, client in enumerate(self.clients):
            np_i = client.t.nat_params
            for i, idx1 in enumerate(z_idx[client_idx]):
                np_q["np1"][..., idx1] += np_i["np1"][..., i]
                for j, idx2 in enumerate(z_idx[client_idx]):
                    np_q["np2"][..., idx1, idx2] += np_i["np2"][..., i, j]

        # # TODO: what effects does this have?
        # # Now constrain to be valid distribution.
        # if len(np_q["np1"].shape) == 2:
        #     for i in range(len(np_q["np2"])):
        #         (eigvals_, eigvecs) = np_q["np2"][i].eig(eigenvectors=True)
        #
        #         #  Assume all real eigenvalues.
        #         eigvals = eigvals_[:, 0]
        #
        #         # Constrain to be negative.
        #         eigvals[eigvals >= 0] = -MIN_EIGVAL
        #
        #         # Reconstruct np2.
        #         np_q["np2"][i] = eigvecs.matmul(
        #             eigvals.diag_embed().matmul(eigvecs.T))
        #
        # elif len(np_q["np1"].shape) == 1:
        #     (eigvals_, eigvecs) = np_q["np2"].eig(eigenvectors=True)
        #
        #     #  Assume all real eigenvalues.
        #     eigvals = eigvals_[:, 0]
        #
        #     # Constrain to be negative.
        #     eigvals[eigvals >= 0] = -MIN_EIGVAL
        #
        #     # Reconstruct np2.
        #     np_q["np2"] = eigvecs.matmul(
        #         eigvals.diag_embed().matmul(eigvecs.T))
        #
        # else:
        #     raise ValueError("Not implemented for more than a single batch "
        #                      "dimension.")

        q = MultivariateGaussianDistributionWithZ(
            nat_params=np_q,
            inducing_locations=z,
            is_trainable=False,
            train_inducing=True
        )

        return q

    def update_posterior(self, t_olds, t_news):
        """
        Computes the projection
        q(f) = argmin KL[q(f) || qold(f) x t_new(bi) / t_old(ai)].
        :param t_olds: Old approximate factors t_old(ai).
        :param t_news: New approximate factors t_new(bi).
        :return: New posterior q(f) = p(f | b) q(b).
        """

        if not self.maintain_inducing:
            return self.compute_posterior_union()

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
        #     kaa = add_diagonal(self.model.kernel(za, za), JITTER).detach()
        #     ikaa = psd_inverse(kaa)
        #
        #     optimiser = getattr(torch.optim, self.config["optimiser"])(
        #         q.parameters(), **self.config["optimiser_params"])
        #     optimiser.zero_grad()
        #
        #     # Dict for logging optimisation progress
        #     training_curve = defaultdict(list)
        #
        #     # Gradient-based optimisation loop -- loop over epochs
        #     epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
        #                       leave=True)
        #     # for i in range(self.config["epochs"]):
        #     for _ in epoch_iter:
        #         epoch = defaultdict(lambda: 0.)
        #
        #         zab = torch.cat([za, zb])
        #         zaibi = torch.cat([zai, zbi])
        #         z = torch.cat([za, zb, zai, zbi])
        #
        #         kab = self.model.kernel(za, zb)
        #         kbb = add_diagonal(self.model.kernel(zb, zb), JITTER)
        #         kabab = add_diagonal(self.model.kernel(zab, zab), JITTER)
        #         ikbb = psd_inverse(kbb)
        #         ikabab = psd_inverse(kabab)
        #
        #         kabaibi = self.model.kernel(zab, zaibi)
        #         kaibiaibi = add_diagonal(self.model.kernel(zaibi, zaibi),
        #                                  JITTER)
        #
        #         # q(a, b) = q(b) p(a | b).
        #         # Remember to order as q(a, b), not q(b, a).
        #         qab_loc, qab_cov = joint_from_marginal(
        #             q, kab.T, kbb=kaa, ikaa=ikbb, b_then_a=True)
        #         qab = type(q)(
        #             inducing_locations=zab,
        #             std_params={
        #                 "loc": qab_loc,
        #                 "covariance_matrix": qab_cov,
        #             }
        #         )
        #         # q(a, b, ai, bi) = q(a, b) p(ai, bi | a, b).
        #         qz_loc, qz_cov = joint_from_marginal(
        #             qab, kabaibi, kbb=kaibiaibi, ikaa=ikabab
        #         )
        #
        #         qz = type(q)(
        #             inducing_locations=z,
        #             std_params={
        #                 "loc": qz_loc,
        #                 "covariance_matrix": qz_cov,
        #             }
        #         )
        #
        #         # q_old(a, b) = q_old(a) p(b | a).
        #         qab_old_loc, qab_old_cov = joint_from_marginal(
        #             q_old, kab, kbb=kbb, ikaa=ikaa)
        #         qab_old = type(q)(
        #             inducing_locations=zab,
        #             std_params={
        #                 "loc": qab_old_loc,
        #                 "covariance_matrix": qab_old_cov,
        #             }
        #         )
        #         # q(a, b, ai, bi) = q(a, b) p(ai, bi | a, b).
        #         qz_old_loc, qz_old_cov = joint_from_marginal(
        #             qab_old, kabaibi, kbb=kaibiaibi, ikaa=ikabab
        #         )
        #         qz_old_std = {
        #             "loc": qz_old_loc,
        #             "covariance_matrix": qz_old_cov,
        #         }
        #         qz_old_np = nat_from_std(qz_old_std)
        #
        #         # Create tilted as q_tilt = q_old(f) * (t(bi) / t(ai)) ** λ.
        #         qz_tilt_np = qz_old_np
        #
        #         # Add t(bi) (last len(zbi) rows and columns of nat params).
        #         for i in range(len(zbi)):
        #             qz_tilt_np["np1"][-len(zbi)+i] += (
        #                     t_new.nat_params["np1"][i]
        #                     * self.config["damping_factor"])
        #             for j in range(len(zbi)):
        #                 qz_tilt_np["np2"][-len(zbi)+i, -len(zbi)+j] += (
        #                         t_new.nat_params["np2"][i, j]
        #                         * self.config["damping_factor"])
        #
        #         # Subtract t(ai) (comes before rows and columns of zbi).
        #         for i in range(len(zai)):
        #             qz_tilt_np["np1"][-len(zaibi)+i] -= (
        #                 t_old.nat_params["np1"][i]
        #                 * self.config["damping_factor"])
        #             for j in range(len(zai)):
        #                 qz_tilt_np["np2"][-len(zaibi)+i, -len(zaibi)+j] -= (
        #                     t_old.nat_params["np2"][i, j]
        #                     * self.config["damping_factor"])
        #
        #         qz_tilt = type(q)(inducing_locations=z, nat_params=qz_tilt_np)
        #
        #         # KL[q(a, b) || q_old(a, b)].
        #         try:
        #             kl = qz.kl_divergence(qz_tilt, calc_log_ap=False).sum()
        #         except RuntimeError:
        #             import pdb
        #             pdb.set_trace()
        #
        #         loss = kl
        #         loss.backward()
        #         optimiser.step()
        #         optimiser.zero_grad()
        #
        #         # Keep track of quantities for current batch
        #         # Will be very slow if training on GPUs.
        #         epoch["elbo"] = -loss.item()
        #         epoch["kl"] = kl.item()
        #
        #         # Log progress for current epoch
        #         training_curve["elbo"].append(epoch["elbo"])
        #         training_curve["kl"].append(epoch["kl"])
        #         training_curve["logt_new"].append(epoch["logt_new"])
        #         training_curve["logt_old"].append(epoch["logt_old"])
        #
        #         epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"])
        #
        #     # Log the training curves for this update
        #     self.log["training_curves"].append(training_curve)
        #
        #     # Create non-trainable copy to send back to server.
        #     q_new = q.non_trainable_copy()
        #
        #     try:
        #         q_new.std_params["covariance_matrix"].cholesky()
        #     except:
        #         import pdb
        #         pdb.set_trace()
        #         print("wtf")
        #
        #     return q_new

        else:
            # TODO: use collapsed lower bound.
            # TODO: check this passes trainable inducing locations too.
            q_old = self.q.non_trainable_copy()
            q = self.q.trainable_copy()
            za = q_old.inducing_locations

            # Perturb Za to get Zb (avoids numerical issues).
            q.inducing_locations = q.inducing_locations + torch.randn_like(
                q.inducing_locations) * JITTER
            zb = q.inducing_locations

            # This remains fixed throughout optimisation.
            kaa = add_diagonal(self.model.kernel(za, za), JITTER).detach()
            ikaa = psd_inverse(kaa)

            optimiser = getattr(torch.optim, self.config["optimiser"])(
                q.parameters(), **self.config["optimiser_params"])
            lr_scheduler = getattr(torch.optim.lr_scheduler,
                                   self.config["lr_scheduler"])(
                optimiser, **self.config["lr_scheduler_params"])
            optimiser.zero_grad()

            # Dict for logging optimisation progress.
            training_metrics = defaultdict(list)

            # Gradient-based optimisation loop.
            epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
                              leave=True)
            # for i in range(self.config["epochs"]):
            for _ in epoch_iter:
                epoch = defaultdict(lambda: 0.)

                z = torch.cat([za, zb])
                kab = self.model.kernel(za, zb)
                kbb = add_diagonal(self.model.kernel(zb, zb), JITTER)
                ikbb = psd_inverse(kbb)

                # q(a, b) = q(b) p(a | b).
                # Remember to order as q(a, b), not q(b, a).
                qab_loc, qab_cov = joint_from_marginal(
                    q, kab.T, kbb=kaa, ikaa=ikbb, b_then_a=True)
                qab = type(q)(
                    inducing_locations=z,
                    std_params={
                        "loc": qab_loc,
                        "covariance_matrix": qab_cov,
                    }
                )

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

                # KL[q(a, b) || q_old(a, b)].
                kl = qab.kl_divergence(qab_old).sum()

                # E_q[log t_new(b)] - E_q[log t_old(a)].
                logt_new, logt_old = 0, 0
                for t_new, t_old in zip(t_news, t_olds):
                    zbi = t_new.inducing_locations
                    zai = t_old.inducing_locations

                    qbi = self.model.posterior(zbi, q, diag=False)
                    qbi = MultivariateGaussianDistributionWithZ(
                        std_params={
                            "loc": qbi.loc,
                            "covariance_matrix": qbi.covariance_matrix,
                        },
                        inducing_locations=zbi,
                    )

                    qai = self.model.posterior(zai, q, diag=False)
                    qai = MultivariateGaussianDistributionWithZ(
                        std_params={
                            "loc": qai.loc,
                            "covariance_matrix": qai.covariance_matrix,
                        },
                        inducing_locations=zai,
                    )

                    logt_new += t_new.eqlogt(qbi)
                    logt_old += t_old.eqlogt(qai)

                loss = kl - logt_new + logt_old
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] = -loss.item()
                epoch["kl"] = kl.item()
                epoch["logt_new"] = logt_new.item()
                epoch["logt_old"] = logt_old.item()

                epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                       logt_new=epoch["logt_new"],
                                       logt_old=epoch["logt_old"])

                # Log progress for current epoch
                training_metrics["elbo"].append(epoch["elbo"])
                training_metrics["kl"].append(epoch["kl"])
                training_metrics["logt_new"].append(epoch["logt_new"])
                training_metrics["logt_old"].append(epoch["logt_old"])

                # Update learning rate.
                lr_scheduler.step()

                # Check whether to stop early.
                if self.config["early_stopping"](training_metrics["elbo"]):
                    break

            # Log the training curves for this update
            self.log["training_curves"].append(training_metrics)

            # Create non-trainable copy to send back to server.
            q_new = q.non_trainable_copy()

            return q_new

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class SequentialSGPServer(SGPServer):

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients)):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_old = client.t

                # Send over q marginalised at t_i_old.inducing_locations.
                za = t_old.inducing_locations
                qa_old = self.model.posterior(za, self.q, diag=False)
                qa_old = MultivariateGaussianDistributionWithZ(
                    std_params={
                        "loc": qa_old.loc,
                        "covariance_matrix": qa_old.covariance_matrix,
                    },
                    inducing_locations=za,
                    is_trainable=False,
                    train_inducing=self.q.train_inducing,   # Should be True.
                )

                _, t_new = client.fit(qa_old)

                logger.debug(
                    "Received client update. Updating global posterior.")

                # Project onto global inducing locations.
                self.q = self.update_posterior([t_old], [t_new])
                # self.q = self.compute_posterior_union()

                clients_updated += 1
                self.communications += 1

                # Log q after each update.
                # self.log["q"].append(self.q.non_trainable_copy())
                self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Update hyperparameters.
        if self.config["train_model"] and \
                self.iterations % self.config["model_update_freq"] == 0:
            self.update_hyperparameters()

        self.log["clients_updated"].append(clients_updated)


class SynchronousSGPServer(SGPServer):

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        t_olds, t_news = [], []
        for i, client in tqdm(enumerate(self.clients)):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_old = client.t

                # Send over q marginalised at t_i_old.inducing_locations.
                za = t_old.inducing_locations
                qa_old = self.model.posterior(za, self.q, diag=False)
                qa_old = MultivariateGaussianDistributionWithZ(
                    std_params={
                        "loc": qa_old.loc,
                        "covariance_matrix": qa_old.covariance_matrix,
                    },
                    inducing_locations=za,
                    is_trainable=False,
                    train_inducing=self.q.train_inducing,   # Should be True.
                )

                _, t_new = client.fit(qa_old)

                t_olds.append(t_old)
                t_news.append(t_new)

                clients_updated += 1
                self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Project onto global inducing locations.
        self.q = self.update_posterior(t_olds, t_news)

        # Log q after each update.
        # self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete.")

        self.iterations += 1

        # Update hyperparameters.
        if self.config["train_model"] and \
                self.iterations % self.config["model_update_freq"] == 0:
            self.update_hyperparameters()

        self.log["clients_updated"].append(clients_updated)
