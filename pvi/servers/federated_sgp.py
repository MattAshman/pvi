import logging
import torch
import numpy as np

from tqdm.auto import tqdm
from .base import *
from pvi.utils.gaussian import joint_from_marginal
from pvi.utils.psd_utils import psd_inverse

logger = logging.getLogger(__name__)


class SequentialSGPServer(Server):
    def __init__(self, model, q, clients, config=None,
                 maintain_inducing=None):
        super().__init__(model, q, clients, config)

        self.maintain_inducing = maintain_inducing
        self.client_inducing_idx = []

        if not self.maintain_inducing:
            # Check q has been initialised correctly (contains all inducing
            # locations of clients) and store indices in self.zi_idx.
            z = q.inducing_locations
            for i, client in enumerate(clients):
                zi = client.t.inducing_locations.detach()
                inducing_idx = np.empty(len(zi))
                for j, zij in enumerate(zi):
                    idx = torch.where((zij == z).all(dim=1))[0]
                    if idx is None:
                        raise ValueError(f"Client {i} contains inducing "
                                         f"locations not include in q.")
                    else:
                        inducing_idx[j] = idx

                self.client_inducing_idx.append(inducing_idx)

        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 25,
            "damping_factor": 1.,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i, client in tqdm(self.clients):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t
                _, t_i_new = client.fit(self.q)

                logger.debug(
                    "Received client update. Updating global posterior.")

                # Project onto global inducing locations.
                self.q = self.update_posterior(self.q, t_i_old, t_i_new)

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

    def update_posterior(self, q_old, client_idx, t_old, t_new):
        """
        Computes the projection
        q(f) = argmin KL[q(f) || qold(f) x t_new(bi) / t_old(ai)].
        :param q_old: Old posterior q(f) = p(f | a) q(a).
        :param client_idx: Index of client.
        :param t_old: Old approximate factor t_old(ai).
        :param t_new: New approximate factor t_new(bi).
        :return: New posterior q(f) = p(f | b) q(b).
        """

        if not self.maintain_inducing:
            # Server uses aggregate of clients inducing points.
            # Remove t_old from q_old.
            z_old = q_old.inducing_locations
            zi = t_old.inducing_locations
            zi_idx = torch.zeros(len(zi))
            for j, zij in enumerate(zi):
                zi_idx[j] = torch.where((zij == z_old).all(dim=1))[0]

            q_cav_old_np = q_old.nat_params
            # Remove t_old(ai) from q_old.
            for k in q_cav_old_np.keys():
                if len(q_cav_old_np[k].shape) == 1:
                    for j, idx in self.client_inducing_idx[client_idx]:
                        # Remove natural parameters.
                        q_cav_old_np[k][idx] -= t_old.nat_params[k][j]
                else:
                    for j1, idx1 in self.client_inducing_idx[client_idx]:
                        for j2, idx2 in self.client_inducing_idx[client_idx]:
                            # Remove natural parameters.
                            q_cav_old_np[k][idx1][idx2] -= \
                                t_old.nat_params[k][j1][j2]

            # Convert to std params, and remove rows and columns from q_old(u)
            # to form q_old(u_{\ ai})
            q_cav_old_std = q_old._std_from_nat(q_cav_old_np)
            for k in q_cav_old_std.keys():
                if len(q_cav_old_std[k].shape) == 1:
                    for idx in self.client_inducing_idx[client_idx]:
                        # Remove row.
                        q_cav_old_std[k] = q_cav_old_std[k][
                            np.arange(len(q_cav_old_std[k])) != idx]
                else:
                    for idx in self.client_inducing_idx[client_idx]:
                        # Remove row.
                        q_cav_old_std[k] = q_cav_old_std[k][
                            np.arange(len(q_cav_old_std[k])) != idx, :]
                        # Remove column.
                        q_cav_old_std[k] = q_cav_old_std[k][
                            :, np.arange(len(q_cav_old_std[k])) != idx]

            # Removing inducing locations from Z_old.
            z_cav_old = q_old.inducing_locations
            for idx in self.client_inducing_idx[client_idx]:
                z_cav_old = z_cav_old[np.arange(len(z_cav_old)) != idx]

            q_cav_old = type(self.q)(inducing_locations=z_cav_old,
                                     std_params=q_cav_old_std)

            # Compute joint posterior q_old(u_{\ ai}, bi)p(bi | u_{\ai}).
            zbi = t_new.inducing_locations
            kuu = self.model.kernel(z_cav_old, z_cav_old)
            kub = self.model.kernel(z_cav_old, zbi)

            z_new_tmp = torch.cat([q_old.inducing_locations, zbi])
            q_cav_new_loc_tmp, q_cav_new_cov_tmp = joint_from_marginal(
                q_cav_old, kub, kaa=kuu)

            # Ensure new rows and columns are in the correct index.
            permutation = list(range(len(z_cav_old)))
            for j, idx in enumerate(self.client_inducing_idx[client_idx]):
                permutation.insert(idx, len(z_cav_old) + j)

            # Ensure new rows and columns are in the correct index.
            z_new = torch.empty_like(z_new_tmp)
            q_cav_new_loc = torch.empty_like(q_cav_new_loc_tmp)
            q_cav_new_cov = torch.empty_like(q_cav_new_cov_tmp)
            for j, idx in enumerate(permutation):
                q_cav_new_loc[j] = q_cav_new_loc_tmp[idx]
                q_cav_new_cov[j, :] = q_cav_new_cov_tmp[idx, :]
                q_cav_new_cov[:, j] = q_cav_new_cov_tmp[:, idx]
                z_new[j] = z_new_tmp[idx]

            q_cav_new_std = {
                "loc": q_cav_new_loc,
                "covariance_matrix": q_cav_new_cov
            }

            # Convert to natural parameters, and add those of t_new(b).
            z_new = torch.cat
            q_new_np = q_old._nat_from_std(q_cav_new_std)
            for k in q_new_np.keys():
                if len(q_new_np[k].shape) == 1:
                    for j, idx in self.client_inducing_idx[client_idx]:
                        # Add natural parameters.
                        q_new_np[k][idx] -= t_new.nat_params[k][j]
                else:
                    for j1, idx1 in self.client_inducing_idx[client_idx]:
                        for j2, idx2 in self.client_inducing_idx[client_idx]:
                            # Add natural parameters.
                            q_new_np[k][idx1][idx2] -= \
                                t_new.nat_params[k][j1][j2]

            q_new = self.q.create_new(inducing_locations=z_new,
                                      nat_params=q_new_np, is_trainable=False)

            return q_new

        else:
            # TODO: use collapsed lower bound.
            # TODO: check this passes trainable inducing locations too.
            q = self.q.non_trainable_copy()
            za = q_old.inducing_locations
            zb = q.inducing_locations
            zbi = t_new.inducing_locations
            zai = t_new.inducing_locations

            kaa = self.model.kernel(za, za).evaluate().detach()
            ikaa = psd_inverse(kaa)

            optimiser = getattr(torch.optim, self.config["optimiser"])(
                q.parameters(), **self.config["optimiser_params"])

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
            for i in range(self.config["epochs"]):
                epoch = {
                    "elbo": 0,
                    "kl": 0,
                    "logt_new": 0,
                    "logt_old": 0,
                }

                z = torch.cat([za, zb])

                kab = self.model.kernel(za, zb).evaluate()
                kbb = self.model.kernel(zb, zb).evaluate()
                ikbb = psd_inverse(kbb)

                qab_loc, qab_cov = joint_from_marginal(q, kab.T, ikaa=ikbb)
                qab = type(q)(
                    inducing_locations=z,
                    std_params={
                        "loc": qab_loc,
                        "covariance_matrix": qab_cov,
                    }
                )
                qab_old_loc, qab_old_cov = joint_from_marginal(
                    q_old, kab, ikaa=ikaa)
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
                qbi = self.model.posterior(zbi, q, diag=False)
                logt_new = t_new.eqlogt(qbi)

                qai = self.model.posterior(zai, q, diag=False)
                logt_old = t_old.eqlogt(qai)

                loss = kl + logt_new - logt_old
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] = -loss.item()
                epoch["kl"] = kl.item()
                epoch["logt_new"] = logt_new.item()
                epoch["logt_old"] = logt_old.item()

                # Log progress for current epoch
                training_curve["elbo"].append(epoch["elbo"])
                training_curve["kl"].append(epoch["kl"])
                training_curve["logt_new"].append(epoch["logt_new"])
                training_curve["logt_old"].append(epoch["logt_old"])

                epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                       logt_new=epoch["logt_new"],
                                       logt_old=epoch["logt_old"])

            # Log the training curves for this update
            self.log["training_curves"].append(training_curve)

            # Create non-trainable copy to send back to server.
            q_new = q.non_trainable_copy()

            return q_new

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1