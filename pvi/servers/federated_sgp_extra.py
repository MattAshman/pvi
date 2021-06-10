import logging
import torch
import numpy as np
import pdb

from tqdm.auto import tqdm
from .base import *
from pvi.utils.gaussian import joint_from_marginal, nat_from_std
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.distributions import MultivariateGaussianDistributionWithZ

logger = logging.getLogger(__name__)

JITTER = 1e-4
MIN_EIGVAL = 1e-3

from .federated_sgp import SGPServer


class SequentialSGPServerNoProjection(SGPServer):
    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients)):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")

                logger.debug(f"Computing the cavity.")
                q_cavity = self.compute_cavity(i)
                _, t_new = client.fit(q_cavity)

                logger.debug(
                    "Received client update. Updating global posterior."
                )

                # Project onto global inducing locations.
                self.q = self.update_posterior([t_old], [t_new])
                # self.q = self.compute_posterior_union()

                clients_updated += 1
                self.communications += 1

                # Log q after each update.
                # self.log["q"].append(self.q.non_trainable_copy())
                self.log["communications"].append(self.communications)

        logger.debug(
            f"Iteration {self.iterations} complete."
            f"\nNew natural parameters:\n{self.q.nat_params}\n."
        )

        self.iterations += 1

        # Update hyperparameters.
        if (
            self.config["train_model"]
            and self.iterations % self.config["model_update_freq"] == 0
        ):
            self.update_hyperparameters()

        self.log["clients_updated"].append(clients_updated)

    def compute_cavity(self, client_to_ignore):
        """
        Computes q(u) = p(u) Π_m t_m(u_m) with m not equal client_to_ignore.
        :return: q(u).
        """
        clients_to_use = []
        for client_idx, client in enumerate(self.clients):
            if client_idx == client_to_ignore:
                continue
            else:
                clients_to_use.append(client)
        # Store inducing locations, and indices for each client.
        z_idx = []
        z = []
        i = 0
        for client_idx, client in enumerate(clients_to_use):
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

        # Compute q(u) = p(u) Π_m t_m(u_m), ignoring client_to_ignore
        np_q = nat_from_std(std_q)

        for client_idx, client in enumerate(clients_to_use):
            np_i = client.t.nat_params
            for i, idx1 in enumerate(z_idx[client_idx]):
                np_q["np1"][..., idx1] += np_i["np1"][..., i]
                for j, idx2 in enumerate(z_idx[client_idx]):
                    np_q["np2"][..., idx1, idx2] += np_i["np2"][..., i, j]

        q = MultivariateGaussianDistributionWithZ(
            nat_params=np_q,
            inducing_locations=z,
            is_trainable=False,
            train_inducing=True,
        )
        pdb.set_trace()

        return q
