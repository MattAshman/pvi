import logging
import numpy as np

from tqdm.auto import tqdm
from .base import *

logger = logging.getLogger(__name__)


class AsynchronousServer(Server):
    """
    Similar to Mrinank's and Michael's implementation.

    In every round, this server samples M (total number of clients) clients,
    inversely proportional to the amount of data on each client, and updates
    them one after another (i.e. incorporating the previous clients updates).
    """
    def __init__(self, model, q, clients, config=None):
        super().__init__(model, q, clients, config)

        client_probs = [1 / client.data["x"].shape[0] for client in clients]
        self.client_probs = [prob / sum(client_probs) for prob in client_probs]

        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 25,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i in tqdm(range(len(self.clients)), leave=False):

            available_clients = [client.can_update() for client in
                                 self.clients]

            if not np.any(available_clients):
                logger.info('All clients report to be finished. Stopping.')
                break

            client_index = int(
                np.random.choice(len(self.clients), 1, replace=False,
                                 p=self.client_probs))
            logger.debug(f"Selected Client {client_index}")
            client = self.clients[client_index]

            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t
                t_i_new = client.fit(self.q)

                # Compute change in natural parameters.
                delta_np = {}
                for k in self.q.nat_params.keys():
                    delta_np[k] = t_i_new.nat_params[k] - t_i_old.nat_params[k]

                logger.debug(
                    "Received client updates. Updating global posterior.")

                # Update global posterior.
                q_new_nps = {k: v + delta_np[k]
                             for k, v in self.q.nat_params.items()}

                self.q = type(self.q)(nat_params=q_new_nps,
                                      is_trainable=False)

                clients_updated += 1
                self.communications += 1

                # Log q after each update.
                self.log["q"].append(self.q.non_trainable_copy())
                self.log["communications"].append(self.communications)

            else:
                logger.debug(f"Skipping client {client_index}, client not "
                             "avalible to update.")
                continue

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Log progress.
        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class DPPVIAsynchronousServer(AsynchronousServer):
    """
    Similar to Mrinank's and Michael's implementation.

    In every round, this server samples M (total number of clients) clients,
    inversely proportional to the amount of data on each client, and updates
    them one after another (i.e. incorporating the previous clients updates).
    """
    def __init__(self, model, q, clients, config=None):
        super().__init__(model, q, clients, config)

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0
        delta_nps = []

        for i in tqdm(range(len(self.clients)), leave=False):

            available_clients = [client.can_update() for client in
                                 self.clients]

            if not np.any(available_clients):
                logger.info('All clients report to be finished. Stopping.')
                break

            client_index = int(
                np.random.choice(len(self.clients), 1, replace=False,
                                 p=self.client_probs))
            logger.debug(f"Selected Client {client_index}")
            client = self.clients[client_index]

            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t
                t_i_new = client.fit(self.q)

                # Compute change in natural parameters.
                delta_np = {}
                for k in self.q.nat_params.keys():
                    delta_np[k] = t_i_new.nat_params[k] - t_i_old.nat_params[k]

                delta_nps.append(delta_np)

                clients_updated += 1
                self.communications += 1

            else:
                logger.debug(f"Skipping client {client_index}, client not "
                             "avalible to update.")
                continue

        logger.debug(
            "Received client updates. Updating global posterior.")

        # Update global posterior.
        q_new_nps = {}
        for k, v in self.q.nat_params.items():
            q_new_nps[k] = (v + sum([delta_np[k] for delta_np in delta_nps]))

        self.q = type(self.q)(nat_params=q_new_nps, is_trainable=False)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Log progress.
        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)
        self.log["clients_updated"].append(clients_updated)


class BayesianAsynchronousServer(BayesianServer):
    """
    Similar to Mrinank's and Michael's implementation.

    In every round, this server samples M (total number of clients) clients,
    inversely proportional to the amount of data on each client, and updates
    them one after another (i.e. incorporating the previous clients updates).
    """
    def __init__(self, model, q, qeps, clients, config=None):
        super().__init__(model, q, qeps, clients, config)

        client_probs = [1 / client.data["x"].shape[0] for client in clients]
        self.client_probs = [prob / sum(client_probs) for prob in client_probs]

        self.log["q"].append(self.q.non_trainable_copy())
        self.log["qeps"].append(self.qeps.non_trainable_copy())
        self.log["communications"].append(self.communications)

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 5,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i in tqdm(range(len(self.clients)), leave=False):

            available_clients = [client.can_update() for client in
                                 self.clients]

            if not np.any(available_clients):
                logger.info('All clients report to be finished. Stopping.')
                break

            client_index = int(
                np.random.choice(len(self.clients), 1, replace=False,
                                 p=self.client_probs))
            logger.debug(f"Selected Client {client_index}")
            client = self.clients[client_index]

            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t
                teps_i_old = client.teps
                t_i_new, teps_i_new = client.fit(self.q, self.qeps)

                # TODO: check parameter update results in valid distribution.
                # Compute change in natural parameters.
                q_delta_np = {k: t_i_new.nat_params[k] - t_i_old.nat_params[k]
                              for k in self.q.nat_params.keys()}
                qeps_delta_np = {
                    k1: {k2: (teps_i_new.nat_params[k1][k2]
                              - teps_i_old.nat_params[k1][k2])
                         for k2 in self.qeps.nat_params[k1].keys()}
                    for k1 in self.qeps.nat_params.keys()}

                logger.debug(
                    "Received client updates. Updating global posterior.")

                # Update global posterior.
                q_new_nps = {k: v + q_delta_np[k]
                             for k, v in self.q.nat_params.items()}
                qeps_new_nps = {
                    k1: {k2: v2 + qeps_delta_np[k1][k2]
                         for k2, v2 in self.qeps.nat_params[k1].items()}
                    for k1 in self.qeps.nat_params.keys()}
                qeps_new_distributions = {
                    k: self.qeps.distributions[k].create_new(
                        nat_params=v, is_trainable=False)
                    for k, v in qeps_new_nps.items()}

                self.q = self.q.create_new(nat_params=q_new_nps,
                                           is_trainable=False)
                self.qeps = type(self.qeps)(
                    distributions=qeps_new_distributions)

                clients_updated += 1
                self.communications += 1

                # Log q after each update.
                self.log["q"].append(self.q.non_trainable_copy())
                self.log["qeps"].append(self.qeps.non_trainable_copy())
                self.log["communications"].append(self.communications)

            else:
                logger.debug(f"Skipping client {client_index}, client not "
                             "avalible to update.")
                continue

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Log progress.
        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
