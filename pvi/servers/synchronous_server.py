import logging

from tqdm.auto import tqdm
from .base import *

logger = logging.getLogger(__name__)


class SynchronousServer(Server):
    def __init__(self, model, q, clients, config=None):
        super().__init__(model, q, clients, config)

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

        delta_nps = []
        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients), leave=False):
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

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        q_new_nps = {}
        for k, v in self.q.nat_params.items():
            q_new_nps[k] = (v + sum([delta_np[k] for delta_np in delta_nps]))

        self.q = type(self.q)(nat_params=q_new_nps, is_trainable=False)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Update hyperparameters.
        if self.config["train_model"] and \
                self.iterations % self.config["model_update_freq"] == 0:
            self.update_hyperparameters()

        # Log progress.
        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)
        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class SynchronousServerBayesianHypers(ServerBayesianHypers):
    def __init__(self, model, q, qeps, clients, config=None):
        super().__init__(model, q, qeps, clients, config)

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

        q_delta_nps, qeps_delta_nps = [], []
        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t
                teps_i_old = client.teps
                t_i_new, teps_i_new = client.fit(self.q, self.qeps)
                # Compute change in natural parameters.
                q_delta_np = {k: (t_i_new.nat_params[k]
                                  - t_i_old.nat_params[k])
                              for k in self.q.nat_params.keys()}
                qeps_delta_np = {
                    k1: {k2: (teps_i_new.nat_params[k1][k2]
                              - teps_i_old.nat_params[k1][k2])
                         for k2 in self.qeps.nat_params[k1].keys()}
                    for k1 in self.qeps.nat_params.keys()}

                q_delta_nps.append(q_delta_np)
                qeps_delta_nps.append(qeps_delta_np)

                clients_updated += 1
                self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        q_new_nps = {k: v + sum([np[k] for np in q_delta_nps])
                     for k, v in self.q.nat_params.items()}
        qeps_new_nps = {
            k1: {k2: v2 + sum([np[k1][k2] for np in qeps_delta_nps])
                 for k2, v2 in self.qeps.nat_params[k1].items()}
            for k1 in self.qeps.nat_params.keys()}
        qeps_new_distributions = {
            k: self.qeps.distributions[k].create_new(
                nat_params=v, is_trainable=False)
            for k, v in qeps_new_nps.items()}

        self.q = self.q.create_new(nat_params=q_new_nps, is_trainable=False)
        self.qeps = type(self.qeps)(distributions=qeps_new_distributions)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Log progress.
        self.log["q"].append(self.q.non_trainable_copy())
        self.log["qeps"].append(self.qeps.non_trainable_copy())
        self.log["communications"].append(self.communications)
        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
