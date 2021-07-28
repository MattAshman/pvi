import logging
import time

from tqdm.auto import tqdm
from .base import Server

logger = logging.getLogger(__name__)


class BayesianCommitteeMachineSame(Server):
    """
    Implementation of the Bayesian committee machine (same), in which the
    global posterior approximation is given by

    q(θ) = ∏_k q_k(θ) / p(θ)^{K-1}

    where each q_k(θ) is the approximate posterior for each data shard

    q_k(θ) ≅ p(θ | D_k) = p(θ) p(D_k | θ) / p(D_k).
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 1,
        }

    def tick(self):
        if self.should_stop():
            return False

        if self.t0 is None:
            self.t0 = time.time()
            self.pc0 = time.perf_counter()
            self.pt0 = time.process_time()

        logger.debug("Getting client updates.")

        nps = []
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")

                if self.iterations == 0:
                    q_i, _ = client.fit(self.q, self.init_q)
                else:
                    q_i, _ = client.fit(self.q)

                # Store natural parameters.
                np = {k: v.detach().clone() for k, v in q_i.nat_params.items()}
                nps.append(np)
                self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        q_nps = {k: sum([np[k] for np in nps]) - (len(self.clients) - 1) * v
                 for k, v in self.q.nat_params.items()}

        self.q = self.q.create_new(nat_params=q_nps, is_trainable=False)

        logger.debug(f"Iteration {self.iterations} complete.")
        self.iterations += 1

        # Log progress.
        self.evaluate_performance()
        self.log["communications"].append(self.communications)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class BayesianCommitteeMachineSplit(Server):
    """
    Implementation of the Bayesian committee machine (split), in which the
    global posterior approximation is given by

    q(θ) = ∏_k q_k(θ)

    where each q_k(θ) is the approximate posterior for each data shard

    q_k(θ) ≅ p(θ | D_k) = p(θ)^{N_k / N} p(D_k | θ) / p(D_k).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nk = [client.data["x"].shape[0] for client in self.clients]
        client_props = [n / sum(nk) for n in nk]
        self.client_props = client_props

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 1,
        }

    def tick(self):
        if self.should_stop():
            return False

        if self.t0 is None:
            self.t0 = time.time()
            self.pc0 = time.perf_counter()
            self.pt0 = time.process_time()

        logger.debug("Getting client updates.")
        nps = []
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")

                # Client prior is weighted by (N_k / N).
                p_i_nps = {k: v * self.client_props[i]
                           for k, v in self.q.nat_params.items()}
                p_i = type(self.q)(nat_params=p_i_nps, is_trainable=False)

                if self.iterations == 0:
                    q_i, _ = client.fit(p_i, self.init_q)
                else:
                    q_i, _ = client.fit(p_i)

                # Store natural parameters.
                np = {k: v.detach().clone() for k, v in q_i.nat_params.items()}
                nps.append(np)
                self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        q_nps = {k: sum([np[k] for np in nps])
                 for k, v in self.q.nat_params.items()}

        self.q = self.q.create_new(nat_params=q_nps, is_trainable=False)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Log progress.
        self.evaluate_performance()
        self.log["communications"].append(self.communications)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
