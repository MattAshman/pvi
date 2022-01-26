import torch

from tqdm.auto import tqdm
from pvi.servers import Server


class StreamingVBServer(Server):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 25,
        }

    def _tick(self):
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                # In streaming VB, we set t(Θ) to 1 so data is recounted.
                # Equivalent to sequential PVI with no deletion.
                client.t.nat_params = {
                    k: torch.zeros_like(v) for k, v in client.t.nat_params.items()
                }
                t_old = client.t

                if self.communications == 0:
                    # First iteration. Pass q_init(θ) to client.
                    _, t_new = client.fit(self.q, self.init_q)
                else:
                    _, t_new = client.fit(self.q)

                self.q = self.q.replace_factor(t_old, t_new, is_trainable=False)
                self.communications += 1

                # Evaluate performance after every posterior update.
                if self.iterations == 0:
                    self.evaluate_performance()
                    self.log["communications"].append(self.communications)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class StreamingVBServerVCL(Server):
    def get_default_config(self):
        return {
            **super().get_default_config(),
        }

    def _tick(self):
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                if self.iterations == 0:
                    # First iteration. Pass q_init(θ) to client.
                    q_new, _ = client.fit(self.q, self.init_q)
                else:
                    q_new, _ = client.fit(self.q)

                self.q = q_new.non_trainable_copy()
                self.communications += 1

                # Evaluate performance after every posterior update.
                if self.iterations == 0:
                    self.evaluate_performance()
                    self.log["communications"].append(self.communications)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
