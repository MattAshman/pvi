from .base import Server


class SequentialServer(Server):
    def __init__(self, model, q, clients, max_iterations=100):
        super().__init__(model, q, clients, max_iterations)

    def tick(self):
        if self.should_stop():
            return False

        delta_nps = []
        for i, client in enumerate(self.clients):
            t_i_old = client.t
            t_i_new = client.fit(self.q)

            # Compute change in natural parameters.
            delta_np = {}
            for k in self.q.nat_params.keys():
                delta_np[k] = t_i_new.nat_params[k] - t_i_old.nat_params[k]

            delta_nps.append(delta_np)

            # Update global posterior.
            q_new_nps = {}
            for k, v in self.q.nat_params.items():
                q_new_nps[k] = v + delta_np[k]

            self.q = type(self.q)(nat_params=q_new_nps, is_trainable=False)

        # Log progress.
        self.log["delta_nps"].append(delta_nps)
        self.log["nps"].append(self.q.nat_params)

    def should_stop(self):
        if self.iterations > self.max_iterations - 1:
            return True
