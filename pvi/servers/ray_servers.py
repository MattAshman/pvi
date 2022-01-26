import ray

from pvi.servers.base import Server


class SequentialRayFactory(Server):
    """
    This acts as both the server and clients.
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "ray_options": {},
        }

    def tick(self):

        # Make ray_options a list to allow for inhomogeneous compute.
        ray_options = self.config["ray_options"]
        if not isinstance(ray_options, (list, tuple)):
            ray_options = [ray_options for _ in range(len(self.clients) + 1)]
        else:
            assert len(ray_options) == (len(self.clients) + 1)

        # Stores ray object refs returned by performance metrics.
        performance_metrics = []
        client_idx = 0
        while not self.should_stop():
            client_idx = client_idx % len(self.clients)
            if self.communications == 0:
                working_client = update_client.options(
                    **ray_options[client_idx]
                ).remote(
                    self.clients[client_idx],
                    self.q,
                    self.init_q,
                    client_idx,
                )
            else:
                working_client = update_client.options(
                    **ray_options[client_idx]
                ).remote(
                    self.clients[client_idx],
                    self.q,
                    client_idx=client_idx,
                )

            # Wait for current client to finish and apply change in factors.
            tmp = ray.get(working_client)
            self.clients[client_idx] = tmp["client"]
            self.q = self.q.replace_factor(
                tmp["t_old"], tmp["t_new"], is_trainable=False
            )
            self.communications += 1

            if (self.communications % len(self.clients) == 0) or (
                self.communications < len(self.clients)
            ):
                # Evaluate current posterior.
                performance_metrics.append(
                    evaluate_performance.options(**ray_options[-1]).remote(self)
                )

            if (self.communications > 1) and (
                self.communications % len(self.clients) == 0
            ):
                # Increment iteration counter.
                self.iterations += 1

            client_idx += 1

        performance_metrics = ray.get(performance_metrics)

        self.log["performance_metrics"] = performance_metrics

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class AsynchronousRayFactory(Server):
    """
    This acts as both the server and clients to enable scalable distributed
    learning
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "ray_options": {},
        }

    def tick(self):

        # Make ray_options a list to allow for inhomogeneous compute.
        ray_options = self.config["ray_options"]
        if not isinstance(ray_options, (list, tuple)):
            ray_options = [ray_options for _ in range(len(self.clients) + 1)]
        else:
            assert len(ray_options) == (len(self.clients) + 1)

        working_clients = []
        for client_idx, client in enumerate(self.clients):
            working_clients.append(
                update_client.options(**ray_options[client_idx]).remote(
                    client,
                    self.q,
                    self.init_q,
                    client_idx,
                )
            )

        # Stores ray object refs returned by performance_metrics.
        performance_metrics = []
        updated_clients = []
        while not self.should_stop():
            working_client, _ = ray.wait(list(working_clients), num_returns=1)
            client_id = working_client[0]

            # Apply change in factors.
            tmp = ray.get(client_id)
            self.clients[client_idx] = tmp["client"]
            self.q = self.q.replace_factor(
                tmp["t_old"], tmp["t_new"], is_trainable=False
            )

            updated_clients.append(client_idx)
            self.communications += 1

            # Get client training again.
            working_clients[client_idx] = update_client.options(
                **ray_options[client_idx]
            ).remote(
                self.clients[client_idx],
                self.q,
                client_idx=client_idx,
            )

            if self.communications % len(self.clients) == 0:
                # Evaluate current posterior.
                performance_metrics.append(
                    evaluate_performance.options(**ray_options[-1]).remote(self)
                )

        performance_metrics = ray.get(performance_metrics)

        self.log["performance_metrics"] = performance_metrics

    def should_stop(self):
        return self.communications > self.config["max_communications"] - 1


class SynchronousRayFactory(Server):
    """
    This acts as both the server and clients to enable scalable distributed
    learning
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "ray_options": {},
        }

    def tick(self):

        # Make ray_options a list to allow for inhomogeneous compute.
        ray_options = self.config["ray_options"]
        if not isinstance(ray_options, (list, tuple)):
            ray_options = [ray_options for _ in range(len(self.clients) + 1)]
        else:
            assert len(ray_options) == (len(self.clients) + 1)

        # Put clients in the object store.
        client_ids = [ray.put(client) for client in self.clients]

        # Stores ray object refs returned by performance_metrics.
        performance_metrics = []
        while not self.should_stop():
            # Pass current q to clients.
            working_clients = []
            for client_idx, client in enumerate(client_ids):
                if self.iterations == 0:
                    working_clients.append(
                        update_client.options(**ray_options[client_idx]).remote(
                            client,
                            self.q,
                            self.init_q,
                            client_idx,
                        )
                    )
                else:
                    working_clients.append(
                        update_client.options(**ray_options[client_idx]).remote(
                            client,
                            self.q,
                            client_idx=client_idx,
                        )
                    )

            # Apply change in factors and update stored clients.
            while len(working_clients) > 0:
                working_client, _ = ray.wait(list(working_clients), num_returns=1)
                client_id = working_client[0]
                working_clients.remove(client_id)

                tmp = ray.get(client_id)
                self.clients[client_idx] = tmp["client"]
                client_ids[client_idx] = tmp["client"]

                self.q = self.q.replace_factor(
                    tmp["t_old"], tmp["t_new"], is_trainable=False
                )

                del client_id

            self.communications += len(self.clients)
            self.iterations += 1

            # Evaluate current posterior.
            performance_metrics.append(
                evaluate_performance.options(**ray_options[-1]).remote(self)
            )

        performance_metrics = ray.get(performance_metrics)

        self.log["performance_metrics"] = performance_metrics

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class BCMSameRayFactory(SynchronousRayFactory):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 1,
        }

    def tick(self):

        # Make ray_options a list to allow for inhomogeneous compute.
        ray_options = self.config["ray_options"]
        if not isinstance(ray_options, (list, tuple)):
            ray_options = [ray_options for _ in range(len(self.clients) + 1)]
        else:
            assert len(ray_options) == (len(self.clients) + 1)

        # Stores ray object refs returned by performance_metrics.
        performance_metrics = []
        while not self.should_stop():
            # Pass current q to clients.
            working_clients = []
            for client_idx, client in enumerate(self.clients):
                if self.iterations == 0:
                    working_clients.append(
                        update_client.options(**ray_options[client_idx]).remote(
                            client,
                            self.q,
                            self.init_q,
                            client_idx,
                        )
                    )
                else:
                    working_clients.append(
                        update_client.options(**ray_options[client_idx]).remote(
                            client,
                            self.q,
                            client_idx=client_idx,
                        )
                    )

            # Apply change in factors and update stored clients.
            nps = []
            while len(working_clients) > 0:
                working_client, _ = ray.wait(list(working_clients), num_returns=1)
                client_id = working_client[0]
                working_clients.remove(client_id)

                tmp = ray.get(client_id)
                self.clients[client_idx] = tmp["client"]
                nps.append(
                    {k: v.detach().clone() for k, v in tmp["q_new"].nat_params.items()}
                )

            q_nps = {
                k: sum([x[k] for x in nps]) - (len(self.clients) - 1) * v
                for k, v in self.q.nat_params.items()
            }
            self.q = self.q.create_new(nat_params=q_nps, is_trainable=False)

            self.communications += len(self.clients)
            self.iterations += 1

            # Evaluate current posterior.
            performance_metrics.append(
                evaluate_performance.options(**ray_options[-1]).remote(self)
            )

        self.log["performance_metrics"] = ray.get(performance_metrics)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class BCMSplitRayFactory(SynchronousRayFactory):
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

        # Make ray_options a list to allow for inhomogeneous compute.
        ray_options = self.config["ray_options"]
        if not isinstance(ray_options, (list, tuple)):
            ray_options = [ray_options for _ in range(len(self.clients) + 1)]
        else:
            assert len(ray_options) == (len(self.clients) + 1)

        # Stores ray object refs returned by performance_metrics.
        performance_metrics = []
        while not self.should_stop():
            # Pass current q to clients.
            working_clients = []
            for client_idx, client in enumerate(self.clients):
                if self.iterations == 0:
                    working_clients.append(
                        update_client.options(**ray_options[client_idx]).remote(
                            client,
                            self.q,
                            self.init_q,
                            client_idx,
                        )
                    )
                else:
                    working_clients.append(
                        update_client.options(**ray_options[client_idx]).remote(
                            client,
                            self.q,
                            client_idx=client_idx,
                        )
                    )

            # Apply change in factors and update stored clients.
            nps = []
            while len(working_clients) > 0:
                working_client, _ = ray.wait(list(working_clients), num_returns=1)
                client_id = working_client[0]
                working_clients.remove(client_id)

                tmp = ray.get(client_id)
                self.clients[client_idx] = tmp["client"]
                nps.append(
                    {k: v.detach().clone() for k, v in tmp["q_new"].nat_params.items()}
                )

            q_nps = {k: sum([x[k] for x in nps]) for k in self.q.nat_params.keys()}
            self.q = self.q.create_new(nat_params=q_nps, is_trainable=False)

            self.communications += len(self.clients)
            self.iterations += 1

            # Evaluate current posterior.
            performance_metrics.append(
                evaluate_performance.options(**ray_options[-1]).remote(self)
            )

        self.log["performance_metrics"] = ray.get(performance_metrics)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


@ray.remote(max_calls=1)
def update_client(client, q, init_q=None, client_idx=None):
    if client_idx is not None:
        print(f"Updating client {client_idx}.")

    t_old = client.t
    q_new, t_new = client.fit(q, init_q)

    # Add IP address of whichever node it was ran on.
    ip_address = ray._private.services.get_node_ip_address()
    client.log["update_time"][-1]["ip_address"] = ip_address

    return {
        "client": client,
        "client_idx": client_idx,
        "q_new": q_new,
        "t_old": t_old,
        "t_new": t_new,
    }


@ray.remote(max_calls=1)
def update_q(q, *ts):
    for t in ts:
        _, _, t_old, t_new = t

        # Update posterior.
        q = q.replace_factor(t_old, t_new, is_trainable=False)

    return q


@ray.remote
def evaluate_performance(server):

    server.evaluate_performance()

    metrics = server.log["performance_metrics"][-1]
    print("Communications: {}.".format(server.communications))
    print(
        "Test mll: {:.3f}. Test acc: {:.3f}.".format(
            metrics["val_mll"], metrics["val_acc"]
        )
    )
    print(
        "Train mll: {:.3f}. Train acc: {:.3f}.\n".format(
            metrics["train_mll"], metrics["train_acc"]
        )
    )

    # Add IP address of whichever node it was ran on.
    metrics["ip_address"] = ray._private.services.get_node_ip_address()

    return metrics
