import logging
import torch

from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from .base import Server
from pvi.utils.dataset import ListDataset

logger = logging.getLogger(__name__)

# =============================================================================
# Global VI class
# =============================================================================


class GlobalVI(Server):
    def __init__(self, model, q, clients, hyperparameters=None,
                 homogenous_split=True):
        super().__init__(model, q, clients, hyperparameters)

        # Global VI server has access to the entire dataset.
        self.data = {k: torch.cat([client.data[k] for client in self.clients],
                                  dim=0)
                     for k in self.clients[0].data.keys()}

        # Dictates whether to use a homogenous data split or not.
        self.homogenous_split = homogenous_split

    def get_default_hyperparameters(self):
        return {
            **super().get_default_hyperparameters(),
            "max_iterations": 1,
        }

    def tick(self):
        """
        Trains the model using the entire dataset.
        """
        if self.should_stop():
            return False

        model_hypers = self.model.hyperparameters

        # Set up data: global VI server can access the entire dataset.

        # TODO: current only supports homogenous minibatches.
        if self.homogenous_split:
            x = self.data["x"]
            y = self.data["y"]

            # Shuffle data.
            tensor_dataset = TensorDataset(x, y)
            loader = DataLoader(tensor_dataset,
                                batch_size=model_hypers["batch_size"],
                                shuffle=True)
        else:
            # Inhomogenous split: order matters.
            m = model_hypers["batch_size"]
            data = defaultdict(list)
            for client in self.clients:
                # Chunk clients data into batch size.
                data = {k: data[k] + [v[i:i+m] for i in range(
                    0, len(client.data["x"]), m)] for k, v in client.data}

            # Lists of tensors size (batch_size, *).
            x = data["x"]
            y = data["y"]
            list_dataset = ListDataset(x, y)
            loader = DataLoader(list_dataset,
                                batch_size=1,
                                shuffle=True)

        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "kl": [],
            "ll": [],
        }

        p = self.q.non_trainable_copy()
        q = self.q.trainable_copy()

        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, model_hypers["optimiser"])(
            q.parameters(), **model_hypers["optimiser_params"])

        communications = 0

        # Gradient-based optimisation loop -- loop over epochs
        for i in range(model_hypers["epochs"]):
            epoch = {
                "elbo": 0,
                "kl": 0,
                "ll": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                communications += 1
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(p).sum() / len(x)

                # Compute E_q[log p(y | x, θ)].
                if str(type(q)) == str(self.model.conjugate_family):
                    ll = self.model.expected_log_likelihood(batch, q).sum()
                else:
                    thetas = q.rsample(
                        (model_hypers["num_elbo_theta_samples"],))
                    ll = self.model.likelihood_log_prob(
                        batch, thetas).mean(0).sum()

                ll /= len(x_batch)

                # Negative local Free Energy is KL minus log-probability
                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item()
                epoch["kl"] += kl.item()
                epoch["ll"] += ll.item()

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if i % model_hypers["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"Epochs: {i}.")

        self.iterations += 1

        # Log training.
        self.log["training_curves"].append(training_curve)
        self.log["q"].append(q)
        self.log["communications"].append(communications)

    def should_stop(self):
        return self.iterations > self.hyperparameters["max_iterations"] - 1
