import logging
import torch

from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from .base import Server
from pvi.utils.dataset import ListDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# Global VI class
# =============================================================================


class GlobalVIServer(Server):
    def __init__(self, model, q, clients, config=None,
                 homogenous_split=False):
        super().__init__(model, q, clients, config)

        # Global VI server has access to the entire dataset.
        self.data = {k: torch.cat([client.data[k] for client in self.clients],
                                  dim=0)
                     for k in self.clients[0].data.keys()}

        # Dictates whether to use a homogenous data split or not.
        self.homogenous_split = homogenous_split

        # Tracks number of epochs.
        self.epochs = 0

        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 1,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "num_elbo_samples": 10,
            "print_epochs": 1
        }

    def tick(self):
        """
        Trains the model using the entire dataset.
        """
        if self.should_stop():
            return False

        model_config = self.model.config

        # Set up data: global VI server can access the entire dataset.

        # TODO: current only supports homogenous minibatches.
        if self.homogenous_split:
            x = self.data["x"]
            y = self.data["y"]

            # Shuffle data.
            tensor_dataset = TensorDataset(x, y)
            loader = DataLoader(tensor_dataset,
                                batch_size=model_config["batch_size"],
                                shuffle=True)
        else:
            # Inhomogenous split: order matters.
            m = model_config["batch_size"]
            data = defaultdict(list)
            for client in self.clients:
                # Chunk clients data into batch size.
                data = {k: data[k] + [v[i:i+m] for i in range(
                    0, len(client.data["x"]), m)]
                        for k, v in client.data.items()}

            # Lists of tensors size (batch_size, *).
            x = data["x"]
            y = data["y"]
            list_dataset = ListDataset(x, y)
            loader = DataLoader(list_dataset,
                                batch_size=None,
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
        optimiser = getattr(torch.optim, model_config["optimiser"])(
            q.parameters(), **model_config["optimiser_params"])

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(model_config["epochs"]), desc="Epochs")
        # for i in range(model_config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "kl": 0,
                "ll": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                self.communications += 1
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(p).sum() / len(self.data["x"])

                # Compute E_q[log p(y | x, θ)].
                if str(type(q)) == str(self.model.conjugate_family):
                    ll = self.model.expected_log_likelihood(batch, q).sum()
                else:
                    thetas = q.rsample(
                        (model_config["num_elbo_samples"],))
                    ll = self.model.likelihood_log_prob(
                        batch, thetas).mean(0).sum()

                ll /= len(x_batch)

                # Negative local Free Energy is KL minus log-probability
                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"])

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if i % model_config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"Epochs: {i}.")

            # Log q so we can track performance each epoch.
            self.log["q"].append(q.non_trainable_copy())
            self.log["communications"].append(self.communications)
            self.epochs += 1

        # Update global posterior.
        self.q = q.non_trainable_copy()

        self.iterations += 1

        # Log training.
        self.log["training_curves"].append(training_curve)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
