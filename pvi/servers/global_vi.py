import logging
import torch

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

# =============================================================================
# Global VI class
# =============================================================================


class GlobalVI:
    def __init__(self, data, model, q, hyperparameters=None):

        if hyperparameters is None:
            hyperparameters = {}

        self.hyperparameters = self.get_default_hyperparameters()
        self.set_hyperparameters(hyperparameters)

        # Global VI server has access to entire dataset.
        self.data = data

        # Shared probabilistic model.
        self.model = model

        # Global posterior q(θ).
        self.q = q

        # Internal iteration counter.
        self.iterations = 0

        self.log = defaultdict(list)

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    @staticmethod
    def get_default_hyperparameters():
        return {
            "max_iterations": 1,
        }

    def tick(self):
        """
        Trains the model using the entire dataset.
        """
        if self.should_stop():
            return False

        model_hypers = self.model.hyperparameters

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=model_hypers["batch_size"],
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

        # Gradient-based optimisation loop -- loop over epochs
        for i in range(model_hypers["epochs"]):
            epoch = {
                "elbo": 0,
                "kl": 0,
                "ll": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(o).sum() / len(x)



    def should_stop(self):
        return self.iterations > self.hyperparameters["max_iterations"] - 1

    def get_compiled_log(self):
        """
        Get full log, including logs from each client.
        :return: full log.
        """
        final_log = {
            "server": self.log
        }

        return final_log
