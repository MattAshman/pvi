import logging
import torch

from .base import Client
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class IPBNNClient(Client):

    def gradient_based_update(self, p, init_q=None):
        # TODO: Can this somehow be merged with the base Client class?
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make non-trainable.
        q_cav = p.non_trainable_copy()
        q_cav, t_idx = q_cav.form_cavity(self.t)

        # Special form for q.
        t_new = self.t.trainable_copy()
        q = p.non_trainable_copy()
        # Replace old factor with new (optimisable) factor.
        q.ts[t_idx] = t_new

        # Parameters are those of t_new(θ) and self.model.
        if self.config["train_model"]:
            if "model_optimiser_params" in self.config:
                parameters = [
                    {"params": t_new.parameters()},
                    {"params": self.model.parameters(),
                     **self.config["model_optimiser_params"]}
                ]
            else:
                parameters = [
                    {"params": t_new.parameters()},
                    {"params": self.model.parameters()}
                ]
        else:
            parameters = t_new.parameters()

        # Reset optimiser.
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])
        lr_scheduler = getattr(torch.optim.lr_scheduler,
                               self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"])

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

        # Dict for logging optimisation progress
        training_curve = defaultdict(list)

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
                          leave=True)
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "kl": 0,
                "ll": 0,
                "logt": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                ll, kl = self.model.elbo(
                    batch, q_cav, q,
                    num_samples=self.config["num_elbo_samples"])
                kl /= len(x)
                ll /= len(x_batch)

                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if self.t is not None:
                training_curve["logt"].append(epoch["logt"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"log t: {epoch['logt']:.3f}, "
                             f"Epochs: {i}.")

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"], logt=epoch["logt"],
                                   lr=optimiser.param_groups[0]["lr"])

            # Update learning rate.
            lr_scheduler.step()

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Create non-trainable copy to send back to server.
        q_new = q.non_trainable_copy()
        t_new = t_new.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        return q_new, t_new
