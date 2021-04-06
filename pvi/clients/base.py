import logging
import torch

from abc import ABC, abstractmethod
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Base client class
# =============================================================================


class PVIClient(ABC):
    
    def __init__(self, data, model, t):
        
        # Set data partition and likelihood
        self.data = data
        self.model = model
        
        # Set likelihood approximating term
        self.t = t
        
        self.log = defaultdict(list)
        self._can_update = True

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        :return:
        """
        return self._can_update
    
    @abstractmethod
    def fit(self, q):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        pass

    def update_q(self, q):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """

        # Type(q) is self.model.conjugate_family.
        if str(type(q)) == str(self.model.conjugate_family):
            # No need to make q trainable.
            return self.model.conjugate_update(self.data, q, self.t)
            
        else:
            # Pass a trainable copy to optimise.
            return self.gradient_based_update(q.trainable_copy())

    def gradient_based_update(self, q):
        hyper = self.model.hyperparameters

        # Cannot update during optimisation.
        self._can_update = False
        
        # Copy the approximate posterior, make old posterior non-trainable
        q_old = q.non_trainable_copy()
           
        # Reset optimiser
        # TODO: not optimising model parameters for now (inducing points,
        #  kernel hyperparameters, observation noise etc.).
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, hyper["optimiser"])(
            q.parameters(), **hyper["optimiser_params"])
        
        # Set up data
        x = self.data["x"]
        y = self.data["y"]
        
        num_data = self.data["x"].shape[0]
        
        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=hyper["batch_size"],
                            shuffle=True)

        # Dict for logging optimisation progress
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
        }
        
        # Gradient-based optimisation loop -- loop over epochs
        # epoch_iter = tqdm(range(hyper["epochs"]), desc="Epoch", leave=False)
        for i in range(hyper["epochs"]):
            epoch = {
                "elbo" : 0,
                "kl"   : 0,
                "ll"   : 0,
            }
            
            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x" : x_batch,
                    "y" : y_batch,
                }
                
                # Compute KL divergence between q and q_old.
                kl = q.kl_divergence(q_old).sum() / len(x)

                # Sample θ from q and compute p(y | θ, x) for each θ
                thetas = q.rsample((hyper["num_elbo_samples"],))
                ll = self.model.likelihood_log_prob(
                    batch, thetas).mean(0).sum() / len(x_batch)
                ll = ll - self.t(thetas).mean(0).sum() / len(x)

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

            if i % hyper["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"Epochs: {i}.")

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Compute new local contribution from old distributions
        t_new = self.t.compute_refined_factor(q, q_old)

        # Finished optimisation, can now update.
        self._can_update = True
        
        return q, t_new


class ContinualLearningClient:

    def __init__(self, data, model):

        # Set data partition and likelihood
        self.data = data
        self.model = model

        self.log = defaultdict(list)
        self._can_update = True

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        :return:
        """
        return self._can_update

    def fit(self, q):
        """
        Computes the refined approximating posterior (q).
        """
        return self.update_q(q)

    def update_q(self, q):
        """
        Computes a refined approximate posterior.
        """

        # Type(q) is self.model.conjugate_family.
        if str(type(q)) == str(self.model.conjugate_family):
            # No need to make q trainable.
            q_new = self.model.conjugate_update(self.data, q, None)
            return q_new
        else:
            # Pass a trainable copy to optimise.
            return self.gradient_based_update(q.trainable_copy())

    def gradient_based_update(self, q):
        hyper = self.model.hyperparameters

        # Cannot update during optimisation.
        self._can_update = False

        # Old posterior = prior, make non-trainable.
        p = q.non_trainable_copy()

        # Reset optimiser.
        # TODO: not optimising model parameters for now (inducing points,
        #  kernel hyperparameters, observation noise etc.).
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, hyper["optimiser"])(
            q.parameters(), **hyper["optimiser_params"])

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=hyper["batch_size"],
                            shuffle=True)

        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "kl": [],
            "ll": [],
        }

        # Gradient-based optimisation loop -- loop over epochs
        # epoch_iter = tqdm(range(hyper["epochs"]), desc="Epoch", leave=False)
        for i in range(hyper["epochs"]):
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
                kl = q.kl_divergence(p).sum() / len(x)

                # Sample θ from q and compute p(y | θ, x) for each θ
                thetas = q.rsample((hyper["num_elbo_samples"],))
                ll = self.model.likelihood_log_prob(
                    batch, thetas).mean(0).sum() / len(x_batch)

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

            # epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
            #                        ll=epoch["ll"])

            if i % hyper["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"Epochs: {i}.")

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Finished optimisation, can now update.
        self._can_update = True

        return q
