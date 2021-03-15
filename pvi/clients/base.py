from abc import ABC, abstractmethod

import logging
import torch

from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Base client class
# =============================================================================


class Client:
    
    def __init__(self, data, model, t):
        
        # Set data partition and likelihood
        self.data = data
        self.model = model
        
        # Set likelihood approximating term
        self.t = t
        
        self._training_curves = []
        
    
    @abstractmethod
    def fit(self):
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
        if type(q) == self.model.conjugate_family:
            return self.model.conjugate_update(self.data, q, self.t)
            
        else:
            return self.gradient_based_update(self.data, q)
        
        
    def gradient_based_update(self, data, q):
        
        hyper = self.model.hyperparameters
        
        # Copy the approximate posterior, make not trainable
        q_ = q.non_trainable_copy()
        qcav = q.non_trainable_copy()

        # Set natural parameters of cavity
        qcav.nat_params = {
            "np1": q.nat_params["np1"] - self.t.nat_params["np1"],
            "np2": q.nat_params["np2"] - self.t.nat_params["np2"],
        }
           
        # Reset optimiser
        # TODO: not optimising model parameters for now (inducing points,
        # kernel hyperparameters, observation noise etc.).
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, hyper["optimiser"])(
            q.parameters(), **hyper["optimiser_params"])
        
        # Set up data
        x = data["x"]
        y = data["y"]
        
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
        for i in range(hyper["epochs"]):
            
            epoch = {
                "elbo" : 0,
                "kl"   : 0,
                "ll"   : 0,
            }
            
            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                
                batch = {
                    "x" : x_batch,
                    "y" : y_batch,
                }
                
                # Compute KL divergence between q and qcav
                kl = q.kl_divergence(qcav)
                
                # Sample θ from q and compute p(y | θ, x) for each θ
                thetas = q.rsample((hyper["num_elbo_samples"],))
                ll = self.model.likelihood_log_prob(
                    batch, thetas).mean(0).sum()

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
        self._training_curves.append(training_curve)

        # Compute new local contribution from old distributions
        t_new = self.t.compute_refined_factor(q, q_)
        
        return q, t_new

    @property
    def training_curves(self):
        return self._training_curves
