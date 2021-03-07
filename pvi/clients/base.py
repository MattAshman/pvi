from abc import ABC, abstracmethod

import logging


# =============================================================================
# Base client class
# =============================================================================


class Client:
    
    def __init__(self, data, likelihood, t, hyperparameters):
        
        # Set data partition and likelihood
        self.data = data
        self.likelihood = likelihood
        
        # Set likelihood approximating term
        self.t = t
        
        # Set hyperparameters and initialise training curves
        self.hyperparameters = hyperparameters
        self._training_curves = []
        
    
    @abstractmethod
    def fit(self):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        pass
    
    
    def q_update(self, q):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """
        
        if q.family is self.likelihood.conjuagte_family:
            return self.likelihood.conjugate_update(self.data, q, self.t)
            
        else:
            return self.gradient_based_update(self.data, q, self.t)
        
        
    def gradient_based_update(self, data, q, t):
        
        # Initialise approximate posterior being optimised
        # TODO: create approximate posterior class
        q_ = q.deepcopy()
           
        # Reset optimiser
        logging.info("Resetting optimiser")
        
        optimiser = getattr(torch.optim, self.hyperparameters["optimiser"])(
                    **self.hyperparameters["optimiser_params"])
        
        # Set up data
        x = data["x"]
        y = data["y"]
        
        tensor_dataset = TensorDataset(x, y)
        
        loader = DataLoader(tensor_dataset,
                            batch_size=self.hyperparameters["batch_size"],
                            shuffle=True)

        # Dict for logging optimisation progress
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
        }
        
        # Gradient-based optimisation loop -- loop over epochs
        for i in range(self.hyperparameters["epochs"]):
            
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
                
                # Compute KL divergence between q and q_
                kl = q.kl_divergence(q_)
                
                # Sample θ from q and compute p(y | θ, x) for each θ
                thetas = q.rsample((self.hyperparameters["num_elbo_samples"],))
                ll = self.likelihood.likelihood_forward(batch, thetas).mean(0).sum()
                ll = ll + t.log_prob(thetas).mean(0).sum()

                # Negative local Free Energy is KL minus log-probability
                loss = kl - ll
                loss.backward()
                self.optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item()
                epoch["kl"] += kl.item()
                epoch["ll"] += ll.item()

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if i % self.hyperparameters["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch["elbo"]:.3f}, "
                             f"LL: {epoch["ll"]:.3f}, "
                             f"KL: {epoch["kl"]:.3f}, "
                             f"Epochs: {i}.")

        # Log the training curves for this update
        self._training_curves.append(training_curve)

        # Compute new local contribution from old distributions
        t = t.compute_refined_factor(q, q_)
        
        return q, t