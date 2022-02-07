import logging
import torch
import numpy as np

from abc import ABC
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

from .base import Client

# =============================================================================
# Param DP client class
# =============================================================================

import sys

class Param_DP_Client(Client):
    
    def __init__(self, data, model, t=None, config=None):

        if config is None:
            config = {}

        super().__init__(data=data, model=model, t=t, config=config)

        #self._config = self.get_default_config()
        #self._config = config
        
        # Set data partition and likelihood
        #self.data = data
        #self.model = model
        
        # Set likelihood approximating term
        #self.t = t
        
        #self.log = defaultdict(list)
        #self._can_update = True

        #self.optimiser = None

        if self._config['track_client_norms']:
            self.pre_dp_norms = []
            self.post_dp_norms = []
            self.noise_norms = []


    def gradient_based_update(self, p, init_q=None):
        # Cannot update during optimisation.
        self._can_update = False

        if self.config['batch_size'] is None:
            batch_size = int(np.floor(self.config['sampling_frac_q']*len(self.data["y"])))
        else:
            batch_size = self.config['batch_size']
        #print(f"batch size {batch_size}, noise std:{self.config['dp_sigma']}, noise var: {self.config['dp_sigma']**2}")
        #sys.exit()
        
        # Copy the approximate posterior, make old posterior non-trainable.
        q_old = p.non_trainable_copy()

        if self.t is None:
            # Standard VI: prior = old posterior.
            q_cav = p.non_trainable_copy()
        else:
            # TODO: check if valid distribution.

            q_cav = p.non_trainable_copy()
            q_cav.nat_params = {k: v - self.t.nat_params[k]
                                for k, v in q_cav.nat_params.items()}

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            # Initialise to prior.
            q = p.trainable_copy()

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            if "model_optimiser_params" in self.config:
                parameters = [
                    {"params": q.parameters()},
                    {"params": self.model.parameters(),
                     **self.config["model_optimiser_params"]}
                ]
            else:
                parameters = [
                    {"params": q.parameters()},
                    {"params": self.model.parameters()}
                ]
        else:
            if self.freeze_var_updates > self.update_counter:
                logger.debug('Freezing log_scale params')
                q._unc_params['log_scale'].requires_grad = False
            parameters = q.parameters()

        # Reset optimiser
        # NOTE: why is optimiser reset here?
        #logging.info("Resetting optimiser")
        #if self.optimiser is None:
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])
        lr_scheduler = getattr(torch.optim.lr_scheduler,
                           self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"])
        self.optimiser = optimiser
        self.lr_scheduler = lr_scheduler
        #else:
        #    optimiser = self.optimiser
        #    lr_scheduler = self.lr_scheduler
        
        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)

        # set up data loader with chosen sampling type
        # sequential data pass modes
        if self.config['dp_mode'] == 'param':
            loader = DataLoader(tensor_dataset,
                            batch_size=batch_size,
                            shuffle=True)
            n_epochs = self.config['epochs']
            n_samples = len(loader)

        # use only fixed single minibatch for local learning for each global update
        elif self.config['dp_mode'] == 'param_fixed':
            raise NotImplementedError('Check param_fixed code before running!')
            inds = torch.randint(low=0,high=len(tensor_dataset),size=(batch_size,))
            loader = DataLoader( torch.utils.data.Subset(tensor_dataset, indices=inds) )
        else:
            raise ValueError(f"Unexpected dp_mode in base client: {self.config['dp_mode']}")


        # Dict for logging optimisation progress
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
            "logt" : [],
        }
        
        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(n_epochs), desc="Epoch", leave=True, disable=self.config['pbar'])

        for i in epoch_iter:
            epoch = {
                "elbo" : 0,
                "kl"   : 0,
                "ll"   : 0,
                "logt" : 0,
            }
            
            # Loop over batches in current epoch
            tmp = iter(loader)

            for i_step in range(n_samples):
                try:
                    (x_batch, y_batch) = tmp.next()
                except StopIteration as err:
                    tmp = iter(loader)
                    (x_batch, y_batch) = tmp.next()

                #logger.debug(f'optimiser starting step {i_step} with total batch_size {len(y_batch)}')

                optimiser.zero_grad()
                batch = {
                    "x" : x_batch,
                    "y" : y_batch,
                }

                # Compute KL divergence between q and q_cav.
                try:
                    kl = q.kl_divergence(q_cav).sum()/len(self.data["x"])
                except ValueError as err:
                    # NOTE: removed dirty fix: q_cav not guaranteed to give proper std, might give errors
                    print('\nException in KL: probably caused by invalid cavity distribution')
                    #print(q._unc_params['log_scale'])
                    print(q_cav)
                    print('nat params')
                    print(q_cav.nat_params)
                    print('std params')
                    print(q_cav.std_params)
                    raise err

                # Sample θ from q and compute p(y | θ, x) for each θ
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]).sum()
                ll /=len(x_batch)

                if self.t is not None:
                    # how slow is this?
                    # Compute E_q[log t(θ)].this is only for bookkeeping, not used in loss
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                    #logt /= len(x) # use full data len to be comparable; doesn't matter since not used in optimisation

                # Negative local free energy is KL minus log-probability.
                loss = kl - ll
                loss.backward()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                if self.config['dp_mode']== 'dpsgd':
                    epoch["elbo"] += trace_tmp[1] /  (n_samples * batch_size)
                    epoch["kl"] += kl.item() / n_samples
                    epoch["ll"] += trace_tmp[0] / (n_samples * batch_size)
                    if self.t is not None:
                        epoch["logt"] += logt.item() / n_samples
                else:
                    epoch["elbo"] += -loss.item() / n_samples
                    epoch["kl"] += kl.item() / n_samples
                    epoch["ll"] += ll.item() / n_samples
                    if self.t is not None:
                        epoch["logt"] += logt.item() / n_samples

                optimiser.step()
                i_step += 1

                ### end loop over minibatches ###

            optimiser.zero_grad()

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

            ### end loop over local steps ###

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)
        
        # NOTE: might make sense to define clipping & noise levels separately for np1, np2
        # use single clip & noise level for now

        with torch.no_grad():
            param_norm = 0

            # clip and noisify natural params as opposed to (unconstrained) loc-scale
            if self.config['noisify_np']:
                for k in q.nat_params:
                    param_norm += torch.sum((q.nat_params[k] - p.nat_params[k])**2)

            else:
                # in unconstrained space
                for i_params, (p_, p_old) in enumerate(zip(q.parameters(),p.trainable_copy().parameters())):
                    if i_params == 0:
                        # difference in params
                        param_norm += torch.sum((p_ - p_old)**2)
                        # params directly
                        #param_norm += torch.sum(p_**2)
                    elif i_params == 1:
                        # difference in params
                        param_norm += torch.sum( (p_ - p_old)**2)
                        # params directly
                        #param_norm += torch.sum(p_**2) # should use exp?
                    else:
                        sys.exit('Model has > 2 sets of params, DP not implemented for this!')

            param_norm = torch.sqrt(param_norm)
            #logger.debug(f'diff in param, norm before clip: {param_norm}')
            if self.config['track_client_norms']:
                self.pre_dp_norms.append(param_norm)
                #self.noise_norms

            # clip and add noise to the difference in params
            if self.config['noisify_np']:
                #print('\nbefore noising: {}\n'.format(q.nat_params))
                tmp = {}
                tmp0 = 0.
                for k in q.nat_params:
                    noise = self.config['dp_C']*self.config['dp_sigma']*torch.randn_like(q.nat_params[k])
                    if self.config['track_client_norms']:
                        tmp0 += torch.linalg.norm(noise, ord=2)
                    tmp[k] = (p.nat_params[k] + (q.nat_params[k]-p.nat_params[k])/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                                + noise).detach().clone()

                if self.config['track_client_norms']:
                    self.noise_norms.append(tmp0)
                tmp = q._unc_from_std(q._std_from_nat(tmp))
                for p_new, k in zip(q.parameters(),tmp):
                    p_new.data =  tmp[k]

                # check clipping:
                '''
                param_norm = 0
                for k in q.nat_params:
                    param_norm += torch.sum((q.nat_params[k] - p.nat_params[k])**2)
                param_norm = torch.sqrt(param_norm)
                print(f'norm after clipping and noise: {param_norm}')

                print('\nafter noising: {}\n'.format(q.nat_params))
                sys.exit()
                '''
            else:
                for i_params, (p_, p_old) in enumerate(zip(q.parameters(),p.trainable_copy().parameters())):
                    if i_params == 0:
                        p_.data = p_old + (p_ - p_old)/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                                + self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_)
                        # params directly
                        #p_.data = (p_/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                        #        + self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_)).detach().clone()
                    elif i_params == 1:
                        # clip change in params
                        p_.data = p_old + (p_ - p_old)/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                                + self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_)
                        
                        # params directly
                        #p_.data = (p_/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                        #        + self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_)).detach().clone()

                    else:
                        sys.exit('Model has > 2 sets of params, param DP not implemented for this!')

            if self.config['track_client_norms']:
                if self.config['noisify_np']:
                    for k in q.nat_params:
                        param_norm += torch.sum((q.nat_params[k] - p.nat_params[k])**2)
                else:
                    raise NotImplementedError('norm tracking not implemented properly!')
                param_norm = torch.sqrt(param_norm)
                self.post_dp_norms.append(param_norm)

        # Create non-trainable copy to send back to server
        q_new = q.non_trainable_copy()

        # Finished optimisation, can now update.
        self.update_counter += 1
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q, q_old, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"])

            # note for DP: only t is currently used by server:
            return q_new, t_new

        else:
            logger.debug('Note: client not returning t')
            return q_new, None


