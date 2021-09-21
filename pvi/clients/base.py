import logging
import torch
import numpy as np

from abc import ABC
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

# =============================================================================
# Client class
# =============================================================================

import sys

class Client(ABC):
    
    def __init__(self, data, model, t=None, config=None):

        if config is None:
            config = {}

        #self._config = self.get_default_config()
        self._config = config
        
        # Set data partition and likelihood
        self.data = data
        self.model = model
        
        # Set likelihood approximating term
        self.t = t
        
        self.log = defaultdict(list)
        self._can_update = True

        self.optimiser = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    @classmethod
    def get_default_config(cls):
        return {}

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        """
        return self._can_update
    
    def fit(self, q, init_q=None):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        return self.update_q(q, init_q)

    def update_q(self, q, init_q=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """

        # Type(q) is self.model.conjugate_family.
        if str(type(q)) == str(self.model.conjugate_family) \
                and not self.config["train_model"]:
            # No need to make q trainable.
            q_new, self.t = self.model.conjugate_update(self.data, q, self.t)
        else:
            # Pass a trainable copy to optimise.
            q_new, self.t = self.gradient_based_update(p=q, init_q=init_q)

        return q_new, self.t

    def gradient_based_update(self, p, init_q=None):
        # Cannot update during optimisation.
        self._can_update = False
        
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
            parameters = q.parameters()

        # Reset optimiser
        # NOTE: why is optimiser reset here?
        logging.info("Resetting optimiser")
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
        if self.config['dp_mode'] not in ['dpsgd','param_fixed']:
            loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)
            n_epochs = self.config['epochs']
            n_samples = len(loader)

        # swor data pass modes
        else:
            # regular SWOR sampler
            if self.config['dp_mode'] == 'dpsgd':
                sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(tensor_dataset, replacement=False), batch_size=self.config['batch_size'], drop_last=False)
            
                loader = DataLoader(tensor_dataset, batch_sampler=sampler)

            # use only fixed single minibatch for local learning for each global update
            elif self.config['dp_mode'] == 'param_fixed':
                inds = torch.randint(low=0,high=len(tensor_dataset),size=(self.config['batch_size'],))
                loader = DataLoader( torch.utils.data.Subset(tensor_dataset, indices=inds) )
            else:
                raise ValueError(f"Unexpected dp_mode in base client: {self.config['dp_mode']}")

            n_epochs = 1
            n_samples = self.config['epochs']

            #print(f"loader len: {len(loader)}, n_steps: {self.config['n_steps']}")
            #print( int(np.ceil(self.config['n_steps']/len(loader) )))
            #print(  self.config['n_steps'] % len(loader) )
            #sys.exit()

        # Dict for logging optimisation progress
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
            "logt" : [],
        }
        
        # Gradient-based optimisation loop -- loop over epochs
        #epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
        #                  leave=True)
        epoch_iter = tqdm(range(n_epochs), desc="Epoch", leave=True)
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo" : 0,
                "kl"   : 0,
                "ll"   : 0,
                "logt" : 0,
            }
            
            # Loop over batches in current epoch
            tmp = iter(loader)
            #for i_step, (x_batch, y_batch) in enumerate(iter(loader)):
            for i_step in range(n_samples):
                try:
                    (x_batch, y_batch) = tmp.next()
                except StopIteration as err:
                    tmp = iter(loader)
                    (x_batch, y_batch) = tmp.next()

                if self.config['dp_mode'] == 'dpsgd':
                    # simple DP-SGD implementation
                    trace_tmp = np.zeros(2) # for keeping track of ELBO and logl

                    # initialise grad accumulator
                    cum_grads = {}
                    for i_weight, p_ in enumerate(filter(lambda p_: p_.requires_grad, q.parameters())):
                        #if p_.grad is not None:
                        cum_grads[str(i_weight)] = torch.zeros_like(p_)

                    # Compute KL divergence between q and q_cav.
                    try:
                        kl = q.kl_divergence(q_cav).sum()/self.config['batch_size']
                        #print(f'kl shape: {q.kl_divergence(q_cav).shape}')
                        #print(kl)
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

                    if self.t is not None:
                        # Compute E_q[log t(θ)]. this is only for bookkeeping, not used in loss
                        logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                        #logt /= self(x) # use full data len to be comparable; doesn't matter since not used in optimisation

                    # quick hack for DP-SGD: process each sample separately
                    for x_single, y_single in zip(x_batch,y_batch):
                        optimiser.zero_grad()

                        batch = {
                            "x" : torch.unsqueeze(x_single,0),
                            "y" : torch.unsqueeze(y_single,0),
                        }
                        #print(batch['x'].shape, batch['y'].shape)
                        #sys.exit()

                        # note: avg over minibatch only after adding clipping per-example grads & noising the sum
                        # Sample θ from q and compute p(y | θ, x) for each θ
                        ll = self.model.expected_log_likelihood(
                            batch, q, self.config["num_elbo_samples"]).sum()
                        #ll /= self.config['batch_size'] # rescale when using minibatches

                        # Negative local free energy is KL minus log-probability.
                        loss = kl - ll
                        loss.backward(retain_graph=True) # keep graph when kl is computed outside loop

                        trace_tmp[0] += ll.item()
                        trace_tmp[1] += -loss.item()


                        # NOTE: assume that all parameters for dp are from q
                        #if self.config['use_dpsgd']:
                        g_norm = torch.zeros(1)
                        for p_ in filter(lambda p_: p_.requires_grad, q.parameters()):
                            #if p_.grad is not None:
                            g_norm += torch.sum(p_.grad**2)
                        g_norm = torch.sqrt(g_norm)
                        #print(f'grad_norm: {g_norm}')
                        
                        # clip and accumulate grads
                        for i_weight, p_ in enumerate(filter(lambda p_: p_.requires_grad, q.parameters())):
                            #if p_.grad is not None:
                            cum_grads[str(i_weight)] += p_.grad/torch.clamp(g_norm/self.config['dp_C'], min=1)
                        # check that clipping is ok
                        #g_norm = torch.zeros(1)
                        #for k in cum_grads:
                        #    g_norm += torch.sum(cum_grads[k]**2)
                        #g_norm = torch.sqrt(g_norm)
                        #print(f'grad_norm: {g_norm}')

                        #print(list(q.parameters()))
                        #print(f'cum grad:\n{cum_grads}')
                        #sys.exit()
                
                # no dpsgd
                else:

                    optimiser.zero_grad()
                    batch = {
                        "x" : x_batch,
                        "y" : y_batch,
                    }

                    # Compute KL divergence between q and q_old.
                    # kl = q.kl_divergence(q_old).sum() / len(x)
                    # Compute KL divergence between q and q_cav.
                    try:
                        kl = q.kl_divergence(q_cav).sum() / self.config['batch_size']#len(x)
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


                # add noise to clipped grads and avg
                if self.config['dp_mode']== 'use_dpsgd':
                    for key, p_ in zip( cum_grads, filter(lambda p_: p_.requires_grad, q.parameters()) ):
                        #print(f'grad before:\n{p_.grad}')
                        #print(f'noiseless accumulated grads:\n{cum_grads[key]}')
                        p_.grad = self.config['dp_C']*self.config['dp_sigma']*torch.randn_like(p_.grad) + cum_grads[key]
                        p_.grad /= self.config['batch_size']
                        #print(f'grad after:\n{p_.grad}') # note: currently scaling changed when noising as well
                        #sys.exit()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                if self.config['dp_mode']== 'use_dpsgd':
                    epoch["elbo"] += trace_tmp[1] /  (n_samples * self.config['batch_size'])
                    epoch["kl"] += kl.item() / n_samples
                    epoch["ll"] += trace_tmp[0] / (n_samples * self.config['batch_size'])
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
        

        #'''
        # NOTE: might make sense to define clipping & noise levels separately for np1, np2
        # use single clip & noise level for now
        # get norm of the change in params, clip and noisify
        #'''

        if self.config['dp_mode'] in ['param','param_fixed']:
            param_norm = 0
            #for p_, p_old in zip(q.parameters(),p.trainable_copy().parameters()):
            for i_params, (p_, p_old) in enumerate(zip(q.parameters(),p.trainable_copy().parameters())):
                if i_params == 0:
                    # difference in params
                    param_norm += torch.sum((p_ - p_old)**2)
                    # params directly
                    #param_norm += torch.sum(p_**2)
                elif i_params == 1:
                    # difference in params
                    param_norm += torch.sum( (p_ - p_old)**2)
                    # difference without log
                    #param_norm += torch.sum( (torch.exp(p_ - p_old))**2)
                    # params directly
                    #param_norm += torch.sum(p_**2) # should use exp?

                else:
                    sys.exit('Model has > 2 sets of params, DP not implemented for this!')
                #print(p_)
            param_norm = torch.sqrt(param_norm)
            logger.debug(f'diff in param, norm before clip: {param_norm}')

            # clip and add noise to the difference in params
            # note sensitivities: even with add/replace DP needs 2*C
            for i_params, (p_, p_old) in enumerate(zip(q.parameters(),p.trainable_copy().parameters())):
                #p_.data = p_old + (p_ - p_old)/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                if i_params == 0:
                    p_.data = p_old + (p_ - p_old)/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                            + 2*self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_)
                    # params directly
                    #p_.data = (p_/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                    #        + 2*self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_)).detach().clone()
                elif i_params == 1:
                    # note: need to change inv-scale to non-log scale for clipping & noise

                    #if self.config['pos_def_constants'][0] > 0:
                    #    p_.data = p_old + torch.log( torch.clamp( (torch.exp(p_ - p_old))/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                    #        + 2*self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_), min=self.config['pos_def_constants'][0]))
                    #else:
                    # clip change in params
                    p_.data = p_old + (p_ - p_old)/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                            + 2*self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_)
                    # clip change in params without log-space: enforce pos values and back to log-space
                    #p_.data = p_old + torch.log( torch.clamp( torch.exp(p_ - p_old)/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                    #       + 2*self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_), min=1e-7, max=1e+6) )
                    
                    # params directly
                    #p_.data = (p_/torch.clamp(param_norm/self.config['dp_C'], min=1) \
                    #        + 2*self.config['dp_C'] * self.config['dp_sigma'] * torch.randn_like(p_)).detach().clone()

                else:
                    sys.exit('Model has > 2 sets of params, param DP not implemented for this!')
            #'''

            # check that clipping works properly, uses log-scale, fix if needed
            '''
            param_norm = 0
            for p_, p_old in zip(q.parameters(),p.trainable_copy().parameters()):
                param_norm += torch.sum((p_ - p_old)**2)
                #print(p_)
            param_norm = torch.sqrt(param_norm)
            print(f'param norm after clip: {param_norm}')
            #'''
            

        #sys.exit()

        # Create non-trainable copy to send back to server
        q_new = q.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q, q_old, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"])

            # note for DP: only t is currently used by server:
            #return q_new, t_new
            return None, t_new 

        else:
            logger.debug('Note: client not returning t')
            return q_new, None


class ClientBayesianHypers(ABC):
    """
    PVI client with Bayesian treatment of model hyperparameters.
    """
    def __init__(self, data, model, t=None, teps=None, config=None):

        if config is None:
            config = {}

        #self._config = self.get_default_config()
        self._config = config

        # Set data partition and likelihood
        self.data = data
        self.model = model

        # Set likelihood approximating term
        self.t = t
        self.teps = teps

        self.log = defaultdict(list)
        self._can_update = True

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    @classmethod
    def get_default_config(cls):
        return {
            "damping_factor": 1.,
            "valid_factors": False,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "num_elbo_samples": 10,
            "num_elbo_hyper_samples": 1,
            "print_epochs": 1
        }

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        :return:
        """
        return self._can_update

    def fit(self, q, qeps):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        return self.update_q(q, qeps)

    def update_q(self, q, qeps):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """
        # Pass a trainable copy to optimise.
        q_new, qeps_new, self.t, self.teps = self.gradient_based_update(
            q.trainable_copy(), qeps.trainable_copy())

        return q_new, qeps_new, self.t, self.teps

    def gradient_based_update(self, q, qeps):
        # Cannot update during optimisation.
        self._can_update = False

        if self.t is None:
            # Old posterior = prior, make non-trainable.
            q_cav = q.non_trainable_copy()
            qeps_cav = qeps.non_trainable_copy()
        else:
            q_cav = q.non_trainable_copy()
            q_cav.nat_params = {k: v - self.t.nat_params[k]
                                for k, v in q_cav.nat_params.items()}

            qeps_cav = qeps.non_trainable_copy()
            for k1, v1 in qeps.distributions.items():
                qeps.distributions[k1].nat_params = {
                    k2: v2 - self.teps.factors[k1].nat_params[k2]
                    for k2, v2 in v1.nat_params.items()}

        parameters = list(q.parameters()) + qeps.parameters()

        # Reset optimiser. Parameters are those of q(θ) and q(ε).
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "kl": [],
            "kleps": [],
            "ll": [],
            "logt": [],
            "logteps": [],
        }

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch")
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "kl": 0,
                "kleps": 0,
                "ll": 0,
                "logt": 0,
                "logteps": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(q_cav).sum() / len(x)
                kleps = sum(qeps.kl_divergence(qeps_cav).values()) / len(x)

                # Estimate E_q[log p(y | x, θ, ε)].
                ll = 0
                for _ in range(self.config["num_elbo_hyper_samples"]):
                    eps = qeps.rsample()
                    self.model.hyperparameters = eps
                    ll += self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]).sum()

                ll /= (self.config["num_elbo_hyper_samples"] * len(x_batch))

                if self.t is not None:
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                    logteps = sum(self.teps.eqlogt(
                        qeps, self.config["num_elbo_samples"]).values())
                    logt /= len(x)
                    logteps /= len(x)

                    # loss = kl + kleps - ll + logt - logteps

                loss = kl + kleps - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["kleps"] += kleps.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

                if self.t is not None:
                    epoch["logt"] += logt.item() / len(loader)
                    epoch["logteps"] += logteps.item() / len(loader)

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["kleps"].append(epoch["kleps"])
            training_curve["ll"].append(epoch["ll"])

            if self.t is not None:
                training_curve["logt"].append(epoch["logt"])
                training_curve["logteps"].append(epoch["logteps"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"KL eps: {epoch['kleps']:.3f}, "
                             f"log t: {epoch['logt']: .3f},"
                             f"log teps: {epoch['logteps']: .3f},"
                             f"Epochs: {i}.")

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"], kleps=epoch["kleps"],
                                   logt=epoch["logt"],
                                   logteps=epoch["logteps"])

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Create non_trainable_copy to send back to server.
        q_new = q.non_trainable_copy()
        qeps_new = qeps.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q, q_cav, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"])
            teps_new = self.teps.compute_refined_factor(
                qeps, qeps_cav, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"])

            return q_new, qeps_new, t_new, teps_new

        else:
            return q_new, qeps_new, None, None
