import logging
import torch
import numpy as np
import sys

from abc import ABC
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from .base import Client

logger = logging.getLogger(__name__)




logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)

# =============================================================================
# Client class
# =============================================================================


class DPSGD_Client(Client):
    
    def __init__(self, data, model, t=None, config=None):

        if config is None:
            config = {}

        super().__init__(data, model,t=t, config=config)

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

        #if self._config['track_client_norms']:
        #    self.pre_dp_norms = []
        #    self.post_dp_norms = []


    def gradient_based_update(self, p, init_q=None):
        # Cannot update during optimisation.
        self._can_update = False

        if self.config['batch_size'] is None:
            batch_size = int(np.floor(self.config['sampling_frac_q']*len(self.data["y"])))
        else:
            batch_size = self.config['batch_size']
        #print(f"batch size {batch_size}, noise std:{self.config['dp_sigma']}")
        assert batch_size > 0
        
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
            #print(q._unc_params['log_scale'].requires_grad)
            if self.freeze_var_updates > self.update_counter:
                logger.debug('Freezing log_scale params')
                q._unc_params['log_scale'].requires_grad = False

            parameters = q.parameters()

        #print(q._unc_params['log_scale'].requires_grad)

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
        # regular SWOR sampler
        if self.config['dp_mode'] == 'dpsgd':
            #logger.debug('setting sampler for dpsgd')
            sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(tensor_dataset, replacement=False), batch_size=batch_size, drop_last=False)
        
            loader = DataLoader(tensor_dataset, batch_sampler=sampler)
        else:
            raise ValueError(f"Unexpected dp_mode in base client: {self.config['dp_mode']}")

        n_epochs = 1
        n_samples = self.config['epochs']

        # Dict for logging optimisation progress
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
            "logt" : [],
        }
        
        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(n_epochs), desc="Epoch", leave=True, disable=self.config['pbar'])
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

            for i_step in range(n_samples):
                try:
                    (x_batch, y_batch) = tmp.next()
                except StopIteration as err:
                    tmp = iter(loader)
                    (x_batch, y_batch) = tmp.next()

                #logger.debug(f'optimiser starting step {i_step} with total batch_size {len(y_batch)}')

                # simple DP-SGD implementation
                trace_tmp = np.zeros(2) # for keeping track of ELBO and logl

                # initialise grad accumulator
                cum_grads = {}
                for i_weight, p_ in enumerate(filter(lambda p_: p_.requires_grad, q.parameters())):
                    #if p_.grad is not None:
                    cum_grads[str(i_weight)] = torch.zeros_like(p_)

                if self.t is not None:
                    # Compute E_q[log t(θ)]. this is only for bookkeeping, not used in loss
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                    #logt /= self(x) # use full data len to be comparable; doesn't matter since not used in optimisation

                grad_norm_tracker = 0
                # quick hack for DP-SGD: process each sample separately
                for x_single, y_single in zip(x_batch,y_batch):
                    optimiser.zero_grad()

                    batch = {
                        "x" : torch.unsqueeze(x_single,0),
                        "y" : torch.unsqueeze(y_single,0),
                    }
                    #print(batch['x'].shape, batch['y'].shape)
                    #sys.exit()

                    # Compute KL divergence between q and q_cav.
                    try:
                        kl = q.kl_divergence(q_cav).sum()/len(self.data["x"])
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

                    # note: avg over minibatch only after clipping per-example grads & noising the sum
                    # Sample θ from q and compute p(y | θ, x) for each θ
                    ll = self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]).sum()
                    #ll /= self.config['batch_size'] # rescale when using minibatches

                    # Negative local free energy is KL minus log-probability.
                    loss = kl - ll


                    loss.backward(retain_graph=False) # keep graph when kl is computed outside loop

                    trace_tmp[0] += ll.item()
                    trace_tmp[1] += -loss.item()

                    # NOTE: assume that all parameters for dp are from q
                    g_norm = torch.zeros(1)
                    for p_ in filter(lambda p_: p_.requires_grad, q.parameters()):
                        #if p_.grad is not None:
                        g_norm += torch.sum(p_.grad**2)
                    g_norm = torch.sqrt(g_norm)
                    #print(f'grad_norm before clipping: {g_norm}')
                    
                    if self._config['track_client_norms']:
                        # track mean grad norm over samples in the minibatch
                        grad_norm_tracker += g_norm.item()/batch_size

                    # clip and accumulate grads
                    for i_weight, p_ in enumerate(filter(lambda p_: p_.requires_grad, q.parameters())):
                        #if p_.grad is not None:
                        cum_grads[str(i_weight)] += (p_.grad/torch.clamp(g_norm/self.config['dp_C'], min=1)).detach().clone()
                    # check that clipping is ok
                    '''
                    g_norm2 = torch.zeros(1)
                    for p_ in filter(lambda p_: p_.requires_grad, q.parameters()):
                        #if p_.grad is not None:
                        g_norm2 += torch.sum((p_.grad/torch.clamp(g_norm/self.config['dp_C'], min=1))**2)
                    g_norm2 = torch.sqrt(g_norm2)
                    print(f'grad_norm after clipping: {g_norm2}')
                    #'''

                # add noise to clipped grads and avg
                for key, p_ in zip( cum_grads, filter(lambda p_: p_.requires_grad, q.parameters()) ):
                    p_.grad = self.config['dp_C']*self.config['dp_sigma']*torch.randn_like(p_.grad) + cum_grads[key]
                    p_.grad /= batch_size

                if self._config['track_client_norms']:
                    self.pre_dp_norms.append(grad_norm_tracker)
                    # calculate grad norm post DP treatment
                    g_norm = torch.zeros(1)
                    for p_ in filter(lambda p_: p_.requires_grad, q.parameters()):
                        g_norm += torch.sum(p_.grad**2)
                    g_norm = torch.sqrt(g_norm)
                    self.post_dp_norms.append(g_norm.item())

                ### end loop over single samples in minibatch ###

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += trace_tmp[1] /  (n_samples * batch_size)
                epoch["kl"] += kl.item() / n_samples
                epoch["ll"] += trace_tmp[0] / (n_samples * batch_size)
                if self.t is not None:
                    epoch["logt"] += logt.item() / n_samples

                optimiser.step()
                i_step += 1

                ### end loop over minibatches ###

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
        
        # Create non-trainable copy to send back to server
        q_new = q.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        self.update_counter += 1

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


class Userlevel_DPSGD_Client(Client):
    """DPSGD client where sampling is on user level:
    config['batch_size'] defines numebr of data points per user (+1 for first users if uneven number of samples), while config['sampling_frac_q'] is used for setting user-level batch size for random sampling
    """
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

        assert config['batch_size'] is not None and config['sampling_frac_q'] is not None

        #self.sample_on_user_level = True
        # note: config['batch_size'] determines number of samples per user, config['sampling_frac_q'] is used for determining batch size when sampling users
        self.n_users = int(np.floor( len(self.data["y"])//config['batch_size'] ) )

        # use given batch_size arg to determine user-level dataset size (fixed over training)
        # give first users more data if samples are left over
        tmp1 = config['batch_size']
        tmp2 = len(self.data['y']) - self.n_users*config['batch_size']
        self.user_data_sizes = np.zeros(self.n_users, dtype=int) + tmp1
        self.user_data_sizes[:tmp2] += 1
        #print(tmp1,tmp2)
        #print(np.unique(self.user_data_sizes))
        #print(len(self.user_data_sizes))
        #print(batch_sizes)
        #print(np.sum(self.user_data_sizes), len(self.data['y']))
        del tmp1,tmp2
        assert np.sum(self.user_data_sizes) == len(self.data['y'])
        assert self.n_users == len(self.user_data_sizes)

        # set batch_size used for sampling users
        self.batch_size = int(np.floor(config['sampling_frac_q']*self.n_users) )
        if self.batch_size < 1:
            raise ValueError('Using batch_size < 1 for sampling users! Try increasing sampling frac!')

        #print(f"total data size: {len(self.data['y'])}, sum of user datas: {np.sum(self.user_data_sizes)}, sample on user level: {self.sample_on_user_level}, sampling batch size {self.batch_size}, noise std:{self.config['dp_sigma']}")
        #print(f"number of users: {self.n_users}, user data size: {self.user_data_size}, amounts to {self.user_data_size*self.n_users} samples")

        if self._config['track_client_norms']:
            self.pre_dp_norms = []
            self.post_dp_norms = []

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
        if self.config['dp_mode'] == 'dpsgd':
            # sampler on user id level
            loader = DataLoader(torch.linspace(0,self.n_users-1,self.n_users, dtype=torch.int),batch_size=self.batch_size,shuffle=True)

        else:
            raise ValueError(f"Unexpected dp_mode in base client: {self.config['dp_mode']}")

        n_epochs = 1
        n_samples = self.config['epochs']

        # Dict for logging optimisation progress
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
            "logt" : [],
        }
        
        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(n_epochs), desc="Epoch", leave=True, disable=self.config['pbar'])
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

            for i_step in range(n_samples):

                # note: implement only user-level sampling here, no need to worry about having both
                try:
                    user_inds = tmp.next()
                except StopIteration as err:
                    tmp = iter(loader)
                    user_inds = tmp.next()

                # simple DP-SGD implementation
                trace_tmp = np.zeros(2) # for keeping track of ELBO and logl

                # initialise grad accumulator
                cum_grads = {}
                for i_weight, p_ in enumerate(filter(lambda p_: p_.requires_grad, q.parameters())):
                    #if p_.grad is not None:
                    cum_grads[str(i_weight)] = torch.zeros_like(p_)

                if self.t is not None:
                    # Compute E_q[log t(θ)]. this is only for bookkeeping, not used in loss
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                    #logt /= self(x) # use full data len to be comparable; doesn't matter since not used in optimisation

                grad_norm_tracker = 0
                # quick hack for DP-SGD: process each sample separately
                # process each sampled users' data as a single batch: avg over chosen users' grads, clip per-user avg and add noise to sum over averages, finally avg over number of users
                # note: could also try clipping & noising sum over single users' data
                for i_user in user_inds: #x_single, y_single in zip(x_batch,y_batch):
                    optimiser.zero_grad()

                    # pick indices first
                    inds = np.linspace( np.sum(self.user_data_sizes[:i_user]), np.sum(self.user_data_sizes[:(i_user+1)])-1, self.user_data_sizes[i_user] )

                    batch = {
                        "x" : tensor_dataset[inds][0],
                        "y" : tensor_dataset[inds][1],
                    }

                    #print(batch['x'].shape, batch['y'].shape)
                    #sys.exit()

                    # Compute KL divergence between q and q_cav.
                    # need to scale to match user data size (compare to LFA scaling)
                    # currently scaled (in avging logl)
                    try:
                        kl = q.kl_divergence(q_cav).sum()/len(self.data["x"])
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

                    # note: avg over minibatch only after clipping per-example grads & noising the sum
                    # Sample θ from q and compute p(y | θ, x) for each θ
                    # NOTE: instead of per-user avg, could also use sum over user grads to give more weight according to number of samples
                    ll = self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]).sum()
                    ll /= len(batch['y']) # rescale when using minibatches over users

                    # Negative local free energy is KL minus log-probability.
                    loss = kl - ll

                    loss.backward(retain_graph=False) # keep graph when kl is computed outside loop

                    trace_tmp[0] += ll.item()
                    trace_tmp[1] += -loss.item()

                    # NOTE: assume that all parameters for dp are from q
                    g_norm = torch.zeros(1)
                    for p_ in filter(lambda p_: p_.requires_grad, q.parameters()):
                        #if p_.grad is not None:
                        g_norm += torch.sum(p_.grad**2)
                    g_norm = torch.sqrt(g_norm)
                    #print(f'grad_norm before clipping: {g_norm}')
                    
                    if self._config['track_client_norms']:
                        # track mean grad norm over users sampled in the minibatch
                        grad_norm_tracker += g_norm.item()/self.batch_size

                    # clip and accumulate grads
                    for i_weight, p_ in enumerate(filter(lambda p_: p_.requires_grad, q.parameters())):
                        #if p_.grad is not None:
                        cum_grads[str(i_weight)] += (p_.grad/torch.clamp(g_norm/self.config['dp_C'], min=1)).detach().clone()
                    # check that clipping is ok
                    '''
                    g_norm2 = torch.zeros(1)
                    for p_ in filter(lambda p_: p_.requires_grad, q.parameters()):
                        #if p_.grad is not None:
                        g_norm2 += torch.sum((p_.grad/torch.clamp(g_norm/self.config['dp_C'], min=1))**2)
                    g_norm2 = torch.sqrt(g_norm2)
                    print(f'grad_norm after clipping: {g_norm2}')
                    #'''

                # add noise to clipped grads and avg over different users
                for key, p_ in zip( cum_grads, filter(lambda p_: p_.requires_grad, q.parameters()) ):
                    p_.grad = self.config['dp_C']*self.config['dp_sigma']*torch.randn_like(p_.grad) + cum_grads[key]
                    p_.grad /= self.batch_size

                if self._config['track_client_norms']:
                    self.pre_dp_norms.append(grad_norm_tracker)
                    # calculate grad norm post DP treatment
                    g_norm = torch.zeros(1)
                    for p_ in filter(lambda p_: p_.requires_grad, q.parameters()):
                        g_norm += torch.sum(p_.grad**2)
                    g_norm = torch.sqrt(g_norm)
                    self.post_dp_norms.append(g_norm.item())

                ### end loop over single samples in minibatch ###

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += trace_tmp[1] /  (n_samples * self.batch_size)
                epoch["kl"] += kl.item() / n_samples
                epoch["ll"] += trace_tmp[0] / (n_samples * self.batch_size)
                if self.t is not None:
                    epoch["logt"] += logt.item() / n_samples

                optimiser.step()
                i_step += 1

                ### end loop over minibatches ###

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
            return q_new, t_new

        else:
            logger.debug('Note: client not returning t')
            return q_new, None
