
from collections import defaultdict
import copy
import itertools
import logging
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from .base import Client
from pvi.distributions.exponential_family_distributions import MeanFieldGaussianDistribution
from pvi.distributions.exponential_family_factors import MeanFieldGaussianFactor
from pvi.servers.synchronous_server import SynchronousServer

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

# =============================================================================
# Local PVI client class
# =============================================================================


class VaryingBatchSampler():
    """
    Batch sampler class that supports different batch size for each batch
    """

    def __init__(self, batch_sizes):
        self.batch_sizes = batch_sizes

    def __iter__(self):
        return iter([list(range(cur-b,cur)) for cur,b in zip(itertools.accumulate(self.batch_sizes),iter(self.batch_sizes))])

    def __len__(self) -> int:
        return len(self.batch_sizes)


class LocalPVIClient(Client):
    
    def __init__(self, data, model, t, config=None):
        
        super().__init__(data=data, model=model, t=t, config=config)

        if config is None:
            config = {}

        self._config = config
        
        #print(self.config)
        
        # Set data partition and likelihood
        self.data = data
        self.model = model

        # Initialise optimiser states
        if config['batch_size'] is None:
            self.n_local_models = int(np.ceil(1/(config['sampling_frac_q']))  )
        else:
            self.n_local_models = int(np.ceil((self.data['y'].shape[-1])/config['batch_size']))
        
        self.optimiser_states = None
        
        # Set likelihood approximating term
        self.t = t # note: use internal partitions instead of single t for optimising, keep track of joint t with this
        
        self.log = defaultdict(list)
        self._can_update = True

        self.pseudo_clients = None

        # note: no tracking currently
        # actually tracks norm of change in params for LFA
        #if self._config['track_client_norms']:
        #    self.pre_dp_norms = []
        #    self.post_dp_norms = []


    def gradient_based_update(self, p, init_q=None):
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make old posterior non-trainable.
        q_old = p.non_trainable_copy()

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            # Initialise to prior.
            q = p.trainable_copy()

        
        if self.config['sampling_frac_q'] is not None:
            # use different b for different models to use all data once: first models might have more data than last
            # min batch_size from number of local models
            tmp1 = int(np.floor((self.data['y'].shape[-1])/self.n_local_models))
            if tmp1 == 0:
                raise ValueError('Using batch_size=0! Try increasing sampling frac!')
            tmp2 = len(self.data['y']) - tmp1*self.n_local_models
            batch_sizes = np.zeros(self.n_local_models, dtype=int) + tmp1
            batch_sizes[:tmp2] += 1
            #print(tmp1,tmp2)
            #sys.exit()
        else:
            batch_sizes =  np.zeros(self.n_local_models, dtype=int) + self.config['batch_size']
            if np.amax(self.n_local_models*batch_sizes) > len(self.data['y']):
                raise ValueError('Too large batch size!')
        #print(f'batch sizes: {batch_sizes}, sum to {np.sum(batch_sizes)}')

        # create internal pseudo clients if don't exist
        if self.pseudo_clients is None:
            self.pseudo_clients = []
            cur_ind = 0
            for i_client, b in enumerate(batch_sizes):
                # Data of ith client
                data = {'x' : self.data['x'][cur_ind:(cur_ind+b),:], 'y' : self.data['y'][cur_ind:(cur_ind+b)] }
                cur_ind += b

                client_config =  copy.deepcopy(self.config)
                client_config['dp_mode'] = 'param'
                client_config['clients'] = self.n_local_models # CHECK: SHOULD INCLUDE PSEUDO-CLIENTS?
                client_config['batch_size'] = int(b)

                t = MeanFieldGaussianFactor(nat_params = self.t.nat_params)

                # Create client and store
                client = Client(data=data, model=self.model, t=t, config=client_config)
                self.pseudo_clients.append(client)


        # should also keep persistent pseudo-server?
        # init internal pseudo-server
        server_config = {
                'max_iterations' : 1,
                'train_model' : False,
                'model_update_freq': 1,
                'dp_C' : self.config['dp_C'],
                'dp_sigma' : self.config['dp_sigma'],
                'enforce_pos_var' : self.config['enforce_pos_var'],
                'dp_mode' : 'param',
                "pbar" : self.config['pbar'], 
                }

        ChosenServer = SynchronousServer
        
        server = ChosenServer(model=self.model,
                                p=p,
                                init_q=q_old,
                                clients=self.pseudo_clients,
                                config=server_config)
            #############
        
        #print(p._std_params)

        # run training loop (just a single global update)
        while not server.should_stop():
            server.tick()

        # Dict for logging optimisation progress
        # for local PVI log just means (mean over all local training step)
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
            "logt" : [],
        }
        
        #if i % self.config["print_epochs"] == 0:
        '''
        logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                     f"LL: {epoch['ll']:.3f}, "
                     f"KL: {epoch['kl']:.3f}, "
                     f"log t: {epoch['logt']:.3f}, ")
                     #f"Epochs: {i}.")
        '''

        for k in training_curve:
            training_curve[k] = np.zeros(self.config['epochs'])
            for i_client, client in enumerate(self.pseudo_clients):
                training_curve[k] += np.array(client.log['training_curves'][-1][k])/len(self.pseudo_clients)
        self.log["training_curves"].append(training_curve)

        
        # Create non-trainable copy to send back to server
        q_new = server.q.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        # check if summing all pseudo_client t:s give same local factor, approximately yes
        '''
        for i_client, client in enumerate(self.pseudo_clients):
            if i_client == 0:
                cur_t = copy.deepcopy(client.t.nat_params)
                cur_t['np1'] *= self.config['damping_factor']
                cur_t['np2'] *= self.config['damping_factor']
            else:
                cur_t['np1'] += client.t.nat_params['np1'].detach().clone()*self.config['damping_factor']
                cur_t['np2'] += client.t.nat_params['np2'].detach().clone()*self.config['damping_factor']
        '''
        
        if self.t is not None:
            # Compute new local contribution from old distributions

            # NOTE: check how damping should be applied here!
            t_new = self.t.compute_refined_factor(
                q_new, q_old, damping=self.config['damping_factor'],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"])

            #print(cur_t['np2'])
            #print(t_new.nat_params['np2'])
            #print(f"nat1 diff: {torch.sum(torch.abs(cur_t['np1'] - t_new.nat_params['np1']))}")
            #print(f"nat2 diff: {torch.sum(torch.abs(cur_t['np2'] - t_new.nat_params['np2']))}")
            #sys.exit()

            # note for DP: only t is currently used by server:
            return q_new, t_new
        else:
            raise NotImplementedError('Local PVI only implemented with explicit t factors!')
            #logger.debug('Note: client not returning t')
            #return q_new, None




