
from abc import ABC
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
from .param_dp_client import Param_DP_Client
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


class Local_PVI_Client(Client):
    
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
            self.n_local_models = int(np.floor((self.data['y'].shape[-1])/config['batch_size']))
            print(f"data len={len(self.data['y'])}, b={config['batch_size']}, n_models={self.n_local_models}, would need data len {config['batch_size']*self.n_local_models}")
            #sys.exit()
        
        #self.optimiser_states = None
        
        # Set likelihood approximating term
        self.t = t # note: use internal partitions instead of single t for optimising, keep track of joint t with this
        
        self.log = defaultdict(list)
        self._can_update = True

        self.pseudo_clients = None
        self.pseudo_server = None
        self.batch_sizes = None

        # note: most tracking currently not working
        # actually tracks norm of change in params for LFA
        if self._config['track_client_norms']:
            self.pre_dp_norms = []
            self.post_dp_norms = []
            self.noise_norms = []


    def gradient_based_update(self, p, init_q=None, global_prior=None):
        # Cannot update during optimisation.
        self._can_update = False

        # Copy the approximate posterior, make old posterior non-trainable.
        '''
        q_old = p.non_trainable_copy()

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            # Initialise to prior.
            q = p.trainable_copy()
        '''
        
        # use different b for different models to use all data once: first models might have more data than last
        if self.batch_sizes is None:
            tmp1 = int(np.floor((self.data['y'].shape[-1])/self.n_local_models))
            if tmp1 == 0:
                raise ValueError('Using batch_size=0! Try increasing sampling frac!')
            tmp2 = len(self.data['y']) - tmp1*self.n_local_models
            batch_sizes = np.zeros(self.n_local_models, dtype=int) + tmp1
            batch_sizes[:tmp2] += 1
            self.batch_sizes = batch_sizes
        else:
            batch_sizes = self.batch_sizes
        #print(tmp1,tmp2)
        #print(np.unique(batch_sizes))
        #print(batch_sizes)
        #print(np.sum(batch_sizes), len(self.data['y']))
        #sys.exit()

        # create internal pseudo clients if don't exist
        if self.pseudo_clients is None:
            self.pseudo_clients = []
            cur_ind = 0
            for i_client, b in enumerate(batch_sizes):
                # Data of ith client
                #if i_client == 0:
                #    print(f"client data size: {len(self.data['y'])}, local model batch size: {b}")
                data = {'x' : self.data['x'][cur_ind:(cur_ind+b),:], 'y' : self.data['y'][cur_ind:(cur_ind+b)] }
                #print(f"client {i_client} should have {b} samples, actually has {data['y'].shape}")
                cur_ind += b

                client_config =  copy.deepcopy(self.config)
                client_config['dp_mode'] = 'param'
                client_config['dp_sigma'] = self.config['dp_sigma']/(np.sqrt(self.n_local_models)) # without secure aggregation
                #print(f"pseudo-client var: {client_config['dp_sigma']**2} will sum to {self.n_local_models*client_config['dp_sigma']**2}, should equal {self.config['dp_sigma']**2}")
                client_config['clients'] = self.n_local_models
                #client_config['batch_size'] = int(b) # note: here batch_size=data size for a given local model
                client_config['batch_size'] = int(np.ceil(self.config['pseudo_client_q']*b))

                t = MeanFieldGaussianFactor(nat_params = copy.deepcopy(self.t.nat_params))

                # Create client and store
                client = Param_DP_Client(data=data, model=self.model, t=t, config=client_config)
                self.pseudo_clients.append(client)

        # init internal pseudo-server
        if self.pseudo_server is None:
            server_config = {
                        'max_iterations' : self.config['n_global_updates'],
                        'train_model' : False,
                        'model_update_freq': 1,
                        'dp_C' : self.config['dp_C'], # no clipping on server, applied by pseudo-clients
                        'dp_sigma' : 0., # note: noise added by pseudo-clients
                        'enforce_pos_var' : self.config['enforce_pos_var'],
                        'dp_mode' : 'param',
                        "pbar" : self.config['pbar'], 
                        }

            ChosenServer = SynchronousServer
            self.pseudo_server = ChosenServer(model=self.model,
                                    p=p.non_trainable_copy(), # global prior=eff prior for given client?
                                    init_q=init_q.non_trainable_copy(), #q_old initial q distr=current model, note:if restarting will only ever use init_q!
                                    clients=self.pseudo_clients,
                                    config=server_config)

            # fix init_q problem with sequential global server:
            # local server uses init_q on 0th iteration, even when not actually 0th globally
            if init_q is None:
                server_config['max_iterations'] += 1
                self.pseudo_server.iterations = 1
                self.pseudo_server.q = p.non_trainable_copy()
                self.pseudo_server.p = global_prior.non_trainable_copy()
        else:
            # update pseudo-server model to match the updated global model
            self.pseudo_server.q = p.non_trainable_copy()
            
        pseudo_server = self.pseudo_server
        
        # run training loop (just a single update)
        pseudo_server.tick()

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
        #'''
        
        if self.config['track_client_norms']:
            tmp = np.zeros(3)
            for client in self.pseudo_clients:
                tmp[0] += client.pre_dp_norms[-1]
                tmp[1] += client.post_dp_norms[-1]
                tmp[2] += client.noise_norms[-1]
                #print(client.pre_dp_norms, client.noise_norms)
                #print(len(client.pre_dp_norms),len(client.noise_norms) )
                #sys.exit()
            self.pre_dp_norms.append(tmp[0]/len(self.pseudo_clients))
            self.post_dp_norms.append(tmp[1]/len(self.pseudo_clients))
            self.noise_norms.append(tmp[2]/len(self.pseudo_clients))


        # Create non-trainable copy to send back to server
        #q_new = pseudo_server.q.non_trainable_copy()

        #print(self.t.__dict__)
        #sys.exit()
        if self.t is not None:
            # Compute new local contribution from old distributions
            # note: possible damping is done on the pseudo-client level
            tmp = {}
            for k,v in self.t.nat_params.items():
                tmp[k] = torch.zeros_like(v)
                for i_client, client in enumerate(self.pseudo_clients):
                    tmp[k] += client.t.nat_params[k].detach().clone()
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

            # Create and return refined t of the same type
            # note: doesn't actually track log_coeff
            t_new = type(self.t)(nat_params=tmp, log_coeff=self.t.log_coeff, enforce_pos_var = self.t.enforce_pos_var)

            #print(cur_t['np2'])
            #print(t_new.nat_params['np2'])
            #print(f"nat1 diff: {torch.sum(torch.abs(cur_t['np1'] - t_new.nat_params['np1']))}")
            #print(f"nat2 diff: {torch.sum(torch.abs(cur_t['np2'] - t_new.nat_params['np2']))}")
            #sys.exit()

            # note for DP: only t is currently used by server:
            # NOTE: q_new doesn't do any damping!
            #return q_new, t_new
            return None, t_new
        else:
            raise NotImplementedError('Local PVI only implemented with explicit t factors!')
            #logger.debug('Note: client not returning t')
            #return q_new, None




