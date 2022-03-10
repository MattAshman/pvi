"""
Script for testing private PVI with logistic regression based on DP-SGD/suff.stats pert
"""

import argparse
import logging
import os
import random
import sys
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm.auto as tqdm
from sklearn.model_selection import KFold


#from mle_logging import MLELogger


module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from pvi.models.logistic_regression import LogisticRegressionModel
#from pvi.clients.synchronous_client import SynchronousClient
from pvi.clients import Client
from pvi.servers.sequential_server import SequentialServer
from pvi.servers.synchronous_server import SynchronousServer
from pvi.servers.bcm import BayesianCommitteeMachineSame
from pvi.servers.bcm import BayesianCommitteeMachineSplit
from pvi.servers.dpsgd_global_vi import GlobalVIServer


from pvi.distributions.exponential_family_distributions import MeanFieldGaussianDistribution
from pvi.distributions.exponential_family_factors import MeanFieldGaussianFactor

from utils import *

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
#handler.setLevel(logging.DEBUG)
handler.setLevel(logging.INFO)

logging.basicConfig(
    #level=logging.DEBUG, 
    level=logging.INFO, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[handler]
)



def main(args, rng_seed, dataset_folder):
    """
    Args:
        dataset_folder : (str) path to data containing x.npy and y.npy files for input and target
    """

    # disable/enable progress bars
    pbar = args.pbar

    # do some args checks
    if args.dp_mode not in ['nondp_batches', 'nondp_epochs','dpsgd', 'param','param_fixed','server','lfa', 'local_pvi']:
        raise ValueError(f"Unknown dp_mode: {args.dp_mode}")

    if args.model not in ['pvi', 'bcm_split', 'bcm_same', 'global_vi']:
        raise ValueError(f"Unknown model: {args.model}")

    
    logger.info(f"Starting {args.model} run with data folder: {dataset_folder}, dp_mode: {args.dp_mode}")

    if args.dp_mode in ['dpsgd','param_fixed']:#[seq','swor']:
        if args.sampling_frac_q is not None and args.batch_size is not None:
            logger.info(f'Using user-level SWOR sampling with sampling frac {args.sampling_frac_q} and fixed user data of size {args.batch_size}. Full batch is used when user is sampled.)')
        
        elif args.sampling_frac_q is not None:
            logger.info(f'Using SWOR sampling with sampling frac {args.sampling_frac_q}')
        elif args.batch_size is not None:
            logger.info(f'Using SWOR sampling with batch size {args.batch_size}')
        else:
            raise ValueError("Need to set at least one of 'batch_size', 'sampling_frac_q'!")

    elif args.dp_mode in ['lfa', 'local_pvi','param']:
        if args.batch_size is not None and args.sampling_frac_q is not None:
            raise ValueError("Exactly one of 'batch_size', 'sampling_frac_q' needs to be None")
        elif args.batch_size is None:
            logger.info(f'Using sequential data passes with local sampling frac {args.sampling_frac_q} (separate models for each batch)')
        elif args.sampling_frac_q is None:
            logger.info(f'Using sequential data passes with batch size {args.batch_size} (separate models for each batch)')
    else:
        if args.dp_mode in ['nondp_batches']:
            logger.info(f'Sampling {args.n_steps} batches per global update with batch size {args.batch_size}')
        elif args.dp_mode in ['nondp_epochs']: 
            logger.info(f'Sampling {args.n_steps} epochs per global update with batch size {args.batch_size}')
        else:
            raise ValueError(f"Unknown dp_mode: {args.dp_mode}")


    # fix random seeds
    np.random.seed(rng_seed)
    torch.random.manual_seed(rng_seed)
    random.seed(rng_seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      torch.cuda.manual_seed(rng_seed)

    #client_data, valid_set, N, prop_positive, full_data_split = standard_client_split(
    client_data, train_set, valid_set, N, prop_positive  = standard_client_split(
            None, args.clients, args.data_bal_rho, args.data_bal_kappa, dataset_folder=dataset_folder
            )
    #x_train, x_valid, y_train, y_valid = full_data_split
    x_train, x_valid, y_train, y_valid = train_set['x'], valid_set['x'], train_set['y'], valid_set['y']

    #print(x_train.shape, x_valid.shape)
    #sys.exit()

    logger.info(f'Proportion of positive examples in each client: {np.array(prop_positive).round(2)}')
    logger.info(f'Total number of examples in each client: {N}')

    # might need some/all if optimising hyperparams? in which case need to check values
    model_hyperparameters = {
    }
    model_config = {
            "use_probit_approximation" : False, 
            "num_predictive_samples"   : 100, # only used when use_probit_approximation = False
            "pbar" : pbar, 
            }

    model = LogisticRegressionModel(hyperparameters=model_hyperparameters, config=model_config)

    # define multistep lr scheduler bounds
    if args.use_lr_scheduler and args.n_global_updates > 5:
        # reduce by 1/2 at the middle and again 2 global updates before end
        lr_scheduler_params  = {'gamma' : .5, 
                            'milestones': [args.n_global_updates//2, args.n_global_updates-2]}
        #'verbose' : False} # note: pytorch 1.6 seems to have bug: doesn't accept verbose keyword
        logger.debug(f"Using multistep lr scheduler with milestones: {lr_scheduler_params['milestones']}")
    else:
        args.use_lr_scheduler = False
        if args.use_lr_scheduler:
            logger.debug(f"Disabling multistep lr scheduler due to having < 6 global updates!")
        lr_scheduler_params = {}

    client_config = {
        'batch_size' : args.batch_size, # will run through entire data on each epoch using this batch size
        'batch_proc_size': args.batch_proc_size, # for DP-SGD and LFA
        'sampling_frac_q' : args.sampling_frac_q, # sampling fraction
        'pseudo_client_q' : args.pseudo_client_q, # sampling frac for local_pvi pseudo-clients
        'damping_factor' : args.damping_factor,
        'valid_factors' : False, # does this work at the moment? i guess not
        'epochs' : args.n_steps, # if sampling_type is 'seq': number of full passes through local data; if sampling_type is 'poisson' or 'swor': number of local SAMPLING steps, so not full passes
        'optimiser' : 'Adam',
        'optimiser_params' : {'lr' : args.learning_rate},
        'lr_scheduler' : 'MultiStepLR',
        'lr_scheduler_params' : lr_scheduler_params,
        'use_lr_scheduler' : args.use_lr_scheduler,
        'num_elbo_samples' : 10, # possible to break if this is low?
        'print_epochs' : 1, # ?
        'train_model' : False, # no need for having trainable model on client
        'update_log_coeff' : False, # no need for log coeff in t factors
        'dp_mode' : args.dp_mode, 
        'dp_C' : args.dp_C, # clipping constant
        'dp_sigma' : args.dp_sigma, # noise std
        'pre_clip_sigma' : args.pre_clip_sigma,
        'enforce_pos_var' : args.enforce_pos_var,
        'track_client_norms' : args.track_client_norms,
        'clients' : args.clients, # total number of clients
        "pbar" : pbar, 
        'noisify_np': True, # for param DP and related dp modes: if True clip and noisify natural parameters, otherwise use unconstrained loc-scale. No effect on DPSGD.
        "freeze_var_updates" : args.freeze_var_updates,
        "n_global_updates" : args.n_global_updates,
    }
    # change batch_size for LFA
    #if args.dp_mode == 'lfa':
    #    client_config['batch_size'] = 1
    #if args.dp_mode == 'dpsgd':
    #    client_config['batch_size'] = None

    # prior params, use data dim+1 when assuming model adds extra bias dim
    prior_std_params = {
        "loc"   : torch.zeros(x_train.shape[1]+1),
        "scale" : torch.ones(x_train.shape[1]+1),
    }
    # these used as initial t-factor params, should match prior, dims as above
    init_nat_params = {
        "np1" : torch.zeros(x_train.shape[1] + 1),
        "np2" : torch.zeros(x_train.shape[1] + 1),
    }

    # Initialise clients, q and server
    clients = set_up_clients(model, client_data, init_nat_params, client_config, dp_mode=args.dp_mode, batch_size=args.batch_size, sampling_frac_q=args.sampling_frac_q)

    #print(clients[0])
    #sys.exit()

    q = MeanFieldGaussianDistribution(std_params=prior_std_params,
                                      is_trainable=False, enforce_pos_var=args.enforce_pos_var)
    server_config = {
            'max_iterations' : args.n_global_updates,
            'train_model' : False, # need False here?
            'model_update_freq': 1,
            #'hyper_optimiser': 'SGD',
            #'hyper_optimiser_params': {'lr': 1},
            #'hyper_updates': 1,
            #'server_add_dp' : args.server_add_dp,
            'dp_C' : args.dp_C,
            'dp_sigma' : args.dp_sigma,
            'pre_clip_sigma' : args.pre_clip_sigma,
            'enforce_pos_var' : args.enforce_pos_var,
            'dp_mode' : args.dp_mode,
            "pbar" : pbar, 
            }

    # try using initial q also as prior here
    if args.model in ['bcm_same','bcm_split']:
        server_config['max_iterations'] = 1
        if args.model == 'bcm_same':
            ChosenServer = BayesianCommitteeMachineSame
        else:
            ChosenServer = BayesianCommitteeMachineSplit

    elif args.model == 'global_vi':
        #server_config["train_model"] = False
        #server_config["model_optimiser_params"] = {"lr": 1e-2}
        #server_config["max_iterations"] = 20 # NOTE: need to fix n_global_steps vs epochs vs max_iterations
        #server_config['max_iterations'] = args.n_global_updates
        server_config["epochs"] = args.n_steps # number of local steps for dpsgd, should match n_steps
        server_config["batch_size"] = None
        server_config["sampling_frac_q"] = args.sampling_frac_q
        server_config["optimiser"] = "Adam"
        server_config["optimiser_params"] = {'lr' : args.learning_rate}
        server_config["lr_scheduler"] = "MultiplicativeLR"
        server_config["lr_scheduler_params"] = {
            "lr_lambda": lambda epoch: 1.
        }
        server_config["num_elbo_samples"] = 100
        server_config["print_epochs"] = 1
        server_config["homogenous_split"] = True
        server_config['track_client_norms'] = args.track_client_norms
        ChosenServer = GlobalVIServer
        args.clients = 1

    elif args.model == 'pvi':
        if args.server == 'synchronous':
            ChosenServer = SynchronousServer
        elif args.server == 'sequential':
            ChosenServer = SequentialServer
        else:
            raise ValueError(f'Unknown server type: {args.server}')


    server = ChosenServer(model=model,
                            p=q,
                            init_q=q,
                            clients=clients,
                            config=server_config)

    train_res = {}
    train_res['acc'] = np.zeros((args.n_global_updates))
    train_res['logl'] = np.zeros((args.n_global_updates))
    train_res['posneg'] = []
    validation_res = {}
    validation_res['acc'] = np.zeros((args.n_global_updates))
    validation_res['logl'] = np.zeros((args.n_global_updates))
    validation_res['posneg'] = []
    client_train_res = {}
    client_train_res['elbo'] = np.zeros((args.clients, args.n_global_updates, args.n_steps))
    client_train_res['logl'] = np.zeros((args.clients, args.n_global_updates, args.n_steps))
    client_train_res['kl'] = np.zeros((args.clients, args.n_global_updates, args.n_steps))


    ################### param tracking
    #track_params = True
    try:
        args.track_params
    except:
        args.track_params = False

    if args.track_params:
        logger.warning('tracking all parameter histories, this might be costly!')
        
        #print(server.q.__dict__)
        #print(server.q._nat_params)
        #print(server.q._std_from_nat())
        #sys.exit()
        #def _std_from_nat(cls, nat_params):
        #np1 = nat_params["np1"]
        #np2 = nat_params["np2"]

        # note: after training get natural params
        param_trace1 = np.zeros((args.n_global_updates+1, len(server.q._std_params['loc']))) 
        param_trace2 = np.zeros((args.n_global_updates+1, len(server.q._std_params['scale'])))
        param_trace1[0,:] = server.q._std_params['loc'].detach().numpy()
        param_trace2[0,:] = server.q._std_params['scale'].detach().numpy()

    
    #print(server.__dict__)
    #print(server.q.__dict__)
    # plot server parameters
    #print(server.q._std_params['loc'])
    #plt.plot
    #sys.exit()

    
    i_global = 0
    logger.info('Starting model training')
    while not server.should_stop():

        # run training loop
        server.tick()

        if args.model != 'global_vi':

            # get client training curves
            for i_client in range(args.clients):
                client_train_res['elbo'][i_client,i_global,:] = server.get_compiled_log()[f'client_{i_client}']['training_curves'][server.iterations-1]['elbo']
                client_train_res['logl'][i_client,i_global,:] = server.get_compiled_log()[f'client_{i_client}']['training_curves'][server.iterations-1]['ll']
                client_train_res['kl'][i_client,i_global,:] = server.get_compiled_log()[f'client_{i_client}']['training_curves'][server.iterations-1]['kl']
            
        # get global train and validation acc & logl, assume to be tensors here
        #train_acc, train_logl = acc_and_ll(server, torch.tensor(x_train).float(), torch.tensor(y_train).float())
        train_acc, train_logl, train_posneg = acc_and_ll(server, x_train, y_train)
        valid_acc, valid_logl, valid_posneg = acc_and_ll(server, valid_set['x'], valid_set['y'])

        train_res['acc'][i_global] = train_acc
        train_res['logl'][ i_global] = train_logl
        train_res['posneg'].append(train_posneg)
        validation_res['acc'][i_global] = valid_acc
        validation_res['logl'][i_global] = valid_logl
        validation_res['posneg'].append(valid_posneg)


        ### mle testing:
        '''
        time_tic = {'global_steps': 1, 'local_steps': args.n_steps}
        stats_tic = {'global_train_logl': train_logl, 'global_test_logl': valid_logl, 'local_train_loss': -1}

        # Update the log with collected data & save it to .hdf5
        log.update(time_tic, stats_tic)
        '''
        ###############


        # param tracking
        if args.track_params:
            #print(server.__dict__,'\n')
            #print(server.q.__dict__,'\n')
            #print(server.p.__dict__,'\n') this is just priors?

            tmp = server.q._std_from_nat(server.q._nat_params)
            param_trace1[i_global+1, :] = tmp['loc'].detach().numpy()
            param_trace2[i_global+1, :] = tmp['scale'].detach().numpy()
            #print(server.q._nat_params)
            #print(server.q._std_from_nat())
            #sys.exit()

        
        print(f'Train: accuracy {train_acc:.3f}, mean-loglik {train_logl:.3f}\n'
              f'Valid: accuracy {valid_acc:.3f}, mean-loglik {valid_logl:.3f}\n')

        i_global += 1

    #print(server.__dict__)
    #sys.exit()

    ### mle testing
    #log.save()
    ###############

    if args.track_client_norms and args.plot_tracked:
        # separate script for lfa/dpsgd etc?
        if args.dp_mode == 'dpsgd':
            if args.model == 'global_vi':
                pre_dp_norms = np.zeros((1, args.n_global_updates * args.n_steps))
                post_dp_norms = np.zeros((1, args.n_global_updates * args.n_steps))
                pre_dp_norms[0,:] = server.pre_dp_norms
                post_dp_norms[0,:] = server.post_dp_norms
            else:
                pre_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
                post_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
                for i_client, client in enumerate(clients):
                    pre_dp_norms[i_client,:] = client.pre_dp_norms
                    post_dp_norms[i_client,:] = client.post_dp_norms
            x1 = np.linspace(1,args.n_global_updates*args.n_steps, args.n_global_updates*args.n_steps)
            x2 = np.linspace(1,args.n_global_updates*args.n_steps, args.n_global_updates*args.n_steps)
        elif args.dp_mode in ['lfa']:
            pre_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
            post_dp_norms = np.zeros((args.clients, args.n_global_updates))
            for i_client, client in enumerate(clients):
                pre_dp_norms[i_client,:] = np.concatenate([norms for norms in client.pre_dp_norms])
                post_dp_norms[i_client,:] = client.post_dp_norms
            x1 = np.linspace(1,args.n_global_updates*args.n_steps, args.n_global_updates*args.n_steps)
            x2 = np.linspace(1,args.n_global_updates, args.n_global_updates)
        elif args.dp_mode in ['local_pvi']:
            pre_dp_norms = np.zeros((args.clients, args.n_global_updates))
            post_dp_norms = np.zeros((args.clients, args.n_global_updates))
            for i_client, client in enumerate(clients):
                pre_dp_norms[i_client,:] =  client.pre_dp_norms
                post_dp_norms[i_client,:] = client.post_dp_norms
            x1 = np.linspace(1,args.n_global_updates, args.n_global_updates)
            x2 = np.linspace(1,args.n_global_updates, args.n_global_updates)


        fig,axs = plt.subplots(2,figsize=(10,7))
        for i_client in range(args.clients):
            axs[0].plot(x1, pre_dp_norms[i_client,:], alpha=.3)
            axs[1].plot(x2, post_dp_norms[i_client,:], alpha=.3)
        axs[0].plot(x1, pre_dp_norms.mean(0), alpha=.8, color='black')
        axs[1].plot(x2, post_dp_norms.mean(0), alpha=.8, color='black')

        for i in range(2):
            axs[i].set_xlabel('Local step')
        axs[0].set_ylabel('Pre DP client norm')
        axs[1].set_ylabel('Post DP client norm')


        figname = 'res_plots/client_norm_traces/client_norms_{}_global{}_local{}_C{}_sigma{}.pdf'.format(args.dp_mode,args.n_global_updates, args.n_steps, args.dp_C, args.dp_sigma)
        plt.tight_layout()
        #plt.savefig(figname)
        plt.show()
        
        #sys.exit()



    if args.track_params and args.plot_tracked:
        # save tracked params
        if False:
            filename = f"res_plots/param_traces/saved_params/saved_params_{args.dp_mode}_globals{args.n_global_updates}_steps{args.n_steps}_clients{args.clients}"
            #print(param_trace1.shape, param_trace2.shape) # eli nämä on globals+1 x loc /scaleparam shape
            np.savez(filename, loc_params=param_trace1, scale_params=param_trace2)
            
            #sys.exit()


        # plot distance from init
        if False:
            x = np.linspace(1,args.n_global_updates,args.n_global_updates)
            y = [ np.sqrt( \
                np.linalg.norm(param_trace1[0,:]-param_trace1[i+1,:],ord=2)**2 \
                + np.linalg.norm(param_trace2[0,:]-param_trace2[i+1,:],ord=2)**2 ) \
                for i in range(args.n_global_updates)]
            fig,axs = plt.subplots(2,figsize=(10,7))
            axs[0].plot(x,y)
            axs[1].plot(x, validation_res['logl'])
            axs[1].set_xlabel('Global update')
            axs[0].set_ylabel('l2 distance from init')
            axs[1].set_ylabel('Model logl')
            figname = 'res_plots/param_traces/param_dist_clients{}_global{}_local{}_C{}_sigma{}.pdf'.format(args.clients,args.n_global_updates, args.n_steps, args.dp_C, args.dp_sigma)
            plt.tight_layout()
            #plt.savefig(figname)
            #plt.close()
            plt.show()
            #sys.exit('break ok')

        # model acc + logl plot
        if False:
            x = np.linspace(1,args.n_global_updates,args.n_global_updates)
            fig,axs = plt.subplots(2,figsize=(10,7))
            axs[0].plot(x, validation_res['acc'])
            axs[1].plot(x, validation_res['logl'])
            for i in range(2):
                axs[i].set_xlabel('Global updates')
            axs[0].set_ylabel('Model acc')
            axs[1].set_ylabel('Model logl')
            plt.suptitle("".format())
            figname = 'res_plots/param_traces/model_perf_clients{}_global{}_local{}_C{}_sigma{}.pdf'.format(args.clients,args.n_global_updates, args.n_steps, args.dp_C, args.dp_sigma)
            plt.tight_layout()
            #plt.savefig(figname)
            #plt.close()
            plt.show()

        #sys.exit()
        # param trace plot over training
        if True:
            x = np.linspace(0,args.n_global_updates,args.n_global_updates+1)
            fig,axs = plt.subplots(2,figsize=(10,7))
            axs[0].plot(x,param_trace1)
            axs[1].plot(x,param_trace2)
            for i in range(2):
                axs[i].set_xlabel('Global updates')
            axs[0].set_ylabel('Loc params')
            axs[1].set_ylabel('Scale params')
            axs[0].grid()
            axs[1].grid()
            figname = 'res_plots/param_traces/param_trace_{}_clients{}_global{}_local{}_C{}_sigma{}.pdf'.format(args.dp_mode,args.clients,args.n_global_updates, args.n_steps, args.dp_C, args.dp_sigma)
            plt.tight_layout()
            #plt.savefig(figname)
            #plt.close()
            plt.show()

    # compile possible tracked norms etc
    tracked = {}
    if args.track_client_norms:
        if args.dp_mode == 'dpsgd':
            if args.model == 'global_vi':
                pre_dp_norms = np.zeros((1, args.n_global_updates * args.n_steps))
                post_dp_norms = np.zeros((1, args.n_global_updates * args.n_steps))
                pre_dp_norms[0,:] = server.pre_dp_norms
                post_dp_norms[0,:] = server.post_dp_norms
            else:
                pre_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
                post_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
                for i_client, client in enumerate(clients):
                    pre_dp_norms[i_client,:] = client.pre_dp_norms
                    post_dp_norms[i_client,:] = client.post_dp_norms
        elif args.dp_mode in ['lfa']:
            pre_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
            post_dp_norms = np.zeros((args.clients, args.n_global_updates))
            noise_norms = np.zeros((args.clients, args.n_global_updates))
            for i_client, client in enumerate(clients):
                pre_dp_norms[i_client,:] = np.concatenate([norms for norms in client.pre_dp_norms])
                post_dp_norms[i_client,:] = client.post_dp_norms
                noise_norms[i_client,:] = client.noise_norms

        elif args.dp_mode == 'local_pvi':
            pre_dp_norms = np.zeros((args.clients, args.n_global_updates))
            post_dp_norms = np.zeros((args.clients, args.n_global_updates))
            noise_norms = np.zeros((args.clients, args.n_global_updates))
            for i_client, client in enumerate(clients):
                pre_dp_norms[i_client,:] = client.pre_dp_norms
                #post_dp_norms[i_client,:] = client.post_dp_norms
                noise_norms[i_client,:] = client.noise_norms

        tracked['client_norms'] = {}
        tracked['client_norms']['pre_dp_norms'] = pre_dp_norms
        tracked['client_norms']['post_dp_norms'] = post_dp_norms
        try:
            tracked['client_norms']['noise_norms'] = noise_norms
        except:
            pass

    # some tracked norm plotting for local PVI
    if args.track_client_norms and args.plot_tracked:
        fix, axs = plt.subplots(1,3)
        for i_client, client in enumerate(clients):
            axs[0].plot(tracked['client_norms']['pre_dp_norms'][i_client,:])
            axs[1].plot(tracked['client_norms']['noise_norms'][i_client,:]/tracked['client_norms']['pre_dp_norms'][i_client,:])
        #axs[2].plot(np.sum(tracked['client_norms']['noise_norms'],0)/server.param_update_norms)
        axs[2].plot(server.param_update_norms)
        for i in range(3):
            axs[i].set_xlabel('Global updates')
        axs[0].set_ylabel('Pre DP norm')
        axs[1].set_ylabel('Relative effect of noise')
        #axs[2].set_ylabel('Relative noise on global update norms')
        axs[2].set_ylabel('Global update norm')
        figname = 'res_plots/client_norm_traces/relative_noise_effect_clients{}_global{}_local{}_C{}_sigma{}.pdf'.format(args.clients,args.n_global_updates, args.n_steps, args.dp_C, args.dp_sigma)
        plt.tight_layout()
        #plt.savefig(figname)
        #plt.close()
        plt.show()
        #plt.plot(client.noise_norms)
        #plt.show()

    #sys.exit()
    

    return validation_res, train_res, client_train_res, prop_positive, tracked



def plot_global_curves(res, measure):
    
    fig,axs = plt.subplots(2,1, figsize=(8,10))
    axs[0].plot(res['acc'])
    axs[0].set_title(f'Global model results, {measure}')
    axs[0].set_ylabel('Acc')
    axs[0].set_xlabel('Global updates')
    axs[0].grid()
    axs[1].plot(res['logl'])
    axs[1].set_ylabel('Logl')
    axs[1].set_xlabel('Global updates')
    axs[1].grid()
    plt.show()



def plot_training_curves(client_train_res, clients):
    """Plotter for training curves
    """
    colors = plt.rcParams['axes.prop_cycle']
    
    fig, axs = plt.subplots(3,1, figsize=(6,10))
    measures = ['elbo', 'logl', 'kl']
    labels = ['ELBO','logl','KL']
    for i in range(3):
        axs[i].set_prop_cycle(colors)
        for i_client in range(clients):
            axs[i].plot( client_train_res[measures[i]][i_client,:,:].reshape(-1), label=f'client {i_client}')
        axs[i].legend()
        axs[i].set_ylabel(labels[i])
        axs[i].set_xlabel('Training iteration')
        axs[i].grid()
    axs[0].set_title(f'Full training curves with {args.n_global_updates} global updates, {args.n_steps} local steps')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--model', default='pvi', type=str, help="Which model to use: \'pvi\', \'bcm_same\', \'bcm_split\', or \'global_vi\'")
    parser.add_argument('--server', default='synchronous', type=str, help="Which server to use: \'synchronous\', or \'sequential\'")
    parser.add_argument('--n_global_updates', default=1, type=int, help='number of global updates')
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--use_lr_scheduler', default=False, action='store_true', help="use multistep lr scheduler, Actual schedule defined in this script when setting up clients")

    parser.add_argument('--batch_size', default=None, type=int, help="batch size; can use if dp_mode not 'dpsgd'")
    parser.add_argument('--batch_proc_size', default=1, type=int, help="batch processing size; for DP-SGD or LFA, currently needs to be 1")
    parser.add_argument('--sampling_frac_q', default=None, type=float, help="sampling fraction, local batch_sizes in dpsgd or lfa are set based on this")
    parser.add_argument('--pseudo_client_q', default=1., type=float, help="sampling fraction used by pseudo-clients in local pvi")
    parser.add_argument('--pre_clip_sigma', default=0., type=float, help='noise magnitude for noise added before clipping (biass mitigation)')
    parser.add_argument('--dp_sigma', default=0., type=float, help='DP noise magnitude')
    parser.add_argument('--dp_C', default=100., type=float, help='gradient norm bound')
    #parser.add_argument('--folder', default='../../data/data/MNIST/', type=str, help='path to combined train-test folder')

    parser.add_argument('--folder', default='../../data/data/adult/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/abalone/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/mushroom/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/credit/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/bank/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/superconductor/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/mimic3/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default=None, type=str, help='path to combined train-test folder')
    parser.add_argument('--freeze_var_updates', default=0, type=int, help='Freeze var params for first given number of global updates')

    parser.add_argument('--clients', default=10, type=int, help='number of clients')
    parser.add_argument('--n_steps', default=10, type=int, help="when sampling type 'poisson' or 'swor': number of local training steps on each client update iteration; when sampling_type = 'seq': number of local epochs, i.e., full passes through local data on each client update iteration")
    parser.add_argument('-data_bal_rho', default=.0, type=float, help='data balance factor, in (0,1); 0=equal sizes, 1=small clients have no data')
    parser.add_argument('-data_bal_kappa', default=.0, type=float, help='minority class balance factor, 0=no effect')
    parser.add_argument('--damping_factor', default=.1, type=float, help='damping factor in (0,1], 1=no damping')
    parser.add_argument('--enforce_pos_var', default=False, action='store_true', help="enforce pos.var by taking abs values when convertingfrom natural parameters; NOTE: bit unclear if works at the moment!")
    
    parser.add_argument('--dp_mode', default='dpsgd', type=str, help="DP mode: 'nondp_epochs': no clipping or noise, do n_steps epochs per global update, 'nondp_batches': no clipping or noise, do n_steps batches per global update, 'dpsgd': DP-SGD, 'param': clip and noisify change in params, 'param_fixed': clip and noisify change in params using fixed minibatch for local training, 'lfa': param DP with hierarchical fed avg., 'local_pvi': partition local data to additional t-factors, add noise as param DP. Sampling type is set based on the mode. Additionally: 'lfa_dpsgd' and 'pvi_dpsgd' run lfa/local_pvi for the first global updates, then change to dpsgd.")

    parser.add_argument('--track_params', default=False, action='store_true', help="track all params")
    parser.add_argument('--track_client_norms', default=False, action='store_true', help="track all (grad) norms pre & post DP")
    parser.add_argument('--plot_tracked', default=False, action='store_true', help="plot all tracked stuff after learning")
    parser.add_argument('--pbar', default=True, action='store_false', help="disable tqdm progress bars")
    args = parser.parse_args()

    main(args, rng_seed=2303, dataset_folder=args.folder)

    '''
    abalone dataset NOT WORKING FOR SOME REASON
    Input  shape: (4177, 10)
    Output shape: (4177, 1)
    adult dataset non-dp ~= .85
    Input  shape: (48842, 108)
    Output shape: (48842, 1)
    mushroom dataset non-dp ~= .99
    Input  shape: (8124, 111)
    Output shape: (8124, 1)
    credit dataset non-dp ~=.89
    Input  shape: (653, 46)
    Output shape: (653, 1)
    bank dataset, non-dp ~= .9
    Input  shape: (45211, 51)
    Output shape: (45211, 1)
    superconductor dataset NOT WORKING FOR SOME REASON
    Input  shape: (21263, 81)
    Output shape: (21263, 1)
    '''

