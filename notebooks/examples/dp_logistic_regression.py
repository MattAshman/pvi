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
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def main(args, rng_seed, dataset_folder):
    """
    Args:
        dataset_folder : (str) path to data containing x.npy and y.npy files for input and target
    """

    # enable progress bars
    pbar = args.pbar

    # do some args checks
    if args.dp_mode not in ['dpsgd', 'param','param_fixed','server','hfa']:
        raise ValueError(f"Unknown dp_mode: {args.dp_mode}")

    if args.model not in ['pvi', 'bcm_split', 'bcm_same', 'global_vi']:
        raise ValueError(f"Unknown model: {args.model}")

    
    logger.info(f"Starting {args.model} run with data folder: {dataset_folder}, dp_mode: {args.dp_mode}")

    if args.dp_mode in ['dpsgd','param_fixed']:#[seq','swor']:
        logger.info(f'Using SWOR sampling with batch size {args.batch_size}')
    elif args.dp_mode in ['hfa']:
        logger.info(f'Using sequential data passes with batch size {args.batch_size} (separate models for each batch)')
    else:
        logger.info(f'Using sequential data passes with batch size {args.batch_size}')


    # fix random seeds
    np.random.seed(rng_seed)
    torch.random.manual_seed(rng_seed)
    random.seed(rng_seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      torch.cuda.manual_seed(rng_seed)

    client_data, valid_set, N, prop_positive, full_data_split = standard_client_split(
            None, args.clients, args.data_bal_rho, args.data_bal_kappa, dataset_folder=dataset_folder
            )
    x_train, x_valid, y_train, y_valid = full_data_split

    #print(x_train.shape, x_valid.shape)
    #sys.exit()

    logger.info(f'Proportion of positive examples in each client: {np.array(prop_positive).round(2)}')
    logger.info(f'Total number of examples in each client: {N}')

    # might need some/all if optimising hyperparams?
    model_hyperparameters = {
        #"D"                        : x_train.shape[1]-10, # ei näytä vaikuttavan mitenkään
        #"optimiser"                : "Adam",
        #"optimiser_params"         : {"lr": 1e-3},
        #"epochs"                   : None,
        #"batch_size"               : None,#100,
        #"num_elbo_samples"         : 20,
        #"num_predictive_samples"   : 10, # only used when use_probit_approximation = False
    }
    model_config = {
            "use_probit_approximation" : False, 
            "num_predictive_samples"   : 100, # only used when use_probit_approximation = False
            "pbar" : pbar, 
            }

    model = LogisticRegressionModel(hyperparameters=model_hyperparameters, config=model_config)

    # note: for DP need to change to use actual sampling, no full data passes
    client_config = {
        'batch_size' : args.batch_size, # will run through entire data on each epoch using this batch size
        'batch_proc_size': args.batch_proc_size, # for DP-SGD and HFA
        'sampling_frac_q' : args.sampling_frac_q, # sampling fraction, only used with Poisson random sampling type
        'damping_factor' : args.damping_factor,
        'valid_factors' : False, # does this work at the moment? i guess not
        'epochs' : args.n_steps, # if sampling_type is 'seq': number of full passes through local data; if sampling_type is 'poisson' or 'swor': number of local SAMPLING steps, so not full passes
        'optimiser' : 'Adam',
        'optimiser_params' : {'lr' : args.learning_rate},
        'lr_scheduler' : 'MultiplicativeLR',
        'lr_scheduler_params' : { 'lr_lambda' : lambda epoch: 1.},
        'num_elbo_samples' : 10, # possible to break if this is low?
        'print_epochs' : 1, # ?
        'train_model' : False, # no need for having trainable model on client
        'update_log_coeff' : False, # no need for log coeff in t factors
        'dp_mode' : args.dp_mode, 
        'dp_C' : args.dp_C, # clipping constant
        'dp_sigma' : args.dp_sigma, # noise std
        'enforce_pos_var' : args.enforce_pos_var,
        'track_client_norms' : args.track_client_norms,
        'clients' : args.clients, # total number of clients
        "pbar" : pbar, 
    }
    # change batch_size for HFA
    #if args.dp_mode == 'hfa':
    #    client_config['batch_size'] = 1

    # prior params, use data dim+1 when assuming model adds extra bias dim
    prior_std_params = {
        "loc"   : torch.zeros(x_train.shape[1]+1),
        "scale" : torch.ones(x_train.shape[1]+1),
    }
    # initial t-factor params, not proper distributions, dims as above
    init_nat_params = {
        "np1" : torch.zeros(x_train.shape[1] + 1),
        "np2" : torch.zeros(x_train.shape[1] + 1),
    }

    # Initialise clients, q and server
    clients = set_up_clients(model, client_data, init_nat_params, client_config, args)

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
        server_config["train_model"] = False
        server_config["model_optimiser_params"] = {"lr": 1e-2}
        server_config["max_iterations"] = 1
        server_config["epochs"] = 1
        server_config["batch_size"] = 100
        server_config["optimiser"] = "Adam"
        server_config["optimiser_params"] = {"lr": 0.05}
        server_config["lr_scheduler"] = "MultiplicativeLR"
        server_config["lr_scheduler_params"] = {
            "lr_lambda": lambda epoch: 1.
        }
        server_config["num_elbo_samples"] = 10
        server_config["print_epochs"] = 1
        server_config["homogenous_split"] = True
        ChosenServer = GlobalVIServer

    elif args.model == 'pvi':
        ChosenServer = SynchronousServer

    server = ChosenServer(model=model,
                            p=q,
                            init_q=q,
                            clients=clients,
                            config=server_config)

    train_res = {}
    train_res['acc'] = np.zeros((args.n_global_updates))
    train_res['logl'] = np.zeros((args.n_global_updates))
    validation_res = {}
    validation_res['acc'] = np.zeros((args.n_global_updates))
    validation_res['logl'] = np.zeros((args.n_global_updates))
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
            
        # get global train and validation acc & logl
        train_acc, train_logl = acc_and_ll(server, torch.tensor(x_train).float(), torch.tensor(y_train).float())
        valid_acc, valid_logl = acc_and_ll(server, valid_set['x'], valid_set['y'])
        train_res['acc'][i_global] = train_acc
        train_res['logl'][ i_global] = train_logl
        validation_res['acc'][i_global] = valid_acc
        validation_res['logl'][i_global] = valid_logl

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


    if args.track_client_norms and args.plot_tracked:
        # separate script for hfa/dpsgd etc?

        if args.dp_mode == 'dpsgd':
            pre_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
            post_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
            for i_client, client in enumerate(clients):
                pre_dp_norms[i_client,:] = client.pre_dp_norms
                post_dp_norms[i_client,:] = client.post_dp_norms
            x1 = np.linspace(1,args.n_global_updates*args.n_steps, args.n_global_updates*args.n_steps)
            x2 = np.linspace(1,args.n_global_updates*args.n_steps, args.n_global_updates*args.n_steps)
        elif args.dp_mode == 'hfa':
            pre_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
            post_dp_norms = np.zeros((args.clients, args.n_global_updates))
            for i_client, client in enumerate(clients):
                pre_dp_norms[i_client,:] = np.concatenate([norms for norms in client.pre_dp_norms])
                post_dp_norms[i_client,:] = client.post_dp_norms
            x1 = np.linspace(1,args.n_global_updates*args.n_steps, args.n_global_updates*args.n_steps)
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
        #plt.savefig(figname)
        plt.show()
        
        #sys.exit()



    if args.track_params and args.plot_tracked:
        x = np.linspace(1,args.n_global_updates,args.n_global_updates)
        fig,axs = plt.subplots(2,figsize=(10,7))
        axs[0].plot(x, validation_res['acc'])
        axs[1].plot(x, validation_res['logl'])
        for i in range(2):
            axs[i].set_xlabel('Global updates')
        axs[0].set_ylabel('Model acc')
        axs[1].set_ylabel('Model logl')
        plt.suptitle("".format())
        figname = 'res_plots/param_traces/model_perf_global{}_local{}_C{}_sigma{}.pdf'.format(args.n_global_updates, args.n_steps, args.dp_C, args.dp_sigma)
        #plt.savefig(figname)
        plt.show()

        #sys.exit()
        x = np.linspace(0,args.n_global_updates,args.n_global_updates+1)
        fig,axs = plt.subplots(2,figsize=(10,7))
        axs[0].plot(x,param_trace1)
        axs[1].plot(x,param_trace2)
        for i in range(2):
            axs[i].set_xlabel('Global updates')
        axs[0].set_ylabel('Loc params')
        axs[1].set_ylabel('Scale params')
        figname = 'res_plots/param_traces/param_trace_global{}_local{}_C{}_sigma{}.pdf'.format(args.n_global_updates, args.n_steps, args.dp_C, args.dp_sigma)
        #plt.savefig(figname)
        plt.show()

    # compile possible tracked norms etc
    tracked = {}
    if args.track_client_norms:
        if args.dp_mode == 'dpsgd':
            pre_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
            post_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
            for i_client, client in enumerate(clients):
                pre_dp_norms[i_client,:] = client.pre_dp_norms
                post_dp_norms[i_client,:] = client.post_dp_norms
        elif args.dp_mode == 'hfa':
            pre_dp_norms = np.zeros((args.clients, args.n_global_updates * args.n_steps))
            post_dp_norms = np.zeros((args.clients, args.n_global_updates))
            for i_client, client in enumerate(clients):
                pre_dp_norms[i_client,:] = np.concatenate([norms for norms in client.pre_dp_norms])
                post_dp_norms[i_client,:] = client.post_dp_norms
        tracked['client_norms'] = {}
        tracked['client_norms']['pre_dp_norms'] = pre_dp_norms
        tracked['client_norms']['post_dp_norms'] = post_dp_norms

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
    parser.add_argument('--n_global_updates', default=2, type=int, help='number of global updates')
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=100, type=int, help="batch size; used if sampling type is 'swor' or 'seq'")
    parser.add_argument('--batch_proc_size', default=1, type=int, help="batch processing size; for DP-SGD or HFA, currently needs to be 1")
    parser.add_argument('--sampling_frac_q', default=.05, type=float, help="sampling fraction; only used if sampling_type is 'poisson' NOT IMPLEMENTED")
    parser.add_argument('--dp_sigma', default=1., type=float, help='DP noise magnitude')
    parser.add_argument('--dp_C', default=2., type=float, help='gradient norm bound')

    parser.add_argument('--folder', default='../../data/data/adult/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/abalone/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/mushroom/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/credit/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/bank/', type=str, help='path to combined train-test folder')
    #parser.add_argument('--folder', default='../../data/data/superconductor/', type=str, help='path to combined train-test folder')

    parser.add_argument('--clients', default=10, type=int, help='number of clients')
    parser.add_argument('--n_steps', default=10, type=int, help="when sampling type 'poisson' or 'swor': number of local training steps on each client update iteration; when sampling_type = 'seq': number of local epochs, i.e., full passes through local data on each client update iteration")
    parser.add_argument('-data_bal_rho', default=.0, type=float, help='data balance factor, in (0,1); 0=equal sizes, 1=small clients have no data')
    parser.add_argument('-data_bal_kappa', default=.0, type=float, help='minority class balance factor, 0=no effect')
    parser.add_argument('--damping_factor', default=.1, type=float, help='damping factor in (0,1], 1=no damping')
    parser.add_argument('--enforce_pos_var', default=False, action='store_true', help="enforce pos.var by taking abs values when convertingfrom natural parameters; NOTE: bit unclear if works at the moment!")
    
    parser.add_argument('--dp_mode', default='dpsgd', type=str, help="DP mode: 'dpsgd': DP-SGD, 'param': clip and noisify change in params, 'param_fixed': clip and noisify change in params using fixed minibatch for local training, 'server': clip and noisify change in params on (synchronous) server end, 'hfa': param DP with hierarchical fed avg. Sampling type is set based on the mode.")

    parser.add_argument('--track_params', default=False, action='store_true', help="track all params")
    parser.add_argument('--track_client_norms', default=False, action='store_true', help="track all (grad) norms pre & post DP")
    parser.add_argument('--plot_tracked', default=False, action='store_true', help="plot all tracked stuff after learning")
    parser.add_argument('--pbar', default=True, action='store_false', help="print tqdm progress bars")
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

