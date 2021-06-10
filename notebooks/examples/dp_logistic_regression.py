"""
Script for testing private PVI with logistic regression based on DP-SGD/suff.stats pert
"""

import argparse
import logging
import os
import sys

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

    logger.info(f"Starting PVI run with data folder: {dataset_folder}")
    if args.sampling_type in ['seq','swor']:
        logger.info(f'Using {args.sampling_type} with sampling batch size {args.batch_size}')
    elif args.sampling_type == 'poisson':
        logger.info(f'Using {args.sampling_type} with sampling fraction {args.sampling_frac_q}')

    np.random.seed(rng_seed)

    client_data, valid_set, N, prop_positive, full_data_split = standard_client_split(
            None, args.clients, args.data_bal_rho, args.data_bal_kappa, dataset_folder=dataset_folder
            )
    x_train, x_valid, y_train, y_valid = full_data_split

    logger.info(f'Proportion of positive examples in each client: {np.array(prop_positive).round(2)}')
    logger.info(f'Total number of examples in each client: {N}')

    # might need some/all if optimising hyperparams
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
            }

    model = LogisticRegressionModel(hyperparameters=model_hyperparameters, config=model_config)

    # note: for DP need to change to use actual sampling, no full data passes
    client_config = {
        'batch_size' : args.batch_size, # will run through entire data on each epoch using this batch size
        'batch_proc_size': args.batch_proc_size, # for DP-SGD
        'sampling_frac_q' : args.sampling_frac_q, # sampling fraction, only used with Poisson random sampling type
        'damping_factor' : args.damping_factor,
        'valid_factors' : False, # does this work at the moment?
        'epochs' : args.n_steps, # if sampling_type is 'seq': number of full passes through local data; if sampling_type is 'poisson' or 'swor': number of local SAMPLING steps, so not full passes
        'optimiser' : 'Adam',
        'optimiser_params' : {'lr' : args.learning_rate},
        'lr_scheduler' : 'MultiplicativeLR',
        'lr_scheduler_params' : { 'lr_lambda' : lambda epoch: 1.},
        'num_elbo_samples' : 10, # possible to break if this is low?
        'print_epochs' : 1, # ?
        'train_model' : False, # no need for having trainable model on client
        'update_log_coeff' : False, # no need for log coeff in t factors
        'sampling_type' : args.sampling_type, # sampling type for clients:'seq' to sequentially sample full local data, 'poisson' for Poisson sampling with fraction q, 'SWOR' for sampling size b batch without replacement. For DP, need either Poisson or SWOR
        'use_dpsgd' : args.use_dpsgd, # if True use DP-SGD for privacy, otherwise clip & noisyfy parameters directly
        'dp_C' : args.dp_C, # clipping constant
        'dp_sigma' : args.dp_sigma, # noise std
    }
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

    q = MeanFieldGaussianDistribution(std_params=prior_std_params,
                                      is_trainable=False)
    server_config = {
            'max_iterations' : args.n_global_updates,
            'train_model' : False, # need False here?
            'model_update_freq': 1,
            #'hyper_optimiser': 'SGD',
            #'hyper_optimiser_params': {'lr': 1},
            #'hyper_updates': 1,
            }

    # try using initial q also as prior here
    server = SequentialServer(model=model,
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

    i_global = 0
    logger.info('Starting model training')
    while not server.should_stop():

        # run training loop
        server.tick()

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

        
        print(f'Train: accuracy {train_acc:.3f}, mean-loglik {train_logl:.3f}\n'
              f'Valid: accuracy {valid_acc:.3f}, mean-loglik {valid_logl:.3f}\n')

        i_global += 1


    # plot all training curves
    # NOTE: seems like elbo = logl, why? check later with longer runs
    #plot_training_curves(client_train_res, args.clients)

    #plot_global_curves(validation_res, 'validation set')
    #print(f'final validation res:\n{validation_res}')
    return validation_res, train_res, client_train_res, prop_positive



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
    parser.add_argument('--n_global_updates', default=1, type=int, help='number of global updates')
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='learning rate')
    parser.add_argument('-batch_size', default=200, type=int, help="batch size; used if sampling_type is 'swor' or 'seq'")
    parser.add_argument('--batch_proc_size', default=1, type=int, help="batch processing size; for DP-SGD, currently needs to be 1")
    parser.add_argument('--sampling_frac_q', default=.05, type=float, help="sampling fraction; only used if sampling_type is 'poisson'")
    parser.add_argument('--dp_sigma', default=1.0, type=float, help='DP noise magnitude')
    parser.add_argument('--dp_C', default=1., type=float, help='gradient norm bound')

    parser.add_argument('--folder', default='data/adult/', type=str, help='path to combined train-test adult data folder')

    parser.add_argument('--clients', default=4, type=int, help='number of clients')
    parser.add_argument('--n_steps', default=5, type=int, help="when sampling_type 'poisson' or 'swor': number of local training steps on each client update iteration; when sampling_type = 'seq': number of local epochs, i.e., full passes throuhg local data on each client update iteration")
    parser.add_argument('-data_bal_rho', default=.0, type=float, help='data balance factor, in (0,1); 0=equal sizes, 1=small clients have no data')
    parser.add_argument('-data_bal_kappa', default=.0, type=float, help='minority class balance factor, 0=no effect')
    parser.add_argument('--damping_factor', default=1., type=float, help='damping factor in (0,1], 1=no damping')
    parser.add_argument('--sampling_type', default='swor', type=str, help="sampling type for clients:'seq' to sequentially sample full local data, 'poisson' for Poisson sampling with fraction q, 'swor' for sampling without replacement. For DP, need either Poisson or SWOR")
    parser.add_argument('--use_dpsgd', default=False, type=bool, help="use dp-sgd or clip & noisify parameters")


    args = parser.parse_args()

    main(args, rng_seed=2303, dataset_folder='../../data/data/adult/')



