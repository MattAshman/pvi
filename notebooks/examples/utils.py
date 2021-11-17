
import logging
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import torch
from tqdm import tqdm

import fourier_accountant

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from pvi.models.logistic_regression import LogisticRegressionModel
#from pvi.clients.synchronous_client import SynchronousClient
from pvi.clients import Client
from pvi.clients import LFAClient
from pvi.servers.sequential_server import SequentialServer
from pvi.distributions.exponential_family_distributions import MeanFieldGaussianDistribution
from pvi.distributions.exponential_family_factors import MeanFieldGaussianFactor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def set_up_clients(model, client_data, init_nat_params, config, args):

    clients = []
    expected_batch = []
    # Create clients
    for i,_client_data in enumerate(client_data):
        # Data of ith client
        data = {k : torch.tensor(v).float() for k, v in _client_data.items()}
        #logger.debug(f"Client {i} data {data['x'].shape}, {data['y'].shape}")
        #if args.sampling_type == 'poisson':
        #    expected_batch.append(args.sampling_frac_q*data['x'].shape[0])

        # Approximating likelihood term of ith client
        t = MeanFieldGaussianFactor(nat_params=init_nat_params)

        # Create client and store
        if args.dp_mode == 'lfa':
            client = LFAClient(data=data, model=model, t=t, config=config)
        else:
            client = Client(data=data, model=model, t=t, config=config)
        clients.append(client)

    #if args.sampling_type == 'poisson':
    #    logger.info(f'Expected batch sizes: {expected_batch}')
        
    return clients



def standard_client_split(dataset_seed, num_clients, client_size_factor, class_balance_factor, total_splits=2, k_split=0, dataset_folder='../../data/data/adult/'):
    """
    Args:
        dataset_seed : seed for client data splitting, None to avoid fixing separate dataset seed
        total_split : k for k-fold train-validation split
        k_split : which data split to use
    """

    # Get data split
    full_data_split = get_nth_split(total_splits, k_split, dataset_folder)
    x_train, x_valid, y_train, y_valid = full_data_split

    #logger.debug(f'shapes, x_train: {x_train.shape}, y_train: {y_train.shape}, x_valid: {x_valid.shape}, y_valid: {y_valid.shape}')

    # Prepare training data held by each client
    client_data, N, prop_positive, _ = generate_clients_data(x=x_train,
                                                             y=y_train,
                                                             M=num_clients,
                                                             client_size_factor=client_size_factor,
                                                             class_balance_factor=class_balance_factor,
                                                             dataset_seed=dataset_seed)

    # Validation set, to predict on using global model
    valid_set = {'x' : torch.tensor(x_valid).float(),
                 'y' : torch.tensor(y_valid).float()}

    return client_data, valid_set, N, prop_positive, full_data_split



def acc_and_ll(server, x, y):
    """Calculate model prediction acc & logl
    """

    pred_probs = server.model_predict(x)
    pred_probs = pred_probs.mean.detach().numpy()
    valid_acc = np.mean((pred_probs > 0.5) == y.numpy())
    
    probs = torch.clamp(torch.tensor(pred_probs),min= 0., max=1.)
    valid_loglik = torch.distributions.Bernoulli(probs=probs).log_prob(y)
    valid_loglik = valid_loglik.mean().numpy()
    
    return valid_acc, valid_loglik



def get_nth_split(n_splits, n, folder):
    """data splitter
    Args:
        n_splits : k for k-fold split
        n : which of the n_splits splits to use, returns only single split
    """
    # Load inputs and outputs
    x = np.load(folder + 'x.npy')
    y = np.load(folder + 'y.npy')[:, 0]
    
    # Kfold splitter from sklearn
    kfold = KFold(n_splits=n_splits, shuffle=False)
    
    # Split data to n_splits splits
    splits = list(kfold.split(x))
    x_train = x[splits[n][0]]
    x_valid = x[splits[n][1]]

    splits = list(kfold.split(y))
    y_train = y[splits[n][0]]
    y_valid = y[splits[n][1]]

    return x_train, x_valid, y_train,  y_valid


def generate_clients_data(x, y, M, client_size_factor, class_balance_factor, dataset_seed):
        # this function ought to return a list of (x, y) tuples.
        # you need to set the seed in the main experiment file to ensure that this function becomes deterministic

        if dataset_seed is not None:
            random_state = np.random.get_state()
            np.random.seed(dataset_seed)

        if M == 1:
            client_data = [{"x": x, "y": y}]
            N_is = [x.shape[0]]
            props_positive = [np.mean(y > 0)]

            return client_data, N_is, props_positive, M

        if M % 2 != 0: raise ValueError('Num clients should be even for nice maths')

        N = x.shape[0]
        small_client_size = int(np.floor((1 - client_size_factor) * N/M))
        big_client_size = int(np.floor((1 + client_size_factor) * N/M))

        class_balance = np.mean(y == 0)

        small_client_class_balance = class_balance + (1 - class_balance) * class_balance_factor
        small_client_negative_class_size = int(np.floor(small_client_size * small_client_class_balance))
        small_client_positive_class_size = int(small_client_size - small_client_negative_class_size)

        if small_client_negative_class_size < 0: raise ValueError('small_client_negative_class_size is negative, invalid settings.')
        if small_client_positive_class_size < 0: raise ValueError('small_client_positive_class_size is negative, invalid settings.')


        if small_client_negative_class_size * M/2 > class_balance * N:
            raise ValueError(f'Not enough negative class instances to fill the small clients. Client size factor:{client_size_factor}, class balance factor:{class_balance_factor}')

        if small_client_positive_class_size * M/2 > (1-class_balance) * N:
            raise ValueError(f'Not enough positive class instances to fill the small clients. Client size factor:{client_size_factor}, class balance factor:{class_balance_factor}')


        pos_inds = np.where(y > 0)
        zero_inds = np.where(y == 0)
        
        assert (len(pos_inds[0]) + len(zero_inds[0])) == len(y), "Some indeces missed."

        y_pos = y[pos_inds]
        y_neg = y[zero_inds]

        x_pos = x[pos_inds]
        x_neg = x[zero_inds]

        client_data = []

        # Populate small classes.
        for i in range(int(M/2)):
            client_x_pos = x_pos[:small_client_positive_class_size]
            x_pos = x_pos[small_client_positive_class_size:]
            client_y_pos = y_pos[:small_client_positive_class_size]
            y_pos = y_pos[small_client_positive_class_size:]

            client_x_neg = x_neg[:small_client_negative_class_size]
            x_neg = x_neg[small_client_negative_class_size:]
            client_y_neg = y_neg[:small_client_negative_class_size]
            y_neg = y_neg[small_client_negative_class_size:]

            client_x = np.concatenate([client_x_pos, client_x_neg])
            client_y = np.concatenate([client_y_pos, client_y_neg])

            shuffle_inds = np.random.permutation(client_x.shape[0])

            client_x = client_x[shuffle_inds, :]
            client_y = client_y[shuffle_inds]

            client_data.append({'x': client_x, 'y': client_y})

        # Recombine remaining data and shuffle.
        x = np.concatenate([x_pos, x_neg])
        y = np.concatenate([y_pos, y_neg])
        shuffle_inds = np.random.permutation(x.shape[0])

        x = x[shuffle_inds]
        y = y[shuffle_inds]

        # Distribute among large clients.
        for i in range(int(M/2)):
            client_x = x[:big_client_size]
            client_y = y[:big_client_size]

            x = x[big_client_size:]
            y = y[big_client_size:]

            client_data.append({'x': client_x, 'y': client_y})

        N_is = [data['x'].shape[0] for data in client_data]
        props_positive = [np.mean(data['y'] > 0) for data in client_data]

        if dataset_seed is not None:
            np.random.set_state(random_state)

        return client_data, N_is, props_positive, M



def bin_search_sigma(target_eps, ncomp, target_delta, q, nx, L, lbound, ubound, tol, max_iters=10):

    for i_iter in tqdm(range(max_iters), disable=True):
        cur = (ubound+lbound)/2

        eps = fourier_accountant.get_epsilon_S(target_delta=target_delta, sigma=cur, q=q, ncomp=ncomp, nx=nx,L=L)
        print(f'iter {i_iter}, sigma={cur}, eps={eps:.5f}, upper={ubound}, lower={lbound}')

        if np.abs(eps - target_eps) <= tol:
            print(f'found eps={eps:.5f} with sigma={cur:.5f}')
            return cur

        if eps < target_eps:
            ubound = cur
        else:
            lbound = cur

        if np.abs(ubound - lbound) < 1e-2:
            print('Upper and lower bounds too close, failing!')
            return None
        
    print(f'Did not converge! final sigma={cur:.5f} wirh eps={eps:.5f}')



if __name__ == '__main__':


    # calculate privacy costs with fourier accountant
    nx=int(1E6)
    L=30.#26.

    target_delta = 1e-5
    q = .05

    ncomp=10*10
    # adult data:
    #samples_per_client = 2442 # with 10 clients
    #samples_per_client = 244 # with 100 clients
    # [2.6, 1.1] eps pitäisi olla suunnilleen

    # LFA: eps [2.6, 1.1, .8, .5, .2] corresponding noise:
    # 20 global updates:
    # sigmas in [14.11., 30.48, 40.81, 63.44, 152.5]
    # 10 global updates:
    # sigmas in [10.,    21.67, 28.95, 44.53, 102.83]

    # DPSGD: eps [2.6, 1.1, .8, .5, .2] corresponding noise:
    # 200 total local steps
    # q=.05
    #sigmas in [2.23, 4.86, 6.49, 9.80, 23.15]
    # q=.1
    #sigmas in [4.45, 9.61, 12.88, 20.14, 46.09]
    # q=.2
    #sigmas in [8.94, 19.41, 25.92, 39.28, 91.95]

    # 100 total local steps:
    # q=.05
    #sigmas in [1.60, 3.45, 4.63, ?, 16.35]
    # q=.1
    #sigmas in [3.15, 6.79, 9.16, ?, 32.48]
    # q=.2
    #sigmas in [6.31, 13.77, 18.22, ?, 65.13]

    # Global VI: DPSGD: eps [.2] corresponding noise:
    # 200 total local steps
    # q=.005
    #sigmas in [2.34]
    # q=.01
    #sigmas in [4.83]
    # q=.02
    #sigmas in [9.42]


    sigma = bin_search_sigma(.2, ncomp, target_delta, q, nx, L, lbound=1., ubound=200., tol=1e-3, max_iters=20)
    if sigma is not None:
        eps = fourier_accountant.get_epsilon_S(target_delta=1e-5, sigma=sigma, q=q, ncomp=ncomp, nx=nx,L=L)
        print(f'eps={eps} with sigma={sigma}')

    if 0:
        all_sigmas = [2.2, 4.7]
        all_eps = np.zeros((len(all_sigmas)))
        for i_sigma, sigma in enumerate(all_sigmas):
            eps = fourier_accountant.get_epsilon_S(target_delta=1e-5, sigma=sigma, q=q, ncomp=ncomp, nx=nx,L=L)
            all_eps[i_sigma] = eps
        print(f'all eps bounds:\n{all_eps}')
        print(f'corresponding sigmas:\n{all_sigmas}')
        #print(f'with clipping C={C} this amounts to actual noise std \n{[s*sens for s in all_sigmas]}')
        #print(f'fourier lower and upper bounds:\n{eps_lower}, {eps_upper}')
    sys.exit()


