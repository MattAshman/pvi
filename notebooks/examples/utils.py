
import logging
import os
import sys

import numpy as np
from sklearn.model_selection import KFold
import torch

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from pvi.models.logistic_regression import LogisticRegressionModel
#from pvi.clients.synchronous_client import SynchronousClient
from pvi.clients import Client
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
        if args.sampling_type == 'poisson':
            expected_batch.append(args.sampling_frac_q*data['x'].shape[0])

        # Approximating likelihood term of ith client
        t = MeanFieldGaussianFactor(nat_params=init_nat_params)

        # Create client and store
        client = Client(data=data, model=model, t=t, config=config)
        clients.append(client)

    if args.sampling_type == 'poisson':
        logger.info(f'Expected batch sizes: {expected_batch}')
        
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
    
    probs = torch.clip(torch.tensor(pred_probs), 0., 1.)
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


