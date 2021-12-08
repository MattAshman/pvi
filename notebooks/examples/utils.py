
import logging
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import torch
import torch.utils.data
from torchvision import transforms, datasets
from tqdm import tqdm

import fourier_accountant

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from pvi.models.logistic_regression import LogisticRegressionModel
from pvi.clients import Client
from pvi.clients import Param_DP_Client
from pvi.clients import DPSGD_Client
from pvi.clients import LFA_Client
from pvi.clients import Local_PVI_Client
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
        if args.dp_mode in ['lfa']:
            client = LFA_Client(data=data, model=model, t=t, config=config)
            if i == 0:
                logger.debug('Init LFA clients')
        elif args.dp_mode in ['local_pvi']:
            client = Local_PVI_Client(data=data, model=model, t=t, config=config)
            if i == 0:
                logger.debug('Init local DPPVI clients')
        elif args.dp_mode in ['param']:
            client = Param_DP_Client(data=data, model=model, t=t, config=config)
            if i == 0:
                logger.debug('Init param DP clients')
        elif args.dp_mode in ['dpsgd']:
            client = DPSGD_Client(data=data, model=model, t=t, config=config)
            if i == 0:
                logger.debug('Init DPSGD clients')
        elif args.dp_mode in ['nondp']:
            client = Client(data=data, model=model, t=t, config=config)
            if i == 0:
                logger.debug('Init non-DP base clients')
        else:
            raise ValueError(f'Unknown dp_mode: {args.dp_mode}!')
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
        # NOTE: need to check which ones are now actually useful
    """

    # Get data split
    if 'mimic3' in dataset_folder:
        logger.debug('Reading MIMIC-III data')
        # read preprocessed mimic data that already includes client splits
        filename = dataset_folder+f'mimic_in-hospital_split_{num_clients}clients.npz'
        tmp = np.load(filename)
        client_data = []
        N, prop_positive = np.zeros(num_clients), np.zeros(num_clients)
        for i_client in range(num_clients):
            client_data.append({})
            client_data[-1]['x'] = tmp[f'x_{i_client}']
            client_data[-1]['y'] = tmp[f'y_{i_client}']
            N[i_client] = int(len(client_data[-1]['y']))
            prop_positive[i_client] = np.sum(client_data[-1]['y']==1)/N[i_client]

        train_set = {'x' : torch.tensor(np.concatenate([data_dict['x'] for data_dict in client_data ])).float(),
                     'y' : torch.tensor(np.concatenate([data_dict['y'] for data_dict in client_data ])).float() }
        # Validation set, to predict on using global model
        valid_set = {'x' : torch.tensor(tmp['x_test'], dtype=torch.float),
                     'y' : torch.tensor(tmp['y_test'], dtype=torch.float)}

        #print( (torch.sum(train_set['y']==0))/len(train_set['y']) )
        #print( (torch.sum(valid_set['y']==0))/len(valid_set['y']) )
        #sys.exit()

    elif 'MNIST' in dataset_folder:
        # note: use balanced split when class_balance_Factor == 0, unbalanced split otherwise
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])

        train_set = datasets.MNIST(root=dataset_folder, train=True, download=False, transform=transform_train)
        test_set = datasets.MNIST(root=dataset_folder, train=False, download=False, transform=transform_test)

        train_set = {
            "x": ((train_set.data - 0) / 255).reshape(-1, 28 * 28),
            "y": train_set.targets,
        }

        valid_set = {
            "x": ((test_set.data - 0) / 255).reshape(-1, 28 * 28),
            "y": test_set.targets,
        }
        logger.debug(f"MNIST shapes, train: {train_set['x'].shape}, {train_set['y'].shape}, test: {valid_set['x'].shape}, {valid_set['y'].shape}")
        # balanced split
        client_data = []
        N = np.zeros(num_clients)
        if class_balance_factor == 0:
            logger.info('Using Fed MNIST data with balanced split.')
            perm = np.random.permutation(len(train_set["y"]))
            for i_client in range(num_clients):
                client_idx = perm[i_client::num_clients]
                client_data.append({"x": train_set["x"][client_idx], "y": train_set["y"][client_idx]})
                N[i_client] = len(client_data[-1]['y'])
            #print(len(client_data))
            #print(client_data[0]['x'].shape,client_data[0]['y'].shape)

        else:
            logger.info('Using Fed MNIST data with unbalanced split.')
            # note: distribution of MNIST labels is not exactly 6000/digit, so some shares might contain 2 labels even when using 100 clients
            shard_len = 60000//(2*num_clients)
            n_shards = 60000//shard_len
            
            sorted_inds = torch.argsort(train_set['y'])
            shards = np.random.permutation(np.linspace(0,n_shards-1,n_shards))
            for i_client in range(num_clients):
                tmp_ind = shards[(2*i_client):(2*i_client+2)]
                shard_inds = sorted_inds[ np.concatenate([ np.linspace(i*shard_len,(i+1)*shard_len-1,shard_len,dtype=int) for i in tmp_ind]) ]
                client_data.append({"x": train_set["x"][shard_inds], "y": train_set["y"][shard_inds]})
                N[i_client] = len(client_data[-1]['y'])
            #print(len(client_data))
            #print(client_data[0]['x'].shape,client_data[0]['y'].shape)

            '''
            # check label sums: each label used exactly once
            tmp = np.zeros((2,10))
            for i_client in range(num_clients):
                for i in range(10):
                    if i_client == 0:
                        tmp[0,i] += torch.sum(train_set['y']==i)
                    tmp[1,i] += torch.sum( client_data[i_client]['y']==i )

            print(tmp)
            '''

        prop_positive = np.zeros(num_clients)*np.nan


    else:
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

        train_set = {'x' : torch.tensor(x_train).float(),
                     'y' : torch.tensor(y_train).float()}
        # Validation set, to predict on using global model
        valid_set = {'x' : torch.tensor(x_valid).float(),
                     'y' : torch.tensor(y_valid).float()}

    return client_data, train_set, valid_set, N, prop_positive


def acc_and_ll(server, x, y, n_points=101):
    """Calculate model prediction acc & logl with LogisticRegression model
    n_points : (int) number of points to calculate classification results
    """
    pred_probs = server.model_predict(x)
    pred_probs = pred_probs.mean.detach().numpy()
    acc = np.mean((pred_probs > 0.5) == y.numpy())
    
    probs = torch.clamp(torch.tensor(pred_probs),min= 0., max=1.)
    loglik = torch.distributions.Bernoulli(probs=probs).log_prob(y)
    loglik = loglik.mean().numpy()

    # get true & false positives and negatives
    if len(torch.unique(y)) == 2:
        posneg = get_posneg(y.numpy(), pred_probs, n_points)
    else:
        posneg = None

    return acc, loglik, posneg



def acc_and_ll_bnn(server, x, y, n_points=101):
    """Calculate model prediction acc & logl for BNNs
    """
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)

    preds, mlls = [], []
    for (x_batch, y_batch) in loader:
        #x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        pp = server.model_predict(x_batch)
        preds.append(pp.component_distribution.probs.mean(1).cpu())
        mlls.append(pp.log_prob(y_batch).cpu())

    mll = torch.cat(mlls).mean()
    preds = torch.cat(preds)
    #acc = sum(torch.argmax(preds, dim=-1) == loader.dataset.tensors[1]) / len(loader.dataset.tensors[1])
    acc = torch.true_divide(sum(torch.argmax(preds, dim=-1) == loader.dataset.tensors[1]), len(loader.dataset.tensors[1]))
    
    # get true & false pos. and neg.
    if len(torch.unique(y)) == 2:
        posneg = get_posneg(y.numpy(), preds[:,1].numpy(), n_points)
    else:
        posneg = None

    return acc, mll, posneg


def get_posneg(y, pred_probs, n_points):
    """Fun for calculating True & False positives and negatives from binary predictions
    """

    # binary classification
    if len(np.unique(y)) == 2:
        if n_points == 1:
            posneg = {
                    'TP' : [np.sum( (pred_probs > 0.5)[y==1] == y[y==1])],
                    'FP' : [np.sum( (pred_probs > 0.5)[y==0] != y[y==0])],
                    'TN' : [np.sum( (pred_probs <= 0.5)[y==0] == 1-y[y==0])],
                    'FN' : [np.sum( (pred_probs <= 0.5)[y==1] != 1-y[y==1])],
                    'n_points' : n_points,
                    }
        else:
            tmp = np.linspace(0,1,n_points)
            posneg = {
                    'TP' : np.zeros(n_points, dtype=int),
                    'FP' : np.zeros(n_points, dtype=int),
                    'TN' : np.zeros(n_points, dtype=int),
                    'FN' : np.zeros(n_points, dtype=int),
                    'n_points' : n_points,
                    }
            for i_thr, thr in enumerate(tmp):
                posneg['TP'][i_thr] =  int(np.sum( (pred_probs > thr)[y==1] == y[y==1]))
                posneg['FP'][i_thr] =  int(np.sum( (pred_probs > thr)[y==0] != y[y==0]))
                posneg['TN'][i_thr] =  int(np.sum( (pred_probs <= thr)[y==0] == 1-y[y==0]))
                posneg['FN'][i_thr] =  int(np.sum( (pred_probs <= thr)[y==1] != 1-y[y==1]))

        # add some ready metrics
        posneg['avg_prec_score'] =  metrics.average_precision_score(y_true=y, y_score=pred_probs)
        posneg['balanced_acc'] =  metrics.balanced_accuracy_score(y_true=y, y_pred=(pred_probs > .5))
        posneg['f1_score'] =  metrics.f1_score(y_true=y, y_pred=(pred_probs > .5))

    else:
        # multiclass classification
        raise NotImplementedError('multiclass bal acc etc not implemented!')


    return posneg



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

        #if np.abs(ubound - lbound) < 1e-3:
        #    print('Upper and lower bounds too close, failing!')
        #    return None
        
    print(f'Did not converge! final sigma={cur:.5f} wirh eps={eps:.5f}')



if __name__ == '__main__':


    # calculate privacy costs with fourier accountant
    nx=int(1E6)
    L=30.#26.

    target_delta = 1e-5
    #q = .2

    ncomp=20*10
    #########################################
    # adult data:
    #########################################
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

    #########################################
    #########################################


    #########################################
    # MIMIC3 data:
    #########################################
    # fixed 5 clients, use only sampling_fracs

    # DPSGD:       eps in [1,     2]
    # 100 comps:
    #   q=.01 dp_sigma in [0.95,  0.75]
    #   q=.05 dp_sigma in [3.73,  2.00]
    #   q=.1 dp_sigma in  [7.46,  3.98]
    #   q=.2 dp_sigma in  [14.91, 7.97]

    # 200 comps:   eps in [1,     2]
    #   q=.01 dp_sigma in [1.13,  0.81]
    #   q=.05 dp_sigma in [5.27,  2.82]
    #   q=.1 dp_sigma in  [10.54, 5.63]
    #   q=.2 dp_sigma in  [21.11, 11.27]

    # 300 comps: eps=1
    #   q=.01 dp_sigma in [1.33]
    #   q=.05 dp_sigma in [6.46]
    #   q=.1 dp_sigma in  [12.92]
    #   q=.2 dp_sigma in  [25.84]

    # 400 comps: eps=1
    #   q=.01 dp_sigma in [1.52]
    #   q=.05 dp_sigma in [7.45]
    #   q=.1 dp_sigma in  [14.92]
    #   q=.2 dp_sigma in  [29.87]

    # 600 comps: eps=1
    #   q=.01 dp_sigma in [1.84]
    #   q=.05 dp_sigma in [9.14]
    #   q=.1 dp_sigma in  [18.28]
    #   q=.2 dp_sigma in  [36.58]

    # 1000 comps: eps=1
    #   q=.01 dp_sigma in [2.36]
    #   q=.05 dp_sigma in [11.79]
    #   q=.1 dp_sigma in  [23.57]
    #   q=.2 dp_sigma in  [47.15]
    #########################################
    #########################################
    q = .2

    sigma = bin_search_sigma(1, ncomp, target_delta, q, nx, L, lbound=.7, ubound=100., tol=1e-3, max_iters=30)
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


