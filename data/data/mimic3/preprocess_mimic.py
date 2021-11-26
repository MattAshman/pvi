"""
Script for preprocessing MIMIC III data for DP-PVI testing.
Assume MIMIC data has been extracted with the code from Harutyunyan et al. 2019 in-hospital mortality prediction: https://github.com/YerevaNN/mimic3-benchmarks
This code uses data readers from the above paper.

After preprocessing:
i) combine train and validation data,
ii) dimensionality reduction: learn logistic regression model on train data, only keep dims with largest abs value weights
iii) divide to 5 clients: do 10 cluster kmeans, combine smallest categories to arrive at 5 partitions with roughly same order of samples, uneven data balance
"""

import os
import sys

import argparse
from copy import copy
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import importlib


#### mimic3 data readers from Harutyunyan et al. 2019 ####
def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])

def get_readers(input_dir):
    '''
    Args:
        input_dir : path to the in-hospital mortality task subfolder
    '''

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(input_dir, 'train'),
                                                 listfile=os.path.join(input_dir, 'train_listfile.csv'),
                                                 period_length=48.0)
    # note: validation data separated from train, no separate folder
    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(input_dir, 'train'),
                                                 listfile=os.path.join(input_dir, 'val_listfile.csv'),
                                                 period_length=48.0)
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(input_dir, 'test'),
                                                 listfile=os.path.join(input_dir, 'test_listfile.csv'),
                                                 period_length=48.0)

    return train_reader, val_reader, test_reader
##########################################################


def do_data_split(args):
    """
    main script for data preprocessing
    """

    train_reader, val_reader, test_reader = get_readers(args.input_dir)
    # read data
    period, features = 'all', 'all'
    print('Reading data and extracting features ...')
    (train_X, train_y, _) = read_and_extract_features(train_reader, period, features)
    (val_X, val_y, _) = read_and_extract_features(val_reader, period, features)
    (test_X, test_y, _) = read_and_extract_features(test_reader, period, features)
    print('Done reading data')
    print('  train data shape = {}'.format(train_X.shape))
    print('  valid data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))

    print('Combining train and validation data, change to np.arrays')
    # concat train and val to create larger train set
    train_X = np.concatenate((train_X,val_X))
    train_y = np.concatenate((train_y,val_y))
    #print('train data shapes: {}, {}'.format(train_X.shape, train_y.shape)) # should be (17903, 714), (17903,)
    assert train_X.shape == (17903,714) and train_y.shape == (17903,)
    print('Done.')

    # change from lists to np.arrays
    test_X, test_y = np.array(test_X), np.array(test_y)
    #print('test data shapes: {}, {}'.format(test_X.shape, test_y.shape)) # should be (3236,714), (3236,)
    assert test_X.shape == (3236,714) and test_y.shape == (3236,)

    # impute missing data
    print('Imputing missing data')
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=False)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    imputer.fit(test_X)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)
    print('Done.')

    # Normalize data to have zero mean and unit variance; note: not fully normalized after dim reduction
    print('Normalizing data')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    scaler.fit(test_X)
    test_X = scaler.transform(test_X)
    print('Done.')

    print('Doing dimensinality reduction via logistic regression modelling')
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X=train_X,y=train_y)
    # find indices for largest weights in abs value
    largest_inds = np.argsort(-np.abs(model.coef_[0]))[:args.target_dim]
    train_X = train_X[:,largest_inds]
    test_X = test_X[:,largest_inds]
    print('Done, new train_X shape: {}, new test_X shape: {}'.format(train_X.shape, test_X.shape))
    assert train_X.shape == (17903,args.target_dim) and test_X.shape == (3236,args.target_dim)

    # Renormalize data to have zero mean and unit variance; note
    print('Renormalizing data')
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    scaler.fit(test_X)
    test_X = scaler.transform(test_X)
    print('Done.')

    print('Dividing training data into client partitions via K-means')
    kmeans = KMeans(n_clusters=args.n_clusters, max_iter=300, tol=0.0001, verbose=0, random_state=2303, copy_x=True, algorithm='auto')
    kmeans_res = kmeans.fit(train_X)
    cur_clusters = np.linspace(0,args.n_clusters-1, args.n_clusters, dtype=int)
    labels = copy(kmeans_res.labels_)
    for i in range(args.n_clusters):
        if len(cur_clusters) <= args.n_clients:
            print(f'Finished combining clusters with {len(cur_clusters)} clusters remaining')
            break
        # combine 2 smallest clusters
        tmp = np.array([np.sum(labels==i_client) for i_client in cur_clusters])
        smallest_inds = np.argsort(tmp)[:2]
        labels[labels == cur_clusters[smallest_inds[0]]] = cur_clusters[smallest_inds[1]]
        cur_clusters = np.delete(cur_clusters, smallest_inds[0])

    print('Final client data splits:')
    for i_client, client in enumerate(cur_clusters):
        print(f'client {i_client}: N={np.sum(labels==client)}, pos.frac.={ (np.sum(train_y[labels==client]==1) )/np.sum(labels==client):.5f}')

    filename = args.output_dir + f'mimic_in-hospital_split_{args.n_clients}clients.npz'
    print(f'Saving all partitions to {filename}')
    partitions = {}
    partitions['x_test'] = test_X
    partitions['y_test'] = test_y
    for i_client in range(args.n_clients):
        partitions[f'x_{i_client}'] = train_X[labels==cur_clusters[i_client],:]
        partitions[f'y_{i_client}'] = train_y[labels==cur_clusters[i_client]]
        
    np.savez_compressed(args.output_dir+f'mimic_in-hospital_split_{args.n_clients}clients', **partitions)
    print('All done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    
    parser.add_argument('--input_dir', default='/Users/mixheikk/Documents/mimic3-benchmarks/data/in-hospital-mortality/', type=str, help="Path to in-hospital mortality subfolder created by MIMIC preprocessing scripts")

    parser.add_argument('--output_dir', default='/Users/mixheikk/Documents/git/DP-PVI/pytorch_pvi/pvi/data/data/mimic3/', type=str, help="Path to output dir")
    parser.add_argument('--target_dim', default=50, type=int, help="Target data dimensionality <= 714, use logistic regression abs weight valeus to determine most important dims")
    parser.add_argument('--n_clusters', default=10, type=int, help="Initial number of clusters for K-means; used in partitioning data to clients; >= n_clients")
    parser.add_argument('--n_clients', default=5, type=int, help="Number of clients; smallest K-means clusters will be combined until n_clients clusters remain")

    args = parser.parse_args()

    # a bit of workaround
    sys.path.append('/Users/mixheikk/Documents/mimic3-benchmarks')
    from mimic3benchmark.readers import InHospitalMortalityReader
    from mimic3models import common_utils

    do_data_split(args)






