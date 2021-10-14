import argparse
import itertools
import logging
import sys
import time

import numpy as np
from numpy.random import SeedSequence
#from numpyencoder import NumpyEncoder

from sacred import Experiment
#from sacred.observers import TinyDbObserver
from sacred.observers import FileStorageObserver
from sacred import Ingredient
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

SETTINGS['CAPTURE_MODE'] = 'fd' #sys, note sys might omit some stuff

from dp_logistic_regression import main

# save some time by omitting gpu info
#SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ex = Experiment('DP-PVI testing', save_git_info=False)

# uncomment chosen experiment here, choose check configs below
ex.observers.append(FileStorageObserver('dp_pvi_hfa_adult'))
#ex.observers.append(FileStorageObserver('dp_pvi_adult_optim'))
#ex.observers.append(FileStorageObserver('dp_pvi_tests_adult_param_dp'))
#ex.observers.append(FileStorageObserver('dp_pvi_clipping_tests_swor'))
#ex.observers.append(FileStorageObserver('dp_pvi_clipping_tests_mushroom'))
#ex.observers.append(FileStorageObserver('dp_pvi_clipping_tests_credit'))
#ex.observers.append(FileStorageObserver('dp_pvi_clipping_tests_bank'))
#ex.observers.append(FileStorageObserver('dp_pvi_noise_tests'))

# try handliong progress bars nicely
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def short_test(_log):
    """Default settings for short testing runs. All of these will be overwritten if any named config is used.
    """

    # static parameters
    track_params = False # track all params, should usually be False
    track_client_norms = True # track all (grad) norms, should usually be False
    plot_tracked = False # plot all tracked stuff after learning, should usually be False
    folder = '../../data/data/adult/' # data folder, uncomment one
    #folder = '../../data/data/abalone/' # NOTE: abalone not working at the moment
    #folder = '../../data/data/mushroom/' # note: data bal (.7,-3) with 10 clients not working, can't populate small classes
    #folder = '../../data/data/credit/' # note: data bal (.7,-3) with 10 clients not working, can't populate small classes
    #folder = '../../data/data/bank/'
    #folder = '../../data/data/superconductor/'
    clients = 10
    n_rng_seeds = 1 # number of repeats to do for a given experiment; initial seed from sacred
    #parallel_updates = True # parallel or sequential updates; for distr_vi can only use True
    #prior_sharing = 'same' # 'same' or 'split': for distr_vi, whether to use same or split prior in BCM
    batch_proc_size =  1 # batch proc size; currently needs to be 1 for DP-SGD
    job_id = 0 # id that defines arg combination to use, in [0,nbo of combinations-1]; replace this by command line arg
    privacy_calculated = None # how many global steps assumed to run with the given privacy budget; doesn't stop run if global steps is more!

    # dynamic parameters; choose which combination to run by job_id
    batch_size =  [1] # batch size; used for dp_mode: 'dpsgd', 'param', 'param_fixed'; for 'hfa' use always batch_size=1
    n_global_updates = [20] # number of global updates
    n_steps = [10] # when sampling_type 'poisson' or 'swor': number of local training steps on each client update iteration; when sampling_type = 'seq': number of local epochs, i.e., full passes through local data on each client update iteration
    damping_factor = [.5] # damping factor in (0,1], 1=no damping
    learning_rate = [1e-2]
    sampling_frac_q = [1e-2] # sampling fraction; only used if sampling_type is 'poisson'
    data_bal = [(0,0)]#,(.75,.95),(.7,-3.)] # list of (rho,kappa) values NOTE: nämä täytyy muuttaa oikeisiin muuttujiin koodissa
    dp_mode = 'hfa' # 'dpsgd', 'param' for param pert. by each client, 'param_fixed' for param. pert. by each client using a fixed minibatch per each global update, 'fha' for hier. fedavg, or 'server' (don't use!)
    dp_sigma = [0.] # dp noise std factor; noise magnitude will be C*sigma
    dp_C = [.5,.75,1.,1.5] # max grad norm
    enforce_pos_var = False # enforce pos.var by taking abs values when convertingfrom natural parameters; NOTE: bit unclear if works at the moment!
    #server_add_dp = False # when not using dp_sgd, clip  & noisify change in parameters on the (synchronous) server, otherwise on each client. NOTE: this currently means that will clip & noisify after damping!
    #param_dp_use_fixed_sample = False # use fixed random sample of given batch size for optimisation with parameter DP (only on clients)")
    
    # get dynamic configuration, this will raise TypeError if params have already been set by some named config
    try:
        if job_id >= (len(batch_size)*len(n_global_updates)*len(n_steps)*len(damping_factor)*len(learning_rate)*len(sampling_frac_q)*len(data_bal)*len(dp_sigma)*len(dp_C)):
            raise ValueError('job_id > number of possible parameter combinations!')
        for i_comb, comb in enumerate(itertools.product(batch_size, n_global_updates, n_steps, damping_factor, learning_rate, sampling_frac_q, data_bal,dp_sigma, dp_C)):
            if i_comb == job_id:
                batch_size, n_global_updates, n_steps, damping_factor, learning_rate, sampling_frac_q, data_bal, dp_sigma, dp_C = comb
                data_bal_rho, data_bal_kappa = data_bal
                break
        del i_comb, comb
        #num_iterations = 50//n_steps
    except TypeError as err:
        logger.debug(f'Default config raised TypeError:{err}. "\'int\' has no len" should occur when using any named config.')
        pass
    del data_bal



@ex.named_config
def unbalanced_test_config(_log):
    """NOTE: NOT UPDATED
    Config for running basic params testing with (un)balanced data, with or without privacy
    """
    logger.info('Using unbalanced_test_config')
    # static parameters
    batch_size = None # list, only used for calculating test acc/loss, no effect on training. None to use full test set size
    folder = 'data/adult/' # data folder
    clients = 10
    model_type = 'pvi' # method: pvi, distr_vi, shared_vi
    n_rng_seeds = 5 # number of repeats to do for a given experiment; initial seed from sacred
    max_batch_factor = 2.5 # for simulating Poisson sampling with sampling without replacement; will crash if too low/high
    parallel_updates = True # parallel or sequential updates; note: can use seq only with pvi or shared_vi
    job_id = 0 # id that defines arg combination to use, in [0,nbo of combinations-1]; replace this by command line arg

    # dynamic parameters; choose which combination to run by job_id
    n_steps = [10,25,50] # number of local steps per iteration
    learning_rate = [5e-2,1e-2,5e-3]
    q = [[5e-3]] # list of Poisson sampling fraction for training
    data_bal = [(0,0),(.75,.95),(.7,-3)] # list of (rho,kappa) values
    # non-dp tests
    #dp_sigma = [0.0] # dp noise std factor; noise magnitude will be C*sigma
    #dp_C = [2000.] # max grad norm
    # DP testing: 
    dp_sigma = [1.] # dp noise std factor; noise magnitude will be C*sigma
    dp_C = [1.,2.,5.] # max grad norm

    # get dynamic configuration
    if job_id >= (len(n_steps)*len(learning_rate)*len(q)*len(data_bal)*len(dp_sigma)*len(dp_C)):
        raise ValueError('job_id > number of possible parameter combinations!')
    for i_comb, comb in enumerate(itertools.product(n_steps, learning_rate, q, data_bal, dp_sigma, dp_C)):
        if i_comb == job_id:
            n_steps, learning_rate, q, data_bal, dp_sigma, dp_C = comb
            data_bal_rho, data_bal_kappa = data_bal
            break
    
    num_iterations = 2000//n_steps
    del i_comb, comb, data_bal


# NOTE: automain needs to be at the end of file, currently handling explicitly with regular main
@ex.main
def dp_pvi_config_handler(_config, _seed):
    
    #print(_config)

    parser = argparse.ArgumentParser(description="help parser")
    parser.set_defaults(**_config)
    args, _ = parser.parse_known_args() # need to avoid conflict with sacred args


    '''
    if args.model_type == 'pvi':
        from pvi_master import main
    elif args.model_type == 'shared_vi':
        from shared_vi_master import main
    elif args.model_type == 'distr_vi':
        from distr_vi_master import main
    else:
        raise ValueError(f"Model type needs to be 'pvi', 'shared_vi', or 'distr_vi'. Got {args.model_type}.")
    '''

    ss = SeedSequence(_seed)
    rng_seed_list = ss.spawn(1)[0].generate_state(_config['n_rng_seeds'])
    ex.info['generated_rng_seeds'] = rng_seed_list
    logger.info(f'Main sacred seed: {_seed}, got generated seeds: {rng_seed_list}')
    for i_seed, rng_seed in enumerate(rng_seed_list):
        logger.info('########## Starting seed {}/{} ##########\n'.format(i_seed+1, len(rng_seed_list)))
        res = main(args, rng_seed, dataset_folder=_config['folder'])

        ex.info[f'validation_res_seed{i_seed}'] = res[0]
        ex.info[f'train_res_seed{i_seed}'] = res[1]
        ex.info[f'client_train_res_seed{i_seed}'] = res[2]
        ex.info[f'prop_positive_seed{i_seed}'] = res[3]
        ex.info[f'tracked_seed{i_seed}'] = res[3]

    logger.info('Sacred test finished.')

if __name__ == '__main__':
    ex.run_commandline()

    # try giving some time for CSC cluster I/O to finish before getting killed
    time.sleep(30)


