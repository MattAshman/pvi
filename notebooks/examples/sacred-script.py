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

SETTINGS['CAPTURE_MODE'] = 'sys' #fd, sys; note sys might omit some stuff, fd might cause problems with heartbeat events resulting in losing results? using sys didn't fix heatbeat failures

from dp_logistic_regression import main as main_log_regr
from run_bnn import main as main_bnn

# save some time by omitting gpu info
#SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
#handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

ex = Experiment('DP-PVI testing', save_git_info=False)

# uncomment chosen experiment here, choose check configs below
ex.observers.append(FileStorageObserver('adult_1bnn_lfa_10clients_5seeds_eps2_fix_runs'))

# try handliong progress bars nicely
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def short_test(_log):
    """Default settings for short testing runs. All of these will be overwritten if any named config is used.
    """

    # static parameters
    model = 'pvi' # which model to use: 'pvi', 'bcm_same', or 'bcm_split'
    track_params = False # track all params, should usually be False
    track_client_norms = False # track all (grad) norms, should usually be False
    plot_tracked = False # plot all tracked stuff after learning, should usually be False
    pbar =  True # disable progress bars
    folder = '../../data/data/adult1/' # data folder, uncomment one
    #folder = '../../data/data/mimic3/' #
    #folder = '../../data/data/abalone/' # NOTE: abalone not working at the moment
    #folder = '../../data/data/mushroom/' # note: data bal (.7,-3) with 10 clients not working, can't populate small classes
    #folder = '../../data/data/credit/' # note: data bal (.7,-3) with 10 clients not working, can't populate small classes
    #folder = '../../data/data/bank/'
    #folder = '../../data/data/superconductor/'
    #folder = '../../data/data/MNIST/' # note: only for BNN models
    clients = 10
    n_rng_seeds = 1 # number of repeats to do for a given experiment; initial seed from sacred
    #prior_sharing = 'same' # 'same' or 'split': for distr_vi, whether to use same or split prior in BCM
    batch_proc_size =  1 # batch proc size; currently needs to be 1 for DP-SGD
    job_id = 0 # id that defines arg combination to use, in [0,nbo of combinations-1]; replace this by command line arg
    privacy_calculated = None # how many global steps assumed to run with the given privacy budget; doesn't stop run if global steps is more!
    server = 'sequential' # server type: 'synchronous' or 'sequential'
    # BNN specific options:
    use_bnn = True # if True use main from run_bnn.py, else from dp_logistic_regression.py
    n_classes = 2 # number of predicted classes, 2 for adult, mimic3, 10 for MNIST
    latent_dim = 50 # latent layer size
    n_layers = 1 # number of latent layers in the network
    init_var = [5e-3] # Initial BNN variance
    freeze_var_updates = [0]
    use_lr_scheduler = False # use learning rate scheduler
    use_nat_grad = False # use natural gradient (only with mean-field Gaussians!)

    # dynamic parameters; choose which combination to run by job_id
    batch_size =  [None] # batch size; used for dp_mode: 'dpsgd', 'param', 'param_fixed'; for 'lfa' use always batch_size=1
    n_global_updates = [40] # number of global updates
    n_steps = [80] # when sampling_type 'poisson' or 'swor': number of local training steps on each client update iteration; when sampling_type = 'seq': number of local epochs, i.e., full passes through local data on each client update iteration
    damping_factor = [.2,.4] # damping factor in (0,1], 1=no damping
    learning_rate = [1e-3]
    sampling_frac_q = [2e-3] # sampling fraction; only used if sampling_type is 'poisson'
    pseudo_client_q = [1.] # sampling frac ONLY for pseudo-clients when using local_pvi

    data_bal = [(.75,.95)] # list of (rho,kappa) values; for MNIST use (0,None) for homogenic split, (anything else,None) for unbalanced, no effect on Mimic3
    dp_mode = 'lfa' # 'dpsgd', 'param' for param pert. by each client, 'param_fixed' for param. pert. by each client using a fixed minibatch per each global update, 'lfa' for local fedavg, 'nondp_epochs' for nondp with n_steps batches per global update, or 'nondp_batches' for nondp with n_steps local epochs per global update
    pre_clip_sigma = [0.] # pre clipping noise to mitigate clipping bias
    dp_sigma = [25.22] # dp noise std factor; full noise std will be C*sigma
    dp_C = [20.] # max grad norm
    enforce_pos_var = False # enforce pos.var by taking abs values when convertingfrom natural parameters; NOTE: bit unclear if works at the moment! better not to use
    mixed_dp_mode_params = {}

    # get dynamic configuration, this will raise TypeError if params have already been set by some named config
    try:
        if job_id >= (len(batch_size)*len(n_global_updates)*len(n_steps)*len(damping_factor)*len(learning_rate)*len(sampling_frac_q)*len(data_bal)*len(pre_clip_sigma)*len(dp_sigma)*len(dp_C)*len(init_var)*len(freeze_var_updates)*len(pseudo_client_q)):
            raise ValueError('job_id > number of possible parameter combinations!')
        for i_comb, comb in enumerate(itertools.product(batch_size, n_global_updates, n_steps, damping_factor, learning_rate, sampling_frac_q, pseudo_client_q, data_bal, pre_clip_sigma, dp_sigma, dp_C, init_var, freeze_var_updates)):
            if i_comb == job_id:
                batch_size, n_global_updates, n_steps, damping_factor, learning_rate, sampling_frac_q, pseudo_client_q, data_bal, pre_clip_sigma, dp_sigma, dp_C, init_var, freeze_var_updates = comb
                data_bal_rho, data_bal_kappa = data_bal
                break
        del i_comb, comb
        #num_iterations = 50//n_steps
    except TypeError as err:
        logger.debug(f'Default config raised TypeError:{err}. "\'int\' has no len" should occur when using any named config.')
        pass
    del data_bal



# NOTE: automain needs to be at the end of file, currently handling explicitly with regular main
@ex.main
def dp_pvi_config_handler(_config, _seed):
    
    #print(_config)

    parser = argparse.ArgumentParser(description="help parser")
    parser.set_defaults(**_config)
    args, _ = parser.parse_known_args() # need to avoid conflict with sacred args

    if args.use_bnn:
        main = main_bnn
    else:
        main = main_log_regr

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
        ex.info[f'tracked_seed{i_seed}'] = res[4]

    # manually save bck to avoid issues due to sacred heartbeat failures
    import json
    import jsonpickle
    import jsonpickle.ext.numpy as jsonpickle_np
    jsonpickle_np.register_handlers()

    code = jsonpickle.encode(ex.info)
    with open(ex.current_run.observers[0].dir+'/info_bck.json', 'w') as f:
        json.dump(code, f)

    logger.info('Sacred test finished.')

if __name__ == '__main__':
    ex.run_commandline()

    # try giving some time for CSC cluster I/O to finish before getting killed
    time.sleep(30)


