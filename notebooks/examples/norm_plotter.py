
from collections import OrderedDict as OD
from pathlib import Path
import sys


import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from matplotlib import pyplot as plt
import numpy as np



# numbers of runs to plot
runs_to_plot = np.linspace(38,43,6,dtype=int)#[1]
#runs_to_plot = np.linspace(60,76,17,dtype=int)#[1]
#runs_to_plot = np.linspace(49,210,162,dtype=int)#[1]
#print(runs_to_plot)

# main folder where the res are
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dp_pvi_clipping_tests_swor/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dp_pvi_clipping_tests_mushroom/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dp_pvi_clipping_tests_credit/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dp_pvi_clipping_tests_bank/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dp_pvi_tests_adult_param_dp/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dp_pvi_adult_optim/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dp_pvi_adult_optim2/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dp_pvi_hfa_adult/'
main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/hfa_adult_lr_clip_runs/'
#main_folder = 'dp_pvi_runs/'

dataset_name = 'adult'
#dataset_name = 'mushroom'
#dataset_name = 'credit'
#dataset_name = 'bank'


# where to save all plots
fig_folder = 'res_plots/'

# baseline models to include: there should be 1 entry for each baseline in every given attribute
include_baselines = False
tmp = OD()
tmp['use_dpsgd'] = [False, False]
tmp['n_steps'] = [10,100]
tmp['data_bal_rho'] = [.0, 0.]
#tmp['data_bal_rho'] = [.7, .7]
#tmp['data_bal_rho'] = [.75, .75]
tmp['data_bal_kappa'] = [0.,0.]
#tmp['data_bal_kappa'] = [-3.,-3.]
#tmp['data_bal_kappa'] = [.95,.95]

restrictions = OD()
restrictions['dp_sigma'] = None#[2.,3.5, 4.,8.]
restrictions['dp_C'] = None#[1.]
#restrictions['use_dpsgd'] = [True]
restrictions['n_global_updates'] = None
restrictions['n_steps'] = [10]
restrictions['batch_size'] = [10]
restrictions['learning_rate'] = [1e-2]
restrictions['damping_factor'] = None#[.5]

# possible balance settings: (0,0), (.7,-3), (.75,.95)
restrictions['data_bal_rho'] = [0.]
restrictions['data_bal_kappa'] = [0.]

# set vars to add to title: key=variable name, value=var name in fig title
add_to_title = {}
add_to_title['data_bal_rho'] = 'rho'
add_to_title['data_bal_kappa'] = 'kappa'
add_to_title['clients'] = 'clients'
add_to_title['dp_mode'] = 'dp mode'
#add_to_title[''] = ''


# set labels to add: key=variable name, value=label name in fig
add_labels = {}
#add_labels['sampling_type'] = 'sampling'
add_labels['dp_C'] = 'C'
add_labels['dp_sigma'] = 'sigma'
add_labels['n_steps'] = 'steps'
add_labels['batch_size'] = 'b'
add_labels['learning_rate'] = 'lr'
add_labels['damping_factor'] = 'damping'
#add_labels['privacy_calculated'] = 'DP epochs'
#add_labels['data_bal_rho'] = 'rho'
#add_labels['data_bal_kappa'] = 'kappa'

# save to disk (or just show)
to_disk = 0
# name for the current plot
#fig_name = "{}_dpsgd_bal({},{})_eps1_1_and_2_7.pdf".format(dataset_name, restrictions['data_bal_rho'][0],restrictions['data_bal_kappa'][0])
fig_name = "{}_hfa_non_dp_no_no_clip_bal({},{})_b{}.pdf".format(dataset_name, restrictions['data_bal_rho'],restrictions['data_bal_kappa'], restrictions['batch_size'])


#####################
# default color cycle
import pylab

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # note: stasndard color cycler has 10 colors
#cm = plt.get_cmap('viridis')
#colors = (cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS))

#print(colors, len(colors))
#sys.exit()

all_res = OD()
all_res['config'] = OD()
all_res['client_train_res'] = OD()
all_res['train_res'] = OD()
all_res['validation_res'] = OD()
all_res['client_norms'] = OD()


if include_baselines:
    baselines = []
    for k in tmp:
        #print(f'baseline k={k}')
        if len(baselines) == 0:
            #print(f'creating {len(tmp[k])} baseline models')
            for i_model in range(len(tmp[k])):
                baselines.append(OD())
        for i_attr, attr in enumerate(tmp[k]):
            baselines[i_attr][k] = attr

    #print(baselines)
    #sys.exit()

jsonpickle_numpy.register_handlers()
failed_runs = []

for i_run in runs_to_plot:
    print(f'run {i_run}')
    filename = main_folder + str(i_run) + '/config.json'
    #print(f'trying {filename}')
    try:
        with open(filename, 'r') as f:
            apu = f.read()
    except FileNotFoundError as err:
        print(f"Can't open file {filename}! Skipping")
        failed_runs.append(i_run)
        continue

    apu = jsonpickle.unpickler.decode(apu)
    all_res['config'][str(i_run)] = apu
    print(apu)

    filename = main_folder + str(i_run) + '/info.json'
    with open(filename, 'r') as f:
        apu = f.read()
    apu = jsonpickle.unpickler.decode(apu)
    #print(apu)
    #for k in apu:
    #    print(k)
    #sys.exit()
    
    client_measures = ['elbo','kl','logl']
    all_res['client_train_res'][str(i_run)] = {}
    for k in client_measures:
        all_res['client_train_res'][str(i_run)][k] = np.zeros((
            all_res['config'][str(i_run)]['clients'],  
            all_res['config'][str(i_run)]['n_global_updates'],  
            all_res['config'][str(i_run)]['n_steps'],  
            all_res['config'][str(i_run)]['n_rng_seeds'],
            ))
    measures = ['acc','logl']
    all_res['train_res'][str(i_run)] = {}
    all_res['validation_res'][str(i_run)] = {}

    for k in measures:
        all_res['train_res'][str(i_run)][k] = np.zeros((
            all_res['config'][str(i_run)]['n_global_updates'],
            all_res['config'][str(i_run)]['n_rng_seeds']
            ))
        all_res['validation_res'][str(i_run)][k] = np.zeros((
            all_res['config'][str(i_run)]['n_global_updates'],
            all_res['config'][str(i_run)]['n_rng_seeds']
            ))

    all_res['client_norms'][str(i_run)] = {}
    all_res['client_norms'][str(i_run)]['pre_dp_norms'] = np.zeros((
            all_res['config'][str(i_run)]['clients'],
            all_res['config'][str(i_run)]['n_global_updates']*all_res['config'][str(i_run)]['n_steps'],
            all_res['config'][str(i_run)]['n_rng_seeds'],
            ))
    all_res['client_norms'][str(i_run)]['post_dp_norms'] = np.zeros((
            all_res['config'][str(i_run)]['clients'],
            all_res['config'][str(i_run)]['n_global_updates'],
            all_res['config'][str(i_run)]['n_rng_seeds'],
            ))


    for i_seed in range(all_res['config'][str(i_run)]['n_rng_seeds']):
        all_res['client_norms'][str(i_run)]['pre_dp_norms'][:,:,i_seed] = apu[f'tracked_seed{i_seed}']['client_norms']['pre_dp_norms']
        all_res['client_norms'][str(i_run)]['post_dp_norms'][:,:,i_seed] = apu[f'tracked_seed{i_seed}']['client_norms']['post_dp_norms']
        # logl, elbo, kl
        '''
        if i_seed == 0:
            for k in apu[f'validation_res_seed{i_seed}']:
                print(k)
            print(apu[f'validation_res_seed{i_seed}']['acc'].shape )
            print(apu[f'client_train_res_seed{i_seed}']['logl'].shape )
        #'''
        #sys.exit()
        for k in client_measures:
            all_res['client_train_res'][str(i_run)][k][:,:,:,i_seed] = apu[f'client_train_res_seed{i_seed}'][k]
        for k in measures:
            all_res['train_res'][str(i_run)][k][:,i_seed] = apu[f'train_res_seed{i_seed}'][k]
            all_res['validation_res'][str(i_run)][k][:,i_seed] = apu[f'validation_res_seed{i_seed}'][k]

        # so this should be of shape: clients * global updates * local steps
        #print(apu[f'client_train_res_seed{i_seed}']['logl'].shape )

        #sys.exit()
    #all_res['client_train_res'][str(i_run)] = apu['client_train_res']
    #all_res['train_res'][str(i_run)] = apu['train_res']
    #all_res['validation_res'][str(i_run)] = apu['validation_res']


if len(failed_runs) > 0:
    print(f'failed runs:\n{failed_runs}')
    runs_to_plot = list(runs_to_plot)
    for i_run in failed_runs:
        runs_to_plot.remove(i_run)
    runs_to_plot = np.array(runs_to_plot)


# plot tracked norms
# check restrictions
list_to_print = []
for i_run in runs_to_plot:
    print_this = True
    for k in restrictions:
        if restrictions[k] is not None and all_res['config'][str(i_run)][k] not in restrictions[k]:
            print_this = False
    if print_this:
        list_to_print.append(i_run)
if len(list_to_print) == 0:
    sys.exit('No runs satisfying restrictions found!')

fig, axs = plt.subplots(2, figsize=(10,10))
for k in add_to_title:
    try:
        cur_title += ", {}={}".format(add_to_title[k], all_res['config'][str(list_to_print[0])][k])
    except:
        try:
            cur_title = ": {}={}".format(add_to_title[k], all_res['config'][str(list_to_print[0])][k])
        except:
            cur_title = ''

for i_line,i_run in enumerate(list_to_print):
    cur_label = None
    config = all_res['config'][str(i_run)]

    for k in add_labels:
        try:
            cur_label += ", {}={}".format(add_labels[k], config[k])
        except:
            cur_label = "{}={}".format(add_labels[k], config[k])

    print('shapes, pre dp:{}, post dp: {}'.format(all_res['client_norms'][str(i_run)]['pre_dp_norms'].shape,all_res['client_norms'][str(i_run)]['post_dp_norms'].shape ))
    # eli pre: clients*(global*local steps) ja post: clients*global

    # pre dp client level
    x1 = np.linspace(1,config['n_global_updates']*config['n_steps'],config['n_global_updates']*config['n_steps'])
    # post dp norms
    x2 = np.linspace(1,config['n_global_updates'],config['n_global_updates'])

    # plot pre & post dp norms for each client, mean over seeds
    for i_client in range(config['clients']):
        axs[0].plot(x1, all_res['client_norms'][str(i_run)]['pre_dp_norms'][i_client,:].mean(-1) )
        axs[1].plot(x2, all_res['client_norms'][str(i_run)]['post_dp_norms'][i_client,:].mean(-1) )

    axs[0].set_ylabel('Pre DP norm')
    axs[0].set_xlabel('Local steps')
    #axs[0].legend()
    axs[0].grid(b=True, which='major', axis='both')
    axs[1].set_ylabel('Post DP norm')
    axs[1].set_xlabel('Global communications')
    #axs[1].legend()
    axs[1].grid(b=True, which='major', axis='both')

    plt.suptitle(f"{dataset_name} dataset tracked norms for all clients, mean over {config['n_rng_seeds']} runs" + cur_title)
    axs[0].set_title(cur_label)

    if to_disk:
        plt.savefig(fig_folder+fig_name)
        fig_folder

    else:
        plt.show()








