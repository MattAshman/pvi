
from collections import OrderedDict as OD
from pathlib import Path
import sys


import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from matplotlib import pyplot as plt
import numpy as np



# numbers of runs to plot
#runs_to_plot = np.linspace(28,76,49,dtype=int)#[1]
runs_to_plot = np.linspace(60,76,17,dtype=int)#[1]
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
main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dp_pvi_hfa_adult/'
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
restrictions['n_steps'] = None#[10
restrictions['batch_size'] = [100]
restrictions['learning_rate'] = [0.01]
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

baseline_acc = 0.761 # for adult data

all_res = OD()
all_res['config'] = OD()
all_res['client_train_res'] = OD()
all_res['train_res'] = OD()
all_res['validation_res'] = OD()


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
            all_res['config'][str(i_run)]['n_rng_seeds']
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

    # does this work with sampling=seq?

    for i_seed in range(all_res['config'][str(i_run)]['n_rng_seeds']):
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

# plot comparisons between given runs: all in same plot

plot_comparisons = True
if plot_comparisons:

    # check restrictions
    list_to_print = []
    for i_run in runs_to_plot:
        print_this = True
        for k in restrictions:
            if restrictions[k] is not None and all_res['config'][str(i_run)][k] not in restrictions[k]:
                print_this = False
        # check baselines
        if include_baselines and not print_this:
            for tmp in baselines:
                print_this = True
                for k in tmp:
                    if tmp[k] is not None and all_res['config'][str(i_run)][k] != tmp[k]:
                        print_this = False
                        break
                if print_this:
                    break
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
        #cur_label = f"{config['sampling_type']}:dp_C={config['dp_C']},dp_sigma={config['dp_sigma']},n_steps={config['n_steps']}, batch_size={config['batch_size']}, lr={config['learning_rate']}"
        #cur_label = f"{config['sampling_type']}:dp_C={config['dp_C']},dp_sigma={config['dp_sigma']},n_steps={config['n_steps']}, batch_size={config['batch_size']}, lr={config['learning_rate']}"
        
        #else:
        #    cur_label = f"{config['sampling_type']}:dp_C=None,dp_sigma=None,n_steps={config['n_steps']}"

        x = np.linspace(1,config['n_global_updates'],config['n_global_updates'])
        axs[0].errorbar(x, all_res['validation_res'][str(i_run)]['acc'].mean(-1), 
                yerr= 2*all_res['validation_res'][str(i_run)]['acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label, 
                color=colors[i_line%len(colors)])
        axs[1].errorbar(x, all_res['validation_res'][str(i_run)]['logl'].mean(-1), 
                yerr= 2*all_res['validation_res'][str(i_run)]['logl'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label,
                color=colors[i_line%len(colors)]
                )
        # add baseline if available
        if i_line == 0:
            try:
                axs[0].hlines(baseline_acc, x[0],x[-1], color='Gray',linestyle=':')
            except:
                pass

        axs[0].set_ylabel('Acc')
        axs[0].set_xlabel('Global communications')
        axs[0].legend()
        axs[0].grid(b=True, which='major', axis='both')
        axs[1].set_ylabel('Logl')
        axs[1].set_xlabel('Global communications')
        axs[1].legend()
        axs[1].grid(b=True, which='major', axis='both')

        plt.suptitle(f"{dataset_name} dataset mean with 2*SEM over {config['n_rng_seeds']} runs" + cur_title)


    if to_disk:
        plt.savefig(fig_folder+fig_name)
        fig_folder
    
    else:
        plt.show()








