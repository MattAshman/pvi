
from collections import OrderedDict as OD
from pathlib import Path
import sys


import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from matplotlib import pyplot as plt
import numpy as np



# numbers of runs to plot
#runs_to_plot = np.linspace(1,36,36,dtype=int)#[1]
#runs_to_plot = np.linspace(1,54,54,dtype=int)#[1]
#runs_to_plot = np.linspace(1,84,84,dtype=int)#[1]
#runs_to_plot = np.linspace(77,149,73,dtype=int)#[1]
#runs_to_plot = np.linspace(1,216,216,dtype=int)#[1]
#runs_to_plot = np.linspace(1,72,72,dtype=int)#[1]
#runs_to_plot = np.linspace(49,210,162,dtype=int)#[1]
#print(runs_to_plot)

# main folder where the res are
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/hfa_adult_dp_100clients_runs/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dpsgd_adult_dp_100clients_runs/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dpsgd_adult_dp_200clients_runs/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dpsgd_adult_dp_200clients_eps05_runs/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/dpsgd_adult_dp_200clients_eps02_10global_runs/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/lfa_adult_dp_200clients_eps02_runs/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/lfa_adult_dp_200clients_eps02_10global_runs/'

### new 200 client 5 seed runs:
# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_200clients_5seeds_eps02_runs/'
#runs_to_plot = np.linspace(1,8,8,dtype=int)#[1] # all bals
#runs_to_plot = np.linspace(9,10,2,dtype=int)#[1] # puuttuva bal(0,0)


# LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_200clients_5seeds_eps02_runs/'
#runs_to_plot = np.linspace(1,3,3,dtype=int)#[1] # all bals
#runs_to_plot = np.linspace(4,7,4,dtype=int)#[1] # bal=(.7,-3) new runs

# LOCAL PVI
main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_200clients_5seeds_eps02_runs/'
runs_to_plot = np.linspace(1,17,17,dtype=int)#[1] # all bals



### mimic3 runs:
# nondp
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_nondp_5clients_runs/'
#runs_to_plot = np.linspace(1,288,288,dtype=int)#[1]
#runs_to_plot = np.linspace(289,324,36,dtype=int)#[1]

# dpsgd mimic
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_dpsgd_5clients_eps1_runs/'
#runs_to_plot = np.linspace(181,288,108,dtype=int)#[1]
#runs_to_plot = np.linspace(253,288,36,dtype=int)#[1]

# mimic nondp bnn res
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_bnn_nondp_5clients_runs/'
# nondp with seq sampling
#runs_to_plot = np.linspace(1,108,108,dtype=int)#[1]
# nondp with SWOR and clipping
#runs_to_plot = np.linspace(109,387,279,dtype=int)#[1]

# nondp test for checking that stuff still works:
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_testi/'
#runs_to_plot = np.linspace(1,4,4,dtype=int)#[1]

## 1 layer bnn, dpsgd eps1
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_bnn_dpsgd_5clients_eps1_runs/'
# hidden units=100
#runs_to_plot = np.linspace(1,24,24,dtype=int)#[1]
#runs_to_plot = np.linspace(25,48,24,dtype=int)#[1]
#runs_to_plot = np.linspace(49,72,24,dtype=int)#[1]
#runs_to_plot = np.linspace(127,132,6,dtype=int)#[1] # 100 global steps
#runs_to_plot = np.linspace(133,134,2,dtype=int)#[1] # 400 global steps
#runs_to_plot = np.linspace(135,140,6,dtype=int)#[1] # 800 global steps
# hidden units=50
#runs_to_plot = np.linspace(73,126,54,dtype=int)#[1]


## 1 layer bnn, dpsgd eps2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_bnn_dpsgd_5clients_eps2_runs/'
#runs_to_plot = np.linspace(1,54,54,dtype=int)#[1]
#runs_to_plot = np.linspace(55,72,18,dtype=int)#[1]
#runs_to_plot = np.linspace(73,90,18,dtype=int)#[1]
#runs_to_plot = np.linspace(91,107,17,dtype=int)#[1] # 80 global steps
#runs_to_plot = np.linspace(108,119,12,dtype=int)#[1] # 100 global steps
#runs_to_plot = np.linspace(120,125,6,dtype=int)#[1] # 200 global steps
#runs_to_plot = np.linspace(126,129,4,dtype=int)#[1] # 400 global steps
#runs_to_plot = np.linspace(130,139,10,dtype=int)#[1] # 800 global steps
#runs_to_plot = np.linspace(140,145,6,dtype=int)#[1] # 600 global steps

## 2layer bnn, nondp:
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_2bnn_nondp_5clients_runs/'
#runs_to_plot = np.linspace(1,24,24,dtype=int)#[1] # 100 global, no clipping, seq sampling
## 2layer bnn, dpsgd eps2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_2bnn_dpsgd_5clients_eps2_runs/'
#runs_to_plot = np.linspace(1,12,12,dtype=int)#[1] # 100 global updates, eps2
#runs_to_plot = np.linspace(13,18,6,dtype=int)#[1] 400 global updates, eps2
#runs_to_plot = np.linspace(19,21,3,dtype=int)#[1] 800 global updates, eps2
#runs_to_plot = np.linspace(22,27,6,dtype=int)#[1] 400 global updates 20 steps, eps2

## 200 clients adult testing
# DPSGD:
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_200clients_eps02_testing_runs/'
#runs_to_plot = np.linspace(1,36,36,dtype=int)#[1] # 100 global
#runs_to_plot = np.linspace(37,72,36,dtype=int)#[1] # 400 global
#runs_to_plot = np.linspace(73,108,36,dtype=int)#[1] # 40 global
#runs_to_plot = np.linspace(109,144,36,dtype=int)#[1] # 200 global
#runs_to_plot = [145] # 300 global

# LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_200clients_eps02_testing_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int)#[1] # 10 global init tests
#runs_to_plot = np.linspace(109,180,72,dtype=int)#[1] # 10 global, 20 steps
#runs_to_plot = np.linspace(181,252,72,dtype=int)#[1] # 10 global, 10 steps
#runs_to_plot = np.linspace(253,276,24,dtype=int)#[1] # 20 global, 10 steps

# LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_200clients_eps02_testing_runs/'
#runs_to_plot = np.linspace(1,162,162,dtype=int)#[1] # 10 global, bal data init tests
#runs_to_plot = np.linspace(163,234,72,dtype=int)#[1] # 10 global, bal,  40,80 steps
#runs_to_plot = np.linspace(235,252,18,dtype=int)#[1] # 10 global, bal,  40 steps
#runs_to_plot = np.linspace(253,279,27,dtype=int)#[1] # 20 global, bal,  80 steps, 0bal
#runs_to_plot = np.linspace(280,287,8,dtype=int)#[1] # 20 global, bal,  80 steps, unbalanced




#print(runs_to_plot)
#sys.exit()


# where to save all plots
fig_folder = 'res_plots/'

# baseline models to include: there should be 1 entry for each baseline in every given attribute
# note: not sure if these work currently
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
restrictions['dp_sigma'] = None#[23.15]
restrictions['dp_C'] = None#[1.]
restrictions['n_global_updates'] = None
restrictions['n_steps'] = None#[80]
restrictions['batch_size'] = None#[5]
restrictions['sampling_frac_q'] = None#[.04]
restrictions['learning_rate'] = None#[5e-3]
restrictions['damping_factor'] = None#[.1]

# possible balance settings: (0,0), (.7,-3), (.75,.95)
restrictions['data_bal_rho'] = [.0]
restrictions['data_bal_kappa'] = [.0]


# save to disk (or just show)
to_disk = 0

plot_test_error = 1 # if 0 plot training err instead
plot_legends = 1

#dataset_name = 'mimic3'
dataset_name = 'adult'
#dataset_name = 'mushroom'
#dataset_name = 'credit'
#dataset_name = 'bank'

#fig_name = "{}_dpsgd_200clients_bal(0,0)_test4.pdf".format(dataset_name)
#fig_name = "{}_dpsgd_200clients_bal(07,-3)_test4.pdf".format(dataset_name)
#fig_name = "{}_dpsgd_200clients_bal(075,095)_test4.pdf".format(dataset_name)

# name for the current plot
'''
#fig_name = "{}_dpsgd_200clients_eps02_best_bal({},{}).pdf".format(dataset_name, 
fig_name = "{}_dpsgd_200clients_eps02_best_bal({},{})_all.pdf".format(dataset_name, 
        #restrictions['dp_C'], 
        restrictions['data_bal_rho'],restrictions['data_bal_kappa'], 
        #restrictions['batch_size'],restrictions['damping_factor'] 
        )
#'''
#fig_name = "{}_dpsgd_5clients_eps1_all_train.pdf".format(dataset_name)
fig_name = "{}_dpsgd_5clients_eps1_all.pdf".format(dataset_name)
#fig_name = "{}_2BNN_nondp_5clients_nondp_all.pdf".format(dataset_name)
'''
#fig_name = "{}_lfa_200clients_eps02_frac_best_bal({},{}).pdf".format(dataset_name, 
#fig_name = "{}_lfa_200clients_eps02_frac_best_bal({},{})_all.pdf".format(dataset_name, 
        #restrictions['dp_C'], 
        restrictions['data_bal_rho'],restrictions['data_bal_kappa'], 
        #restrictions['batch_size'],restrictions['damping_factor'] 
        )
#'''

# set to None to skip:
if dataset_name == 'mimic3':
    ylims= ((.87,.9),(-.55,-.26))
    if plot_test_error:
        baseline_acc = 0.8844
        baseline_logl = None
    else:
        baseline_acc = 0.8647
        baseline_logl = None
elif dataset_name == 'adult':
    ylims= ((.7,.9),(-.6,-.26))
    if plot_test_error:
        baseline_acc = 0.761
        baseline_logl = None
    else:
        baseline_acc = None
        baseline_logl = None

# set vars to add to title: key=variable name, value=var name in fig title
add_to_title = {}
#add_to_title['data_bal_rho'] = 'rho'
#add_to_title['data_bal_kappa'] = 'kappa'
add_to_title['clients'] = 'clients'
add_to_title['dp_mode'] = 'dp mode'
#add_to_title[''] = ''


# set labels to add: key=variable name, value=label name in fig
add_labels = {}
#add_labels['sampling_type'] = 'sampling'
add_labels['dp_C'] = 'C'
add_labels['dp_sigma'] = 'sigma'
add_labels['n_steps'] = 'steps'
#add_labels['batch_size'] = 'b'
add_labels['sampling_frac_q'] = 'q'
add_labels['learning_rate'] = 'lr'
add_labels['damping_factor'] = 'damping'
#add_labels['privacy_calculated'] = 'DP epochs'
#add_labels['data_bal_rho'] = 'rho'
#add_labels['data_bal_kappa'] = 'kappa'

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

if include_baselines:
    # note: not sure if this works anymore
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
    #print(apu)

    filename = main_folder + str(i_run) + '/info.json'
    with open(filename, 'r') as f:
        apu = f.read()
    try:
        apu = jsonpickle.unpickler.decode(apu)
    except:
        print(f'error in JSON decoding in run {i_run}')
        with open(filename, 'r') as f:
            apu = f.read()
        print('results from file: {}\n{}'.format(filename,apu))
        sys.exit()


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
    posneg_measures = ['avg_prec_score','balanced_acc','f1_score']
    
    all_res['train_res'][str(i_run)] = {}
    all_res['validation_res'][str(i_run)] = {}
    for k in (measures+posneg_measures):
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
            try:
                all_res['client_train_res'][str(i_run)][k][:,:,:,i_seed] = apu[f'client_train_res_seed{i_seed}'][k]
            except KeyError as err:
                print(f'KeyError in run {i_run} (=folder)')
                print(f'got\n{apu}')
                print("config: batch_size={}, jobid={}".format(all_res['config'][str(i_run)]['batch_size'], all_res['config'][str(i_run)]['job_id'] ))
                raise err

        for k in measures:
            all_res['train_res'][str(i_run)][k][:,i_seed] = apu[f'train_res_seed{i_seed}'][k]
            all_res['validation_res'][str(i_run)][k][:,i_seed] = apu[f'validation_res_seed{i_seed}'][k]

        # posneg = list of posneg dicts with len=n_global_updates
        # NOTE: need to check format when n_seeds > 1
        for i_global in range(all_res['config'][str(i_run)]['n_global_updates']):
            for k in posneg_measures:
                #print(apu[f"validation_res_seed0"]['posneg'][i_global][k])
                all_res['train_res'][str(i_run)][k][i_global,i_seed] = apu[f"train_res_seed{i_seed}"]['posneg'][i_global][k]
                all_res['validation_res'][str(i_run)][k][i_global,i_seed] = apu[f"validation_res_seed{i_seed}"]['posneg'][i_global][k]
                #print(len(apu[f"validation_res_seed0"]['posneg']),len(apu[f"train_res_seed0"]['posneg'] ))
                #sys.exit()

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
    best_acc = 0
    best_logl = -1e6
    best_confs = [None,None]

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
    #fig, axs = plt.subplots(2, figsize=(10,10))
    fig, axs = plt.subplots(2,2, figsize=(10,10))

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
                if k == 'dp_sigma':
                    cur_label += ", {}={:.1f}".format(add_labels[k], config[k])
                else:
                    cur_label += ", {}={}".format(add_labels[k], config[k])
            except:
                if k == 'dp_sigma':
                    cur_label = "{}={:.1f}".format(add_labels[k], config[k])
                else:
                    cur_label = "{}={}".format(add_labels[k], config[k])
        #cur_label = f"{config['sampling_type']}:dp_C={config['dp_C']},dp_sigma={config['dp_sigma']},n_steps={config['n_steps']}, batch_size={config['batch_size']}, lr={config['learning_rate']}"
        #cur_label = f"{config['sampling_type']}:dp_C={config['dp_C']},dp_sigma={config['dp_sigma']},n_steps={config['n_steps']}, batch_size={config['batch_size']}, lr={config['learning_rate']}"
        
        #else:
        #    cur_label = f"{config['sampling_type']}:dp_C=None,dp_sigma=None,n_steps={config['n_steps']}"
        if np.amax(all_res['validation_res'][str(i_run)]['acc'].mean(-1) ) > best_acc:
            best_acc = np.amax(all_res['validation_res'][str(i_run)]['acc'].mean(-1))
            best_confs[0] = config
        if np.amax(all_res['validation_res'][str(i_run)]['logl'].mean(-1)) > best_logl:
            best_logl = np.amax(all_res['validation_res'][str(i_run)]['logl'].mean(-1))
            best_confs[1] = config

        x = np.linspace(1,config['n_global_updates'],config['n_global_updates'])
        # testing error plot
        if plot_test_error:
            axs[0,0].errorbar(x, all_res['validation_res'][str(i_run)]['acc'].mean(-1), 
                    yerr= 2*all_res['validation_res'][str(i_run)]['acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label, 
                    color=colors[i_line%len(colors)])
            axs[1,0].errorbar(x, all_res['validation_res'][str(i_run)]['logl'].mean(-1), 
                    yerr= 2*all_res['validation_res'][str(i_run)]['logl'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label,
                    color=colors[i_line%len(colors)]
                    )

            #posneg_measures = ['avg_prec_score','balanced_acc','f1_score']
            axs[0,1].errorbar(x, all_res['validation_res'][str(i_run)]['balanced_acc'].mean(-1), 
                    yerr= 2*all_res['validation_res'][str(i_run)]['balanced_acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label,
                    color=colors[i_line%len(colors)]
                    )
            axs[1,1].errorbar(x, all_res['validation_res'][str(i_run)]['avg_prec_score'].mean(-1), 
                    yerr= 2*all_res['validation_res'][str(i_run)]['avg_prec_score'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label,
                    color=colors[i_line%len(colors)]
                    )
        # training error plots
        else:
            axs[0,0].errorbar(x, all_res['train_res'][str(i_run)]['acc'].mean(-1), 
                    yerr= 2*all_res['train_res'][str(i_run)]['acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label, 
                    color=colors[i_line%len(colors)])
            axs[1,0].errorbar(x, all_res['train_res'][str(i_run)]['logl'].mean(-1), 
                    yerr= 2*all_res['train_res'][str(i_run)]['logl'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label,
                    color=colors[i_line%len(colors)]
                    )
            axs[0,1].errorbar(x, all_res['train_res'][str(i_run)]['balanced_acc'].mean(-1), 
                    yerr= 2*all_res['train_res'][str(i_run)]['balanced_acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label,
                    color=colors[i_line%len(colors)]
                    )
            axs[1,1].errorbar(x, all_res['train_res'][str(i_run)]['avg_prec_score'].mean(-1), 
                    yerr= 2*all_res['train_res'][str(i_run)]['avg_prec_score'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label,
                    color=colors[i_line%len(colors)]
                    )
        # add baseline if available
        if i_line == 0:
            try:
                axs[0,0].hlines(baseline_acc, x[0],x[-1], color='black',linestyle=':')
            except:
                pass
            try:
                axs[1,0].hlines(baseline_logl, x[0],x[-1], color='black',linestyle=':')
            except:
                pass
        

        axs[0,0].set_ylabel('Acc')
        axs[0,0].set_xlabel('Global communications')
        if plot_legends:
            axs[0,0].legend(loc='lower right')
            axs[1,0].legend(loc='lower right')
        if ylims[0] is not None:
            axs[0,0].set_ylim(ylims[0])
        if ylims[1] is not None:
            axs[1,0].set_ylim(ylims[1])

        axs[1,0].set_ylabel('Logl')
        axs[1,0].set_xlabel('Global communications')
        axs[0,1].set_ylabel('Balanced acc')
        axs[0,1].set_xlabel('Global communications')
        axs[1,1].set_ylabel('Avg precision score')
        axs[1,1].set_xlabel('Global communications')
        for i in range(2):
            for ii in range(2):
                axs[i,ii].grid(b=True, which='major', axis='both')

        if plot_test_error:
            plt.suptitle(f"{dataset_name} dataset mean with 2*SEM over {config['n_rng_seeds']} runs" + cur_title)
        else:
            plt.suptitle(f"{dataset_name} dataset training err mean with 2*SEM over {config['n_rng_seeds']} runs" + cur_title)

    print(f'\nBest acc: {best_acc:.5f} found with config:\n{best_confs[0]}')
    if best_confs[1] == best_confs[0]:
        best_confs[1] = None
    print(f'\nBest logl: {best_logl:.5f} found with config:\n{best_confs[1]}')

    if to_disk:
        plt.savefig(fig_folder+fig_name)
        fig_folder
    
    else:
        plt.show()


    """ # ROC curve plotting and checks
    print(f'1s in data: {np.sum(y==1)}, 0s in data: {np.sum(y==0)}')
    #'''
    #print(f'len y={len(y)} =? {np.sum([v[n_points//2] for k,v in posneg.items()])}')
    for k,v in posneg.items():
        print(f'{k} = {v}')
    print(f"TP+FP: {posneg['TP'][n_points//2]+posneg['FP'][n_points//2]}, TN+FN: {posneg['TN'][n_points//2]+posneg['FN'][n_points//2]}")
    print(f"TP+TN: {posneg['TP'][n_points//2]+posneg['TN'][n_points//2]}, FP+FN: {posneg['FP'][n_points//2]+posneg['FN'][n_points//2]}")
    print(f"acc: { (posneg['TP'][n_points//2]+posneg['TN'][n_points//2])/len(y)}")
    #'''

    to_plot = np.zeros((2,n_points)) # x,y for plotting
    for i_thr,thr in enumerate(tmp):
        to_plot[0,i_thr] = posneg['TN'][i_thr]/(posneg['TN'][i_thr]+posneg['FP'][i_thr])  # TNR
        to_plot[1,i_thr] = posneg['TP'][i_thr]/(posneg['TP'][i_thr]+posneg['FN'][i_thr])  # TPR


    print(to_plot)
    plt.plot(tmp,to_plot[0,:] )
    plt.plot(tmp,to_plot[1,:] )
    plt.show()
    
    plt.plot(1-to_plot[0,:],to_plot[1,:] )
    plt.show()

    sys.exit()
    """




