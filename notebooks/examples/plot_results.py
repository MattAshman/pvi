
from collections import OrderedDict as OD
from pathlib import Path
import sys


import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from matplotlib import pyplot as plt
import numpy as np


runs_to_plot = [1,2]
main_folder = 'dp_pvi_runs/'

# where to save all random plots
fig_folder = 'res_plots/'

# default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


baseline_acc = 0.761 # for adult



all_res = OD()
all_res['config'] = OD()
all_res['client_train_res'] = OD()
all_res['train_res'] = OD()
all_res['validation_res'] = OD()

jsonpickle_numpy.register_handlers()
for i_run in runs_to_plot:
    #print(i_run)
    filename = main_folder + str(i_run) + '/config.json'
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

    #sys.exit()

    filename = main_folder + str(i_run) + '/info.json'
    with open(filename, 'r') as f:
        apu = f.read()
    apu = jsonpickle.unpickler.decode(apu)
    #print(apu)
    #for k in apu:
    #    print(k)
    
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
        #print(apu[f'client_train_res_seed{i_seed}'][''].shape)
        #print(apu[f'client_train_res_seed{i_seed}'].shape)
        #print(len(apu[f'validation_res_seed{i_seed}'] ))
        #for k in apu[f'validation_res_seed{i_seed}']:
        #    print(k)

        #print(apu[f'validation_res_seed{i_seed}']['acc'].shape )
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

    #sys.exit()


# plot comparisons between given runs: all in same plot

plot_comparisons = True
if plot_comparisons:
    fig, axs = plt.subplots(2, figsize=(10,10))

    for i_run in runs_to_plot:
        config = all_res['config'][str(i_run)]
        x = np.linspace(1,config['n_global_updates'],config['n_global_updates'])
        axs[0].errorbar(x, all_res['validation_res'][str(i_run)]['acc'].mean(-1), 
                yerr= 2*all_res['validation_res'][str(i_run)]['acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=f"noise={config['dp_sigma']}"
                )
        axs[1].errorbar(x, all_res['validation_res'][str(i_run)]['logl'].mean(-1), 
                yerr= 2*all_res['validation_res'][str(i_run)]['logl'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=f"noise={config['dp_sigma']}"
                )
        axs[0].set_ylabel('Acc')
        axs[0].set_xlabel('Global communications')
        axs[0].legend()
        axs[0].grid()
        axs[1].set_ylabel('Logl')
        axs[1].set_xlabel('Global communications')
        axs[1].legend()
        axs[1].grid()

        plt.suptitle(f"Adult dataset mean with 2*SEM over {config['n_rng_seeds']} runs: DP-SGD={config['use_dpsgd']}, noise sigma={config['dp_sigma']}, data balance=({config['data_bal_rho']},{config['data_bal_kappa']}), sampling={config['sampling_type']}\nclients={config['clients']}, local steps={config['n_steps']}, damping factor={config['damping_factor']}")



    plt.show()








