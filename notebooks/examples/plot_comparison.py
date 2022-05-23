
from collections import OrderedDict as OD
from pathlib import Path
import sys


import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from matplotlib import pyplot as plt
import numpy as np


### new 200 client 5 seed runs:
# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_200clients_5seeds_eps02_runs/'
#runs_to_plot = np.linspace(1,17,17,dtype=int)#[1] # kaikki yhdessä, myös korjaukset


# LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_200clients_5seeds_eps02_runs/'
#runs_to_plot = np.linspace(1,3,3,dtype=int)#[1] # all bals
#runs_to_plot = np.linspace(4,7,4,dtype=int)#[1] # bal=(.7,-3) new runs

# LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_200clients_5seeds_eps02_runs/'
#runs_to_plot = np.linspace(1,19,19,dtype=int)#[1] # all bals



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
#runs_to_plot = np.linspace(1,54,54,dtype=int)#[1] # 1)
#runs_to_plot = np.linspace(55,72,18,dtype=int)#[1] # 2)
#runs_to_plot = np.linspace(73,90,18,dtype=int)#[1] # 3)
#runs_to_plot = np.linspace(91,107,17,dtype=int)#[1] # 4) 80 global steps
#runs_to_plot = np.linspace(108,119,12,dtype=int)#[1] # 5) 100 global steps
#runs_to_plot = np.linspace(120,125,6,dtype=int)#[1] # 6) 200 global steps
#runs_to_plot = np.linspace(126,129,4,dtype=int)#[1] # 7) 400 global steps
#runs_to_plot = np.linspace(130,139,10,dtype=int)#[1] # 8) 800 global steps
#runs_to_plot = np.linspace(140,145,6,dtype=int)#[1] # 600 global steps

## 2layer bnn, nondp:
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_2bnn_nondp_5clients_runs/'
#runs_to_plot = np.linspace(1,24,24,dtype=int)#[1] # 100 global, no clipping, seq sampling
## 2layer bnn, dpsgd eps2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_2bnn_dpsgd_5clients_eps2_runs/'
#runs_to_plot = np.linspace(1,12,12,dtype=int)#[1] # 1) 100 global updates, eps2
#runs_to_plot = np.linspace(13,18,6,dtype=int)#[1] 2) 400 global updates, eps2
#runs_to_plot = np.linspace(19,21,3,dtype=int)#[1] 3) 800 global updates, eps2
#runs_to_plot = np.linspace(22,27,6,dtype=int)#[1] 4) 400 global updates 20 steps, eps2

## new MIMIC 5 seeds runs:
# 1 layer bnn, dpsgd eps1
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_bnn_dpsgd_5clients_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,3,3,dtype=int)#[1] # 1)

# 1 layer bnn, dpsgd eps2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_bnn_dpsgd_5clients_5seeds_eps2_runs/'
#runs_to_plot = [1] #np.linspace(1,1,1,dtype=int)#[1] # 1)

# 2 layer bnn, dpsgd eps2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_2bnn_dpsgd_5clients_5seeds_eps2_runs/'
#runs_to_plot = [1] #np.linspace(1,1,1,dtype=int)#[1] # 1)


## MNIST testing
# balanced data, b=1, eps=2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_dpsgd_bal_10clients_b1_eps2_runs/'
#runs_to_plot = np.linspace(1,54,54,dtype=int)#[1] # 1) # 200 globals
#runs_to_plot = np.linspace(55,59,5,dtype=int)#[1] # 2) # 400 globals

# balanced data, b=5, eps=2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_dpsgd_bal_10clients_b5_eps2_runs/'
#runs_to_plot = np.linspace(1,18,18,dtype=int)#[1] # 1) # 200-400 globals

# unbalanced data, b=1, eps=2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_dpsgd_unbal_100clients_b1_eps2_runs/'
#runs_to_plot = np.linspace(1,25,25,dtype=int)#[1] # 1) # 200-400 globals


# NONDP MNIST: balanced data, b=1
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_nondp_10clients_b1_runs/'
#runs_to_plot = np.linspace(1,10,10,dtype=int)#[1] # 1) # 200 globals

# all runs to plot
# NOTE: olisi ehkö parempi muuttaa tämä pelkäksi compare eps scriptiksi, sit kerää suoraan eri eps toisiaan vastaavat datapisteet yhteen, jolloin ei tarvitse yrittää yhdistellä plottausvaiheessa

all_runs = [
            {'folder' : '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_bnn_dpsgd_5clients_5seeds_eps1_runs/', 
                'run_id' : 1, 
                'label' : '1BNN, eps=1',
                'eps' : 1.},

            {'folder' : '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_bnn_dpsgd_5clients_5seeds_eps2_runs/', 
                'run_id' : 1, 
                'label' : '1BNN, eps=2',
                'eps' : 2.},
            {'folder' : '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_2bnn_dpsgd_5clients_5seeds_eps2_runs/', 
                'run_id' : 1, 
                'label' : '2BNN, eps=2',
                'eps' : 2.},
            ]
all_eps = [1.,2.]


#print(runs_to_plot)
#sys.exit()


# where to save all plots
fig_folder = 'res_plots/'

restrictions = OD()
restrictions['dp_sigma'] = None#[23.15]
restrictions['dp_C'] = None#[5.]
restrictions['n_global_updates'] = None#[400]
restrictions['n_steps'] = None#[20]
restrictions['batch_size'] = None#[5]
restrictions['sampling_frac_q'] = None#[.04]
restrictions['learning_rate'] = None#[1e-3]
restrictions['damping_factor'] = None#[.4]

# possible balance settings: (0,0), (.7,-3), (.75,.95)
restrictions['data_bal_rho'] = [.0]
restrictions['data_bal_kappa'] = [.0]


# save to disk (or just show)
to_disk = 0

plot_test_error = 1 # if 0 plot training err instead
plot_legends = 1


#dataset_name = 'mnist'
dataset_name = 'mimic3'
#dataset_name = 'adult'
#dataset_name = 'mushroom'
#dataset_name = 'credit'
#dataset_name = 'bank'


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
'''
fig_name = "{}_dpsgd_200clients_5seeds_eps02_bal({},{})_best.pdf".format(
        dataset_name, restrictions['data_bal_rho'][0], restrictions['data_bal_kappa'][0]
        ) #'''
#fig_name = "{}_2BNN_dpsgd_5clients_5seeds_eps2.pdf".format(dataset_name)
fig_name = "{}_1BNN_dpsgd_10clients_1seeds_bal_b1_eps2.pdf".format(dataset_name)
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
    #ylims= ((.87,.9),(-.55,-.26), None, None)
    ylims= (None,None, None, None)
    if plot_test_error:
        baseline_acc = 0.8844
        baseline_logl = None
    else:
        baseline_acc = 0.8647
        baseline_logl = None
elif dataset_name == 'adult':
    #ylims= ((.7,.9),(-.6,-.26),(.5,.8),(.5,.8)) # acc, logl, bal acc, avg prec score
    ylims= ((.7,.9),(-.6,-.26),(-.05,1.05),(.5,.8)) # acc, logl, ROC, avg prec score
    if plot_test_error:
        baseline_acc = 0.761
        baseline_logl = None
    else:
        baseline_acc = None
        baseline_logl = None
elif dataset_name == 'mnist':
    #ylims= ((.7,.95),(-.8,-.3), None, None)
    ylims = (None,None,None,None)
    if plot_test_error:
        baseline_acc = None#0.8844
        baseline_logl = None
    else:
        baseline_acc = None#0.8647
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
#add_labels['dp_C'] = 'C'
#add_labels['dp_sigma'] = 'sigma'
add_labels['n_steps'] = 'steps'
#add_labels['batch_size'] = 'b'
add_labels['sampling_frac_q'] = 'q'
add_labels['learning_rate'] = 'lr'
add_labels['damping_factor'] = 'damping'
#add_labels['privacy_calculated'] = 'DP epochs'
#add_labels['data_bal_rho'] = 'rho'
#add_labels['data_bal_kappa'] = 'kappa'



#####################
# set baselines

baseline_cols = ['silver', 'dimgrey', 'whitesmoke']

baseline_folders = []
baseline_runs_to_plot = []
baseline_names = []

# adult nondp global vi
if dataset_name == 'adult':
    baseline_folders.append( '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/baselines/adult_nondp_global_vi_5seeds_runs/')
    baseline_runs_to_plot.append([1]) #np.linspace(1,1,1,dtype=int)#[1] # 1)
    baseline_names.append(['non-DP global VI'])



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

all_baselines = OD()
all_baselines['config'] = OD()
all_baselines['client_train_res'] = OD()
all_baselines['train_res'] = OD()
all_baselines['validation_res'] = OD()


jsonpickle_numpy.register_handlers()
failed_runs = []


def read_config(filename, failed_runs):
    try:
        with open(filename, 'r') as f:
            apu = f.read()
    except FileNotFoundError as err:
        print(f"Can't open file {filename}! Skipping")
        failed_runs.append(i_run)
        return None
    apu = jsonpickle.unpickler.decode(apu)
    #print(apu)
    return apu, failed_runs


def read_results(filename, filename_bck):
    try:
        with open(filename, 'r') as f:
            apu = f.read()
        try:
            apu = jsonpickle.unpickler.decode(apu)
        except:
            print(f'error in JSON decoding in run {filename}')
            with open(filename, 'r') as f:
                apu = f.read()
            print('results from file: {}\n{}'.format(filename,apu))
            sys.exit()
    except FileNotFoundError as err:
        import json
        with open(filename_bck, 'r') as f:
            apu = f.read()
            apu = json.loads(apu)
        try:
            #apu = jsonpickle.unpickler.decode(apu)
            #print(apu)
            apu = jsonpickle.decode(apu, keys=True)
            #print(apu)
            #print('at bck decode')
        except:
            print(f'error in JSON decoding in {filename_bck}')
            #with open(filename, 'r') as f:
            #    apu = f.read()
            #print('results from file: {}\n{}'.format(filename,apu))
            sys.exit()
    return apu

def format_results(apu, run_id, client_measures, all_res):

    all_res['client_train_res'][run_id] = {}
    #print(all_res['config'][run_id])
    for k in client_measures:
        all_res['client_train_res'][run_id][k] = np.zeros((
            all_res['config'][run_id]['clients'],  
            all_res['config'][run_id]['n_global_updates'],  
            all_res['config'][run_id]['n_steps'],  
            all_res['config'][run_id]['n_rng_seeds']
            ))
    measures = ['acc','logl']
    posneg_measures = ['avg_prec_score','balanced_acc','f1_score']
    
    all_res['train_res'][run_id] = {}
    all_res['validation_res'][run_id] = {}
    for k in (measures+posneg_measures):
        all_res['train_res'][run_id][k] = np.zeros((
            all_res['config'][run_id]['n_global_updates'],  
            all_res['config'][run_id]['n_rng_seeds']
            ))
        all_res['validation_res'][run_id][k] = np.zeros((
            all_res['config'][run_id]['n_global_updates'],  
            all_res['config'][run_id]['n_rng_seeds']
            ))
        all_res['train_res'][run_id]['best_'+k] = np.zeros((
            all_res['config'][run_id]['n_rng_seeds']
            ))
        all_res['validation_res'][run_id]['best_'+k] = np.zeros((
            all_res['config'][run_id]['n_rng_seeds']
            ))

    # does this work with sampling=seq?
    if dataset_name != 'mnist':
        # for plotting ROC curve for max logl global update, one for each seed
        all_res['validation_res'][run_id]['TPR'] = np.zeros((
            all_res['config'][run_id]['n_rng_seeds'],
            apu['validation_res_seed0']['posneg'][0]['n_points']
            ))
        all_res['validation_res'][run_id]['TNR'] = np.zeros((
            all_res['config'][run_id]['n_rng_seeds'],
            apu['validation_res_seed0']['posneg'][0]['n_points']
            ))
        all_res['validation_res'][run_id]['ROC_thresholds'] = np.linspace(0,1,apu['validation_res_seed0']['posneg'][0]['n_points'])


    for i_seed in range(all_res['config'][run_id]['n_rng_seeds']):
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
                all_res['client_train_res'][run_id][k][:,:,:,i_seed] = apu[f'client_train_res_seed{i_seed}'][k]
            except KeyError as err:
                print(f'KeyError in run {i_run} (=folder)')
                print(f'got\n{apu}')
                print("config: batch_size={}, jobid={}".format(all_res['config'][run_id]['batch_size'], all_res['config'][run_id]['job_id'] ))
                raise err

        for k in measures:
            all_res['train_res'][run_id][k][:,i_seed] = apu[f'train_res_seed{i_seed}'][k]
            all_res['validation_res'][run_id][k][:,i_seed] = apu[f'validation_res_seed{i_seed}'][k]

            all_res['train_res'][run_id]['best_'+k][i_seed] = np.amax(all_res['train_res'][run_id][k][:,i_seed])
            all_res['validation_res'][run_id]['best_'+k][i_seed] = np.amax(all_res['validation_res'][run_id][k][:,i_seed])

        # calculate true positive and true negative rates at global update with best logl
        #print( all_res['validation_res'][run_id]['logl'][:,i_seed] )
        best_global_logl = np.argmax( all_res['validation_res'][run_id]['logl'][:,i_seed] )
        #print(all_res['validation_res'][run_id]['logl'][:,i_seed][best_global_logl])
        #print(all_res['validation_res'][run_id]['best_logl'][i_seed])
        # all_res['validation_res'][run_id] = {}
        #sys.exit()

        if dataset_name != 'mnist':
            all_res['validation_res'][run_id]['TPR'][i_seed,:] = apu[f"validation_res_seed{i_seed}"]['posneg'][best_global_logl]['TP']/( apu[f"validation_res_seed{i_seed}"]['posneg'][best_global_logl]['TP'] + apu[f"validation_res_seed{i_seed}"]['posneg'][best_global_logl]['FN'])

            all_res['validation_res'][run_id]['TNR'][i_seed,:] = apu[f"validation_res_seed{i_seed}"]['posneg'][best_global_logl]['TN']/( apu[f"validation_res_seed{i_seed}"]['posneg'][best_global_logl]['TN'] + apu[f"validation_res_seed{i_seed}"]['posneg'][best_global_logl]['FP'])

            # posneg = list of posneg dicts with len=n_global_updates
            # NOTE: need to check format when n_seeds > 1
            for i_global in range(all_res['config'][run_id]['n_global_updates']):
                for k in posneg_measures:
                    #print(apu[f"validation_res_seed0"]['posneg'][i_global][k])
                    all_res['train_res'][run_id][k][i_global,i_seed] = apu[f"train_res_seed{i_seed}"]['posneg'][i_global][k]
                    all_res['validation_res'][run_id][k][i_global,i_seed] = apu[f"validation_res_seed{i_seed}"]['posneg'][i_global][k]
                    #print(len(apu[f"validation_res_seed0"]['posneg']),len(apu[f"train_res_seed0"]['posneg'] ))
                    #sys.exit()

            for k in posneg_measures:
                all_res['train_res'][run_id]['best_'+k][i_seed] = np.amax(all_res['train_res'][run_id][k][:,i_seed])
                all_res['validation_res'][run_id]['best_'+k][i_seed] = np.amax(all_res['validation_res'][run_id][k][:,i_seed])



"""
all_runs = [
            {'folder' : '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_bnn_dpsgd_5clients_5seeds_eps1_runs/', 
                'runs' : [1], 
                'label' : '1BNN, eps=1'},"""

for running_id, run_pars in enumerate(all_runs):
    run_id = str(run_pars['run_id'])
    print(f"run {run_id} in {run_pars['label']}")
    filename = run_pars['folder'] + run_id + '/config.json'
    #print(f'trying {filename}')
    tmp = read_config(filename, failed_runs)
    if int(run_id) in failed_runs:
        raise ValueError(f"failed run: {run_pars['folder']}\trun {run_id}. Aborting!")

    all_res['config'][str(running_id)] = tmp[0]
    #print(all_res['config'][running_id])
    #sys.exit()

    # try opening sacred records, if missing open manual bck instead
    filename = run_pars['folder'] + run_id + '/info.json'
    filename_bck = run_pars['folder'] + run_id + '/info_bck.json'
    apu = read_results(filename, filename_bck)

    #print(apu['validation_res_seed0']['logl'].shape)
    #sys.exit()

    #for k in apu:
    #    print(k)
    #sys.exit()
    
    # format results for plotting
    client_measures = ['elbo','kl','logl']
    format_results(apu, str(running_id), client_measures, all_res)



# read baselines
if len(baseline_folders) > 0:
    running_id = 0
    for folder, baseline_name, baseline_ids in zip(baseline_folders, baseline_names, baseline_runs_to_plot):
        for i_run in baseline_ids:
            run_id = str(running_id)
            print(f'baseline run {run_id}')
            filename = folder + str(i_run) + '/config.json'
            #print(f'trying {filename}')
            tmp = read_config(filename, failed_runs)
            all_baselines['config'][run_id] = tmp[0]

            # try opening sacred records, if missing open manual bck instead
            filename = folder + str(i_run) + '/info.json'
            filename_bck = folder + str(i_run) + '/info_bck.json'
            apu = read_results(filename, filename_bck)

            #print(apu)
            #sys.exit()

            #for k in apu:
            #    print(k)
            #sys.exit()
            
            # format results for plotting
            client_measures = ['elbo','kl','logl']
            format_results(apu, run_id, client_measures, all_baselines)

            running_id += 1


# plot comparisons between given runs: all in same plot
best_acc = 0
best_logl = -1e6
best_confs = [None,None]

# check restrictions
#fig, axs = plt.subplots(2, figsize=(10,10))
fig, axs = plt.subplots(2,2, figsize=(10,10))

'''
for k in add_to_title:
    try:
        cur_title += ", {}={}".format(add_to_title[k], all_res['config'][str(list_to_print[0])][k])
    except:
        try:
            cur_title = ": {}={}".format(add_to_title[k], all_res['config'][str(list_to_print[0])][k])
        except:
            cur_title = '''

'''
all_runs = [
            {'folder' : '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic3_bnn_dpsgd_5clients_5seeds_eps1_runs/', 
                'runs' : [1], 
                'label' : '1BNN, eps=1'},'''

for i_run, run_pars in enumerate(all_runs):
    cur_label = run_pars['label']
    config = all_res['config'][str(i_run)]

    '''
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
    '''
    #else:
    #    cur_label = f"{config['sampling_type']}:dp_C=None,dp_sigma=None,n_steps={config['n_steps']}"
    if np.amax(all_res['validation_res'][str(i_run)]['acc'].mean(-1) ) > best_acc:
        best_acc = np.amax(all_res['validation_res'][str(i_run)]['acc'].mean(-1))
        best_confs[0] = config
    if np.amax(all_res['validation_res'][str(i_run)]['logl'].mean(-1)) > best_logl:
        best_logl = np.amax(all_res['validation_res'][str(i_run)]['logl'].mean(-1))
        best_confs[1] = config

    #x = np.linspace(1,config['n_global_updates'],config['n_global_updates'])
    x = all_eps # all eps values here? ei onnistu suoraan näillä, pitäisi rakentaa suoraan sen perusteella

    # testing error plot
    #print( np.amax(all_res['validation_res'][str(i_run)]['acc'].mean(-1)) )
    #sys.exit()
    if plot_test_error:
        axs[0,0].plot(run_pars['eps'], np.amax(all_res['validation_res'][str(i_run)]['acc'].mean(-1)) ,'*', label=run_pars['label'])
        axs[1,0].plot(run_pars['eps'], np.amax(all_res['validation_res'][str(i_run)]['logl'].mean(-1)), '*' , label=run_pars['label'])
        '''
        axs[0,0].errorbar(x, np.amax(all_res['validation_res'][str(i_run)]['acc'].mean(-1)), 
                yerr= 2*all_res['validation_res'][str(i_run)]['acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label, 
                color=colors[i_run%len(colors)])
        axs[1,0].errorbar(x, np.amax(all_res['validation_res'][str(i_run)]['logl'].mean(-1)), 
                yerr= 2*all_res['validation_res'][str(i_run)]['logl'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label,
                color=colors[i_run%len(colors)]
                )'''

        #posneg_measures = ['avg_prec_score','balanced_acc','f1_score']
        # balanced acc
        #'''

        x = np.linspace(1,config['n_global_updates'],config['n_global_updates'])

        if dataset_name == 'mnist':
            axs[0,1].errorbar(x, all_res['validation_res'][str(i_run)]['balanced_acc'].mean(-1), 
                    yerr= 2*all_res['validation_res'][str(i_run)]['balanced_acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label,
                    color=colors[i_run%len(colors)]
                    )
        # avg ROC curve at best logl global update
        else:
            axs[0,1].plot( 1-all_res['validation_res'][str(i_run)]['TNR'].mean(0), all_res['validation_res'][str(i_run)]['TPR'].mean(0), 
                    #yerr= 2*all_res['validation_res'][str(i_run)]['balanced_acc'].std(0)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label,
                    color=colors[i_run%len(colors)]
                    )
        #plt.plot(1-all_res['validation_res'][str(i_run)]['TNR'][i_seed,:], all_res['validation_res'][str(i_run)]['TPR'][i_seed,:] )

        axs[1,1].errorbar(x, all_res['validation_res'][str(i_run)]['avg_prec_score'].mean(-1), 
                yerr= 2*all_res['validation_res'][str(i_run)]['avg_prec_score'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label,
                color=colors[i_run%len(colors)]
                )
    # training error plots
    else:
        axs[0,0].errorbar(x, all_res['train_res'][str(i_run)]['acc'].mean(-1), 
                yerr= 2*all_res['train_res'][str(i_run)]['acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label, 
                color=colors[i_run%len(colors)])
        axs[1,0].errorbar(x, all_res['train_res'][str(i_run)]['logl'].mean(-1), 
                yerr= 2*all_res['train_res'][str(i_run)]['logl'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label,
                color=colors[i_run%len(colors)]
                )
        # balanced acc
        #'''
        axs[0,1].errorbar(x, all_res['train_res'][str(i_run)]['balanced_acc'].mean(-1), 
                yerr= 2*all_res['train_res'][str(i_run)]['balanced_acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label,
                color=colors[i_run%len(colors)]
                )
        #'''
        axs[1,1].errorbar(x, all_res['train_res'][str(i_run)]['avg_prec_score'].mean(-1), 
                yerr= 2*all_res['train_res'][str(i_run)]['avg_prec_score'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label,
                color=colors[i_run%len(colors)]
                )

    # add baselines if available
    if i_run == 0:
        try:
            axs[0,0].hlines(baseline_acc, x[0],x[-1], color='black',linestyle=':', label='constant pred.')
        except:
            pass
        try:
            axs[1,0].hlines(baseline_logl, x[0],x[-1], color='black',linestyle=':')
        except:
            pass
    
        if len(baseline_folders) > 0:
            running_id = 0
            for folder, baseline_name, baseline_ids in zip(baseline_folders, baseline_names, baseline_runs_to_plot):
                for i_run in baseline_ids:
                    run_id = str(running_id)
                    baseline_config = all_baselines['config'][run_id]

                    x = np.linspace(1,config['n_global_updates'],config['n_global_updates'])
                    # eli plotataan vain hlines baseline parhaan tuloksen kohdalle?
                    # FIKSAA

                    #all_res['train_res'][run_id]['best_'+k][i_seed] = np.amax(all_res['train_res'][run_id][k][:,i_seed])
                    #all_res['validation_res'][run_id]['best_'+k][i_seed] = np.amax(all_res['validation_res'][run_id][k][:,i_seed])

                    # testing error plot
                    if plot_test_error:
                        axs[0,0].hlines(all_baselines['validation_res'][run_id]['best_acc'],1,config['n_global_updates'], color=baseline_cols[running_id], linestyle='--', label=baseline_name[0])
                        '''
                        axs[0,0].errorbar(x, np.zeros(config['n_global_updates'])+all_baselines['validation_res'][run_id]['best_acc'].mean(), 
                                yerr= 2*all_baselines['validation_res'][run_id]['best_acc'].std()/np.sqrt(baseline_config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                label=baseline_name[0], 
                                color=baseline_cols[running_id], linestyle='--') #'''
                        axs[1,0].hlines(all_baselines['validation_res'][run_id]['best_logl'],1,config['n_global_updates'], color=baseline_cols[running_id], linestyle='--', label=baseline_name[0])
                        '''
                        axs[1,0].errorbar(x, np.zeros(config['n_global_updates'])+all_baselines['validation_res'][run_id]['best_logl'].mean(), 
                                yerr= 2*all_baselines['validation_res'][run_id]['best_logl'].std()/np.sqrt(baseline_config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                label=baseline_name[0], 
                                color=baseline_cols[running_id], linestyle='--') #'''

                        # balanced acc
                        if dataset_name == 'mnist':
                            axs[0,1].hlines(all_baselines['validation_res'][run_id]['best_balanced_acc'],1,config['n_global_updates'], color=baseline_cols[running_id], linestyle='--', label=baseline_name[0])
                            '''
                            axs[0,1].errorbar(x, np.zeros(config['n_global_updates'])+all_baselines['validation_res'][run_id]['best_balanced_acc'].mean(), 
                                    yerr= 2*all_baselines['validation_res'][run_id]['best_balanced_acc'].std()/np.sqrt(baseline_config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                    label=baseline_name[0], 
                                    color=baseline_cols[running_id], linestyle='--') #'''
                        # avg ROC curve at best logl global update
                        else:
                            axs[0,1].plot( 1-all_baselines['validation_res'][run_id]['TNR'].mean(0), all_baselines['validation_res'][run_id]['TPR'].mean(0),linestyle='--', color=baseline_cols[running_id], label=baseline_name[0]  )
                            #plt.plot(1-all_res['validation_res'][str(i_run)]['TNR'][i_seed,:], all_res['validation_res'][str(i_run)]['TPR'][i_seed,:] )

                        # avg prec score
                        axs[1,1].hlines(all_baselines['validation_res'][run_id]['best_avg_prec_score'],1,config['n_global_updates'], color=baseline_cols[running_id], linestyle='--', label=baseline_name[0])
                        '''
                        axs[1,1].errorbar(x, np.zeros(config['n_global_updates'])+all_baselines['validation_res'][run_id]['best_avg_prec_score'].mean(), 
                                yerr= 2*all_baselines['validation_res'][run_id]['best_avg_prec_score'].std()/np.sqrt(baseline_config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                label=baseline_name[0], 
                                color=baseline_cols[running_id], linestyle='--') #'''

                    running_id += 1


                    """

                        #posneg_measures = ['avg_prec_score','balanced_acc','f1_score']
                        # balanced acc
                        #'''
                        axs[0,1].errorbar(x, all_res['validation_res'][str(i_run)]['balanced_acc'].mean(-1), 
                                yerr= 2*all_res['validation_res'][str(i_run)]['balanced_acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                label=cur_label,
                                color=colors[i_line%len(colors)]
                                )
                        #'''
                        # avg ROC curve at best logl global update
                        '''
                        axs[0,1].plot( 1-all_res['validation_res'][str(i_run)]['TNR'].mean(0), all_res['validation_res'][str(i_run)]['TPR'].mean(0), 
                                #yerr= 2*all_res['validation_res'][str(i_run)]['balanced_acc'].std(0)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                label=cur_label,
                                color=colors[i_line%len(colors)]
                                )
                        #plt.plot(1-all_res['validation_res'][str(i_run)]['TNR'][i_seed,:], all_res['validation_res'][str(i_run)]['TPR'][i_seed,:] )
                        #'''
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
                        # balanced acc
                        #'''
                        axs[0,1].errorbar(x, all_res['train_res'][str(i_run)]['balanced_acc'].mean(-1), 
                                yerr= 2*all_res['train_res'][str(i_run)]['balanced_acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                label=cur_label,
                                color=colors[i_line%len(colors)]
                                )
                        #'''
                        axs[1,1].errorbar(x, all_res['train_res'][str(i_run)]['avg_prec_score'].mean(-1), 
                                yerr= 2*all_res['train_res'][str(i_run)]['avg_prec_score'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                label=cur_label,
                                color=colors[i_line%len(colors)]
                    )"""


    axs[0,0].set_ylabel('Acc')
    axs[0,0].set_xlabel('Global communications')
    if plot_legends:
        axs[0,0].legend(loc='lower right')
        axs[1,0].legend(loc='lower right')
    if ylims[0] is not None:
        axs[0,0].set_ylim(ylims[0])
    if ylims[1] is not None:
        axs[1,0].set_ylim(ylims[1])
    if ylims[2] is not None:
        axs[0,1].set_ylim(ylims[2])
    if ylims[3] is not None:
        axs[1,1].set_ylim(ylims[3])

    axs[1,0].set_ylabel('Logl')
    axs[1,0].set_xlabel('Global communications')
    #axs[0,1].set_ylabel('Balanced acc')
    #axs[0,1].set_xlabel('Global communications')
    axs[0,1].set_ylabel('ROC curve at best logl global update')
    axs[0,1].set_xlabel('Threshold')
    axs[1,1].set_ylabel('Avg precision score')
    axs[1,1].set_xlabel('Global communications')
    for i in range(2):
        for ii in range(2):
            axs[i,ii].grid(b=True, which='major', axis='both')

    '''
    if plot_test_error:
        plt.suptitle(f"{dataset_name} dataset mean with 2*SEM over {config['n_rng_seeds']} runs" + cur_title)
    else:
        plt.suptitle(f"{dataset_name} dataset training err mean with 2*SEM over {config['n_rng_seeds']} runs" + cur_title)
    '''

print(f'\nBest acc: {best_acc:.5f} found with config:\n{best_confs[0]}')
if best_confs[1] == best_confs[0]:
    best_confs[1] = None
print(f'\nBest logl: {best_logl:.5f} found with config:\n{best_confs[1]}')

if to_disk:
    plt.savefig(fig_folder+fig_name)
    fig_folder

else:
    plt.show()


