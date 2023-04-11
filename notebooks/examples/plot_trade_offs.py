
from collections import OrderedDict as OD
import logging
from pathlib import Path
import sys

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from matplotlib import pyplot as plt
import numpy as np


### TRADE OFF RUNS
## ADULT
## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_1seeds_trade_off_runs/'
#runs_to_plot = np.linspace(1,185,185,dtype=int) # 1) & 2) eps comparison runs with varying q, all Cs
#runs_to_plot = np.append(np.linspace(186,465,280,dtype=int),np.linspace(36,42,7,dtype=int)) # 3) pre clip noise tests
#runs_to_plot = np.append(np.linspace(466,535,70,dtype=int),np.linspace(36,42,7,dtype=int)) # 4) pre clip noise tests again

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_1seeds_trade_off_runs/'
##runs_to_plot = np.linspace(1,42,42,dtype=int)#[1] # 1) eps comparison runs with varying q, C=.5
##runs_to_plot = np.append(np.linspace(43,182,140,dtype=int),np.linspace(36,42,7,dtype=int)) # 2) eps comparison runs with varying q, more Cs
##runs_to_plot = np.linspace(1,182,182,dtype=int) # 2) eps comparison runs with varying q, more Cs
##runs_to_plot = np.append(np.linspace(183,462,280,dtype=int), np.array([7,8,9,10,11,28,29],dtype=int) ) # 3) pre clip noise tests
#runs_to_plot = np.linspace(1,182,182,dtype=int)#[1] # 1) eps comparison runs with varying q and C, seq server
#runs_to_plot = np.append(np.linspace(183,357,175,dtype=int),np.linspace(176,182,7,dtype=int)) #[1] # 2) eps comparison runs with varying q and C, seq server
#runs_to_plot = np.append(np.linspace(183,357,175,dtype=int),np.linspace(358,364,7,dtype=int)) #[1] # 2) eps comparison runs with varying q and C, synch server
#runs_to_plot = np.append(np.linspace(358,364,7,dtype=int),np.linspace(176,182,7,dtype=int)) # nondp seq vs synch server
#runs_to_plot = np.append(np.linspace(358,574,217,dtype=int),np.linspace(176,182,7,dtype=int)) # 3) add pseudo-client sampling frac
#runs_to_plot = np.append(np.linspace(365,574,210,dtype=int),np.linspace(575,593,19,dtype=int)) # 3) add pseudo-client sampling frac, also to nondp

################ 1BNN trade offs ################
### ADULT 1BNN
## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_1seeds_trade_off_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) nondp runs, compare q with fixed C

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_1seeds_trade_off_runs/'
#runs_to_plot = np.concatenate([np.linspace(1,72,72,dtype=int),np.linspace(265,328,64,dtype=int)]) # 1)&3) nondp runs, compare q with fixed C
#runs_to_plot = np.linspace(73,264,192,dtype=int) # 2) dp runs, compare q with fixed C

### BALANCED MIMIC3 1BNN
## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_1bnn_lfa_10clients_1seeds_trade_off_runs/'
#runs_to_plot = np.linspace(1,74,74,dtype=int) # 1) nondp runs, compare q with fixed C

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_1bnn_local_pvi_10clients_1seeds_trade_off_runs/'
#runs_to_plot = np.linspace(1,80,80,dtype=int) # 1) nondp runs, compare q with fixed C


### REBUTTAL TRADE OFF PLOTTING
main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_trade_off_plotting_10clients_5seeds_runs/'
runs_to_plot = np.linspace(1,15,15,dtype=int) # 1) nondp runs, compare q with fixed C


### TRADE OFF 2 RUNS
## NONDP BATCHES
# note: plotting this requires dedicated code, since not using q but numebr of clients
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_nondp_batches_1seeds_trade_off2_runs/'
#runs_to_plot = np.linspace(1,120,120,dtype=int) # 1)

## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_1seeds_trade_off2_runs/'
#runs_to_plot = np.linspace(1,126,126,dtype=int) # 1)

#print(runs_to_plot)
#print(len(runs_to_plot))
#sys.exit()


# where to save all plots
fig_folder = 'res_plots/'

# set plotting type: 
# 'eps_trade_off': plot acc,logl etc vs list of eps values, C values as separate lines
# 'q_trade_off': plot acc,logl etc vs list of q values, 

#plot_type = 'eps_trade_off'
#plot_type = 'q_trade_off'
#plot_type = 'q_trade_off_with_C'
plot_type = 'q_trade_off_rebuttal' # plot best performance over global vs q; separate line for eps, C levels
#plot_type = 'q_trade_off_rebuttal2' # plot perf vs global step; separate line for eps, C for each q






#all_eps_sigma = np.asarray([(np.inf,np.inf,10,5,1,.2),(0., 0., 3.16, 5.64, 23.61, 103.58) ])
all_eps_sigma = np.asarray([(np.inf,np.inf,4.),(0., 0.,34.1849) ])
#all_q = np.asarray([.5,.1,5e-2,1e-2,5e-3,1e-3,5e-4])
#all_q = np.asarray([.1,5e-2,1e-2,5e-3,1e-3,5e-4])
all_q = np.asarray([1e-2,5e-2,.1,.5,1.])
all_steps = np.asarray([50])
#all_steps = np.asarray([10,50,100]) # for trade off2 plotting
#all_C = np.asarray([5.,10.,20.])
all_C = np.asarray([1.,1000.])

nondp_C = 100. # C at least this big with dp_sigma=0 considered to be nonDP

restrictions = OD()
restrictions['dp_sigma'] = None#[7.98]
restrictions['dp_C'] = None#[1000.]
restrictions['n_global_updates'] = [5]
restrictions['n_steps'] = None#[100]
restrictions['batch_size'] = None#[5]
restrictions['sampling_frac_q'] = None#[5e-3]
restrictions['pseudo_client_q'] = None#[.1]
restrictions['learning_rate'] = None#[5e-3]
restrictions['damping_factor'] = None#[.4]
restrictions['init_var'] = None#[1e-3]
restrictions['dp_mode'] = None#['nondp_batches']
restrictions['pre_clip_sigma'] = None#[50.]

# possible balance settings: (0,0), (.7,-3), (.75,.95)
restrictions['data_bal_rho'] = [.0]
restrictions['data_bal_kappa'] = [.0]


# save to disk (or just show)
to_disk = 0


dataset_name = 'mimic3_bal'
#dataset_name = 'mnist'
#dataset_name = 'mimic3'
#dataset_name = 'adult'
#dataset_name = 'mushroom'
#dataset_name = 'credit'
#dataset_name = 'bank'


# name for the current plot
plot_filename = f"tradeoff_lfa_{dataset_name}_{restrictions['dp_C']}_nondp_log.pdf"
#plot_filename = f"tradeoff_local_pvi_{dataset_name}_{restrictions['dp_C']}_nondp_log.pdf"


# set to None to skip:
if dataset_name == 'mimic3_bal':
    ylims= ((.5,.8),(-.85,-.5), None, None)
    baseline_acc = 0.761
    baseline_logl = -.517
if dataset_name == 'mimic3':
    ylims= ((.87,.9),(-.55,-.26), None, None)
    baseline_acc = 0.8844
    baseline_logl = None

elif dataset_name == 'adult':
    #ylims= ((.7,.9),(-.6,-.26),(.5,.8),(.5,.8)) # acc, logl, bal acc, avg prec score
    ylims= ((.7,.9),(-.6,-.26),(-.05,1.05),(.5,.8)) # acc, logl, ROC, avg prec score
    baseline_acc = 0.761
    baseline_logl = None
elif dataset_name == 'mnist':
    ylims= ((.94,1.),(-.2,-.0), None, None)
    #ylims= ((.7,.95),(-.8,-.3), None, None)
    #ylims = (None,None,None,None)
    baseline_acc = None#0.8844
    baseline_logl = None


#####################
# set baselines

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

to_plot = OD()


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
        try:
            all_res['validation_res'][run_id]['TPR'] = np.zeros((
                all_res['config'][run_id]['n_rng_seeds'],
                apu['validation_res_seed0']['posneg'][0]['n_points']
                ))
            all_res['validation_res'][run_id]['TNR'] = np.zeros((
                all_res['config'][run_id]['n_rng_seeds'],
                apu['validation_res_seed0']['posneg'][0]['n_points']
                ))
            all_res['validation_res'][run_id]['ROC_thresholds'] = np.linspace(0,1,apu['validation_res_seed0']['posneg'][0]['n_points'])
        except:
            print('error in posneg results')

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
            try:
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

            except:
                print('error in AUCROC')




for i_run in runs_to_plot:

    run_id = str(i_run)
    print(f'run {run_id}')
    filename = main_folder + run_id + '/config.json'
    #print(f'trying {filename}')
    tmp = read_config(filename, failed_runs)
    if i_run in failed_runs:
        continue

    all_res['config'][run_id] = tmp[0]

    # try opening sacred records, if missing open manual bck instead
    filename = main_folder + run_id + '/info.json'
    filename_bck = main_folder + run_id + '/info_bck.json'
    apu = read_results(filename, filename_bck)

    #print(apu['validation_res_seed0']['logl'].shape)

    #for k in apu:
    #    print(k)
    #sys.exit()
    
    # format results for plotting
    client_measures = ['elbo','kl','logl']
    format_results(apu, run_id, client_measures, all_res)




if len(failed_runs) > 0:
    print(f'failed runs:\n{failed_runs}')
    runs_to_plot = list(runs_to_plot)
    for i_run in failed_runs:
        runs_to_plot.remove(i_run)
    runs_to_plot = np.array(runs_to_plot)


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



# check restrictions
list_to_print = []
for i_run in runs_to_plot:
    print_this = True
    for k in restrictions:
        try:
            if restrictions[k] is not None and all_res['config'][str(i_run)][k] not in restrictions[k]:
                print_this = False
        except:
            continue
    # check baselines
    '''
    if include_baselines and not print_this:
        for tmp in baselines:
            print_this = True
            for k in tmp:
                if tmp[k] is not None and all_res['config'][str(i_run)][k] != tmp[k]:
                    print_this = False
                    break
            if print_this:
                break
    '''
    if print_this:
        list_to_print.append(i_run)
if len(list_to_print) == 0:
    sys.exit('No runs satisfying restrictions found!')
else:
    print(f'Found {len(list_to_print)} runs to plot')

if plot_type == 'q_trade_off':
    tmp = []
    for n in all_steps:
        if restrictions['n_steps'] is None or n in restrictions['n_steps']:
            tmp.append(n)
    all_steps = np.asarray(tmp)
    #print(f'set all_steps to {all_steps}')
elif plot_type == 'q_trade_off_with_C':
    tmp = []
    for c in all_C:
        if restrictions['dp_C'] is None or c in restrictions['dp_C']:
            tmp.append(c)
    all_C = np.asarray(tmp)

# want mean, logl, avg.prec. + mean ROC at best logl point separately
# will be one value for each run on list
#all_eps_sigma
#all_q
if plot_type == 'eps_trade_off':
    to_plot['best_mean_acc'] = np.zeros((2,len(all_eps_sigma[0]), len(all_q)))
    to_plot['best_mean_logl'] = np.zeros((2,len(all_eps_sigma[0]), len(all_q)))
    to_plot['best_mean_avg_prec_score'] = np.zeros((2,len(all_eps_sigma[0]), len(all_q)))
    to_plot['mean_ROC_at_best_mean_logl'] = OD()
    #np.zeros(( len(all_eps_sigma[0]), len(all_q), n_points, n_points))
elif plot_type == 'q_trade_off':
    to_plot['best_mean_acc'] = np.zeros((2,len(all_steps),len(all_q)))
    to_plot['best_mean_logl'] = np.zeros((2,len(all_steps), len(all_q)))
    to_plot['best_mean_avg_prec_score'] = np.zeros((2,len(all_steps), len(all_q)))
    to_plot['mean_ROC_at_best_mean_logl'] = OD()
elif plot_type == 'q_trade_off_with_C':
    to_plot['best_mean_acc'] = np.zeros((2,len(all_C),len(all_q)))
    to_plot['best_mean_logl'] = np.zeros((2,len(all_C), len(all_q)))
    to_plot['best_mean_avg_prec_score'] = np.zeros((2,len(all_C), len(all_q)))
    to_plot['mean_ROC_at_best_mean_logl'] = OD()

elif plot_type == 'q_trade_off_rebuttal':
    # plot mean (over seeds) acc/logl against q values, with fixed eps, C; use max performance on any global
    to_plot['best_mean_acc'] = np.zeros((2,len(all_eps_sigma[0]), len(all_q)))
    to_plot['best_mean_logl'] = np.zeros((2,len(all_eps_sigma[0]), len(all_q)))
    to_plot['best_mean_avg_prec_score'] = np.zeros((2,len(all_eps_sigma[0]), len(all_q)))
    to_plot['mean_ROC_at_best_mean_logl'] = OD()
elif plot_type == 'q_trade_off_rebuttal2':
    # plot q as different lines; acc/logl against global update to show convergence speed
    # these are now not best means, but just means over seeds
    # take number of globals from any run; should be same for all to make any sense
    to_plot['mean_acc'] = np.zeros((2,len(all_eps_sigma[0]),len(all_q), all_res['config']['1']['n_global_updates']))
    to_plot['mean_logl'] = np.zeros((2,len(all_eps_sigma[0]),len(all_q), all_res['config']['1']['n_global_updates']))
    to_plot['mean_avg_prec_score'] = np.zeros((2,len(all_eps_sigma[0]), len(all_q),all_res['config']['1']['n_global_updates']))
    to_plot['mean_ROC_at_best_mean_logl'] = OD()

    # JATKA: kts miten fiksata ja tuleeko mitään okta
    #to_plot['dp_C'] = np.zeros()

else:
    sys.exit(f'Unknown plot type: {plot_type}')



#print( all_res['validation_res']['1']['TPR'].shape ) # TPR, TNR, ROC_thresholds
#sys.exit()

for i_line,i_run in enumerate(list_to_print):

    config = all_res['config'][str(i_run)]
    res = all_res['validation_res'][str(i_run)]

    # best mean over all global updates
    tmp = ['acc', 'logl', 'avg_prec_score']
    for i_tmp,tmp_name in enumerate(tmp):
        # take argmax logl as the best model
        #print(all_res['validation_res'][str(i_run)]['logl'].shape)
        i_max = np.argmax(all_res['validation_res'][str(i_run)]['logl'],0)
        #print(i_max)
        #sys.exit()
        
        if plot_type in ['eps_trade_off']:
            raise NotImplementedError('fix argmax')
            #logging.warning('check argmax here!')
            if config['dp_sigma'] != 0 and config['dp_sigma'] is not None:
                i_max = np.argmax(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
                to_plot[f'best_mean_{tmp_name}'][0,all_eps_sigma[1] == config['dp_sigma'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].mean(-1)[i_max]
                to_plot[f'best_mean_{tmp_name}'][1,all_eps_sigma[1] == config['dp_sigma'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].std(-1)[i_max]

            else:
                if config['dp_C'] < nondp_C:
                    # only clipping
                    i_eps = 1
                else:
                    # nonDP
                    if i_tmp == 0:
                        try:
                            print(f"nondp run: {i_line}: dp_C={config['dp_C']}, dp_sigma={config['dp_sigma']}, sampling q={config['sampling_frac_q']}, pseudo q={config['pseudo_client_q']}")
                        except:
                            print("nondp doesn't have pseudo client conf?")
                    i_eps = 0

                i_max = np.argmax(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
                to_plot[f'best_mean_{tmp_name}'][0,i_eps, all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].mean(-1)[i_max]
                to_plot[f'best_mean_{tmp_name}'][1,i_eps, all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].std(-1)[i_max]

        elif plot_type == 'q_trade_off':
            raise NotImplementedError('fix argmax')
            #print(config['n_steps'], all_steps, all_steps == config['n_steps'])
            i_max = np.argmax(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
            to_plot[f'best_mean_{tmp_name}'][0, all_steps == config['n_steps'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].mean(-1)[i_max]
            to_plot[f'best_mean_{tmp_name}'][1,all_steps == config['n_steps'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].std(-1)[i_max]

        elif plot_type == 'q_trade_off_with_C':
            # NOTE: check that works with several seeds if used
            #print(all_res['validation_res'][str(i_run)][tmp_name][i_max].shape)
            #print(all_res['validation_res'][str(i_run)][tmp_name][i_max].mean(-1))

            #print(config['n_steps'], all_steps, all_steps == config['n_steps'])
            #i_max = np.argmax(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
            to_plot[f'best_mean_{tmp_name}'][0, all_C == config['dp_C'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name][i_max].mean(-1)
            to_plot[f'best_mean_{tmp_name}'][1,all_C == config['dp_C'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name][i_max].std(-1)
            #i_max = np.argmax(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
            #to_plot[f'best_mean_{tmp_name}'][0, all_C == config['dp_C'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].mean(-1)[i_max]
            #to_plot[f'best_mean_{tmp_name}'][1,all_C == config['dp_C'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].std(-1)[i_max]


        elif plot_type in ['q_trade_off_rebuttal']:
            #print(i_run,tmp_name)
            #print(all_res['validation_res'][str(i_run)][tmp_name].shape)
            #print(all_res['validation_res'][str(i_run)][tmp_name])
            #print(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
            #print(all_res['validation_res'][str(i_run)][tmp_name][i_max].mean(-1))

            if config['dp_sigma'] != 0 and config['dp_sigma'] is not None:
                i_max = np.argmax(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
                to_plot[f'best_mean_{tmp_name}'][0,all_eps_sigma[1] == config['dp_sigma'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].mean(-1)[i_max]
                to_plot[f'best_mean_{tmp_name}'][1,all_eps_sigma[1] == config['dp_sigma'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].std(-1)[i_max]
            else:
                if config['dp_C'] < nondp_C:
                    # only clipping
                    i_eps = 1
                else:
                    # nonDP
                    if i_tmp == 0:
                        try:
                            print(f"nondp run: {i_line}: dp_C={config['dp_C']}, dp_sigma={config['dp_sigma']}, sampling q={config['sampling_frac_q']}, pseudo q={config['pseudo_client_q']}")
                        except:
                            print("nondp doesn't have pseudo client conf?")
                    i_eps = 0
                i_max = np.argmax(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
                to_plot[f'best_mean_{tmp_name}'][0,i_eps, all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].mean(-1)[i_max]
                to_plot[f'best_mean_{tmp_name}'][1,i_eps, all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].std(-1)[i_max]


        elif plot_type in ['q_trade_off_rebuttal2']:
            #print(i_run,tmp_name)
            #print(all_res['validation_res'][str(i_run)][tmp_name].shape)
            #print(all_res['validation_res'][str(i_run)][tmp_name])
            #print(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
            #print(all_res['validation_res'][str(i_run)][tmp_name][i_max].mean(-1))

            # this needs array as well!
            logging.warning('Need array here!')
            to_plot['dp_C'] = config['dp_C']

            if config['dp_sigma'] != 0 and config['dp_sigma'] is not None:
                #i_max = np.argmax(all_res['validation_res'][str(i_run)][tmp_name].mean(-1))
                to_plot[f'mean_{tmp_name}'][0, all_eps_sigma[1] == config['dp_sigma'], all_q == config['sampling_frac_q'],:]  = all_res['validation_res'][str(i_run)][tmp_name].mean(-1)
                to_plot[f'mean_{tmp_name}'][1, all_eps_sigma[1] == config['dp_sigma'], all_q == config['sampling_frac_q'] ]  = all_res['validation_res'][str(i_run)][tmp_name].std(-1)
            else:
                if config['dp_C'] < nondp_C:
                    # only clipping
                    i_eps = 1
                else:
                    # nonDP
                    if i_tmp == 0:
                        try:
                            print(f"nondp run: {i_line}: dp_C={config['dp_C']}, dp_sigma={config['dp_sigma']}, sampling q={config['sampling_frac_q']}, pseudo q={config['pseudo_client_q']}")
                        except:
                            print("nondp doesn't have pseudo client conf?")
                    i_eps = 0
                to_plot[f'mean_{tmp_name}'][0,i_eps, all_q == config['sampling_frac_q'],:]  = all_res['validation_res'][str(i_run)][tmp_name].mean(-1)
                to_plot[f'mean_{tmp_name}'][1,i_eps, all_q == config['sampling_frac_q'],:]  = all_res['validation_res'][str(i_run)][tmp_name].std(-1)

#to_plot['mean_acc'] = np.zeros((2,len(all_eps_sigma[0]),len(all_q), all_res['config']['1']['n_global_updates']))


#########

if plot_type in ['eps_trade_off','q_trade_off_rebuttal']:
    fig, axs = plt.subplots(2,2)
    plt.suptitle(f"Included clipping C: {restrictions['dp_C']}")
    for i_line, eps in enumerate(all_eps_sigma[0]):
        # NOTE: FIX THIS
        if i_line < 2:
            C = all_C[i_line]
        else:
            C = 1
        #axs[0,0].plot(all_q, to_plot['best_mean_acc'][0,i_line,:], label=f'eps={eps}' )
        axs[0,0].errorbar(np.log10(all_q), to_plot['best_mean_acc'][0,i_line,:], 
                    yerr= 2*to_plot['best_mean_acc'][1,i_line,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=f'eps={eps}, C={C}', 
                    color=colors[i_line%len(colors)]
                    )
        #axs[1,0].plot(all_q, to_plot['best_mean_logl'][0,i_line,:], label=f'eps={eps}' )
        axs[1,0].errorbar(np.log10(all_q), to_plot['best_mean_logl'][0,i_line,:], 
                    yerr= 2*to_plot['best_mean_logl'][1,i_line,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=f'eps={eps}, C={C}', 
                    color=colors[i_line%len(colors)]
                    )
        #axs[0,1].plot(all_q, to_plot['best_mean_avg_prec_score'][0,i_line,:], label=f'eps={eps}' )
        axs[0,1].errorbar(np.log10(all_q), to_plot['best_mean_avg_prec_score'][0,i_line,:], 
                    yerr= 2*to_plot['best_mean_avg_prec_score'][1,i_line,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=f'eps={eps}, C={C}', 
                    color=colors[i_line%len(colors)]
                    )
        axs[1,1].plot(0,0, label=f"eps={eps}, C={C}") # this is currently just used for labels
    axs[1,1].tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False, labelleft=False)

elif plot_type == 'q_trade_off':
    fig, axs = plt.subplots(2,2)
    #plt.suptitle(f"Included clipping C: {restrictions['dp_C']}")
    for i_line, n in enumerate(all_steps):
        axs[0,0].errorbar(np.log(all_q), to_plot['best_mean_acc'][0,i_line,:], 
                yerr= 2*to_plot['best_mean_acc'][1,i_line,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=f"{config['dp_mode']}: steps={n}", 
                        #color=colors[i_line%len(colors)]
                        )
        #axs[1,0].plot(all_q, to_plot['best_mean_logl'][0,i_line,:], label=f'eps={eps}' )
        axs[1,0].errorbar(np.log(all_q), to_plot['best_mean_logl'][0,i_line,:], 
                        yerr= 2*to_plot['best_mean_logl'][1,i_line,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                        label=f"{config['dp_mode']}: steps={n}",
                        #color=colors[i_line%len(colors)]
                        )
        #axs[0,1].plot(all_q, to_plot['best_mean_avg_prec_score'][0,i_line,:], label=f'eps={eps}' )
        axs[0,1].errorbar(np.log(all_q), to_plot['best_mean_avg_prec_score'][0,i_line,:], 
                        yerr= 2*to_plot['best_mean_avg_prec_score'][1,i_line,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                        label=f"{config['dp_mode']}: steps={n}",
                        #color=colors[i_line%len(colors)]
                        )
        axs[1,1].plot(0,0, label=f"{config['dp_mode']}: steps={n}") # this is currently just used for labels
        axs[1,1].tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False, labelleft=False)
        #'''

elif plot_type == 'q_trade_off_with_C':
    fig, axs = plt.subplots(2,2)
    #plt.suptitle(f"Included clipping C: {restrictions['dp_C']}")
    for i_line, c in enumerate(all_C):
        axs[0,0].errorbar(np.log(all_q), to_plot['best_mean_acc'][0,i_line,:], 
                yerr= 2*to_plot['best_mean_acc'][1,i_line,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=f"{config['dp_mode']}: C={c}", 
                        #color=colors[i_line%len(colors)]
                        )
        #axs[1,0].plot(all_q, to_plot['best_mean_logl'][0,i_line,:], label=f'eps={eps}' )
        axs[1,0].errorbar(np.log(all_q), to_plot['best_mean_logl'][0,i_line,:], 
                        yerr= 2*to_plot['best_mean_logl'][1,i_line,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                        label=f"{config['dp_mode']}: C={c}",
                        #color=colors[i_line%len(colors)]
                        )
        #axs[0,1].plot(all_q, to_plot['best_mean_avg_prec_score'][0,i_line,:], label=f'eps={eps}' )
        axs[0,1].errorbar(np.log(all_q), to_plot['best_mean_avg_prec_score'][0,i_line,:], 
                        yerr= 2*to_plot['best_mean_avg_prec_score'][1,i_line,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                        label=f"{config['dp_mode']}: C={c}",
                        #color=colors[i_line%len(colors)]
                        )
        axs[1,1].plot(0,0, label=f"{config['dp_mode']}: dp_C={c}") # this is currently just used for labels
        axs[1,1].tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False, labelleft=False)
        #'''


elif plot_type in ['q_trade_off_rebuttal2']:
    fig, axs = plt.subplots(2,2)
    plt.suptitle(f"Included clipping C: {restrictions['dp_C']}")
    for i_line, eps in enumerate(all_eps_sigma[0]):
        C = to_plot['dp_C']
        if restrictions['dp_C'] is not None and C not in restrictions['dp_C']: continue
        n_globals= restrictions['n_global_updates'][0]
        #axs[0,0].plot(all_q, to_plot['best_mean_acc'][0,i_line,:], label=f'eps={eps}' )
        for i_q, q in enumerate(all_q):
            axs[0,0].errorbar(np.linspace(1,n_globals,n_globals), to_plot['mean_acc'][0,i_line,i_q,:], 
                        yerr= 2*to_plot['mean_acc'][1,i_line,i_q,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                        label=f'eps={eps}, C={C}', 
                        color=colors[i_q%len(colors)]
                        )
        #axs[1,0].plot(all_q, to_plot['best_mean_logl'][0,i_line,:], label=f'eps={eps}' )
        #"""
        axs[1,0].errorbar(np.linspace(1,n_globals,n_globals), to_plot['mean_logl'][0,i_line,i_q,:], 
                    yerr= 2*to_plot['mean_logl'][1,i_line,i_q,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=f'eps={eps}, C={C}', 
                    color=colors[i_line%len(colors)]
                    )
        #"""
        #axs[0,1].plot(all_q, to_plot['best_mean_avg_prec_score'][0,i_line,:], label=f'eps={eps}' )
        #"""
        axs[0,1].errorbar(np.linspace(1,n_globals,n_globals), to_plot['mean_avg_prec_score'][0,i_line,i_q,:], 
                    yerr= 2*to_plot['mean_avg_prec_score'][1,i_line,i_q,:]/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=f'eps={eps}, C={C}', 
                    color=colors[i_line%len(colors)]
                    )
        #"""
        axs[1,1].plot(0,0, label=f"eps={eps}, C={C}") # this is currently just used for labels
    axs[1,1].tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False, labelleft=False)


#to_plot['mean_acc'] = np.zeros((2,len(all_eps_sigma[0]),len(all_q), all_res['config']['1']['n_global_updates']))

axs[1,1].legend()
axs[0,0].grid()
axs[1,0].grid()
axs[0,1].grid()
axs[0,0].set_ylabel('Acc')
axs[1,0].set_ylabel('Logl')
axs[0,1].set_ylabel('Avg prec score')
axs[0,0].set_xlabel('Sampling frac q')
axs[1,0].set_xlabel('Sampling frac q')
axs[0,1].set_xlabel('Sampling frac q')
plt.tight_layout()
if to_disk:
    plt.savefig(fig_folder + plot_filename)
else:
    plt.show()

'''
axs[1].errorbar(all_q, np.zeros(config['n_global_updates'])+all_baselines['validation_res'][run_id]['best_logl'].mean(), 
                                yerr= 2*all_baselines['validation_res'][run_id]['best_logl'].std()/np.sqrt(baseline_config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                label=baseline_name[0], 
                                color=baseline_cols[running_id], linestyle='--') #'''

def pyh():
    # ja vielä infty erikseen, koska pitää kts onko klipattu vai ei

    #x = config['sampling_frac_q']
    #if do_trade_off_runs:
    #    to_plot[str(i_run)]['best_mean_acc'] = np.zeros((config['dp_sigma'],config['sampling_frac_q']))

    # eli x-akseli=q, halutaan näyttää miten eri q vaikuttaa; eri viivat=eri eps

    #x = np.linspace(1,config['n_global_updates'],config['n_global_updates'])


    #continue
    #sys.exit('ok')


    # testing error plot
    '''
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
    '''
    #posneg_measures = ['avg_prec_score','balanced_acc','f1_score']
    # balanced acc
    '''
        axs[0,1].plot( 1-all_res['validation_res'][str(i_run)]['TNR'].mean(0), all_res['validation_res'][str(i_run)]['TPR'].mean(0), 
                #yerr= 2*all_res['validation_res'][str(i_run)]['balanced_acc'].std(0)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                label=cur_label,
                color=colors[i_line%len(colors)]
                )
    #plt.plot(1-all_res['validation_res'][str(i_run)]['TNR'][i_seed,:], all_res['validation_res'][str(i_run)]['TPR'][i_seed,:] )

    axs[1,1].errorbar(x, all_res['validation_res'][str(i_run)]['avg_prec_score'].mean(-1), 
            yerr= 2*all_res['validation_res'][str(i_run)]['avg_prec_score'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
            label=cur_label,
            color=colors[i_line%len(colors)]
            )
    '''

    """
    # add baselines if available
    if i_line == 0:
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



                        #posneg_measures = ['avg_prec_score','balanced_acc','f1_score']
                        # avg ROC curve at best logl global update
                        '''
                        axs[0,1].plot( 1-all_res['validation_res'][str(i_run)]['TNR'].mean(0), all_res['validation_res'][str(i_run)]['TPR'].mean(0), 
                                #yerr= 2*all_res['validation_res'][str(i_run)]['balanced_acc'].std(0)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                                label=cur_label,
                                color=colors[i_line%len(colors)]
                                )
                        #plt.plot(1-all_res['validation_res'][str(i_run)]['TNR'][i_seed,:], all_res['validation_res'][str(i_run)]['TPR'][i_seed,:] )
    """



#print(to_plot)
