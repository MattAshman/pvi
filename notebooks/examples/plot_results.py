
from collections import OrderedDict as OD
from pathlib import Path
import sys

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from matplotlib import pyplot as plt
import numpy as np


############ trade off run checks
### BALANCED MIMIC3 1BNN
## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_1bnn_lfa_10clients_1seeds_trade_off_runs/'
#runs_to_plot = np.linspace(1,74,74,dtype=int) # 1) nondp runs, compare q with fixed C

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_1bnn_local_pvi_10clients_1seeds_trade_off_runs/'
#runs_to_plot = np.linspace(1,80,80,dtype=int) # 1) nondp runs, compare q with fixed C

### ADULT 1BNN
## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_1seeds_trade_off_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) nondp runs, compare q with fixed C

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_1seeds_trade_off_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) nondp runs, compare q with fixed C
#runs_to_plot = np.linspace(73,264,192,dtype=int) # 1) dp runs, compare q with fixed C


##########################################################
### 200 clients adult 1seed runs
## eps=.2
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
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_200clients_1seeds_eps02_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int)#[1] # 1) 10 global, bal data, fiksattu bugi 8.3.
#runs_to_plot = np.linspace(109,162,54,dtype=int)#[1] # 2) 15 global, bal data, fiksattu bugi 8.3.
#runs_to_plot = np.linspace(163,270,108,dtype=int)#[1] # 2) 15 global, unbal data, fiksattu bugi 8.3.


## eps=.3
# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_200clients_1seeds_eps03_runs/'
#runs_to_plot = np.linspace(1,32,32,dtype=int) # 1) bal
#runs_to_plot = np.linspace(33,80,48,dtype=int) # 2) unbal

# LFA sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_200clients_1seeds_eps03_sec_runs/'
#runs_to_plot = np.linspace(1,64,64,dtype=int) # 1) bal
#runs_to_plot = np.linspace(65,192,128,dtype=int) # 2) unbal

# LOCAL PVI sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_200clients_1seeds_eps03_sec_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) bal
#runs_to_plot = np.linspace(73,216,144,dtype=int) # 1) unbal

## eps=.5

# BCM SAME
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_bcm_same_200clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,432,432,dtype=int) # 1)-3 bal & unbal

# BCM SPLIT
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_bcm_split_200clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,436,436,dtype=int) # 1)-3 bal & unbal


# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_200clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,18,18,dtype=int) # 1) bal
#runs_to_plot = np.linspace(19,30,12,dtype=int) # 2) bal eri q
#runs_to_plot = np.linspace(31,54,24,dtype=int) # 3) unbal

# LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_200clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,141,141,dtype=int)#[1] # 1) bal

# LFA sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_200clients_1seeds_eps05_sec_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int)#[1] # 1) bal
#runs_to_plot = np.linspace(109,324,216,dtype=int)#[1] # 2) unbal


# LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_200clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,115,115,dtype=int)#[1] # 1) bal

# LOCAL PVI sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_200clients_1seeds_eps05_sec_runs/'
#runs_to_plot = np.linspace(1,121,121,dtype=int)#[1] # 1) bal
#runs_to_plot = np.linspace(122,283,162,dtype=int)#[1] # 2) unbal



## eps=1
# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_200clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,36,36,dtype=int) # 1)


##########################################################
### 200 clients ADULT 5seed runs

## eps=.3
# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_200clients_5seeds_eps03_runs/'
#runs_to_plot = np.linspace(1,8,8,dtype=int) # 1) unbal
#runs_to_plot = [9] # 2) bal

# LFA sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_200clients_5seeds_eps03_sec_runs/'
#runs_to_plot = [1,2] # 1) unbal
#runs_to_plot = np.linspace(3,6,4,dtype=int) # 2) bal
#runs_to_plot = np.linspace(7,10,4,dtype=int) # 3) unbal
#runs_to_plot = np.linspace(11,14,4,dtype=int) # 3) unbal

# LOCAL PVI sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_200clients_5seeds_eps03_sec_runs/'
#runs_to_plot = np.linspace(1,10,10,dtype=int) # 1) bal
#runs_to_plot = np.linspace(11,14,4,dtype=int) # 2) unbal
#runs_to_plot = np.linspace(15,16,2,dtype=int) # 3) bal


## eps=.5
# BCM SAME
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_bcm_same_200clients_5seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,34,34,dtype=int) # 1)+2) bal+unbal

# BCM SPLIT
main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_bcm_split_200clients_5seeds_eps05_runs/'
runs_to_plot = np.linspace(1,21,21,dtype=int) # 1)-4) bal+unbal


# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_200clients_5seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,8,8,dtype=int) # 1)+2) bal
#runs_to_plot = np.linspace(9,16,8,dtype=int) # 3) unbal

# LFA sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_200clients_5seeds_eps05_sec_runs/'
#runs_to_plot = [1] # 1) bal
#runs_to_plot = np.linspace(2,17,16,dtype=int)#[1] # 2) unbal

# LOCAL PVI sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_200clients_5seeds_eps05_sec_runs/'
#runs_to_plot = [3] # 1) bal
#runs_to_plot = np.linspace(4,11,8,dtype=int) # 2) unbal
#runs_to_plot = np.linspace(12,15,4,dtype=int) # 3) unbal


##########################################################
### ADULT BNN 10 client 1 seed trade off 2 runs:
## NONDP BATCHES
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_nondp_batches_1seeds_trade_off2_runs/'
#runs_to_plot = np.linspace(1,160,160,dtype=int) # 1)
#runs_to_plot = np.linspace(185,232,48,dtype=int) # 2)
#runs_to_plot = np.linspace(224,232,9,dtype=int) # 2) osia


## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_1seeds_trade_off2_runs/'
#runs_to_plot = np.linspace(1,84,84,dtype=int) # 1)


##########################################################
#### ADULT 1BNN 10 client 5 seed runs:


### EPS=2 runs
## COMMS BASELINE
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_comms_baseline_5seeds_eps2_runs/'
#runs_to_plot = [1]

## BCM SAME
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_bcm_same_10clients_5seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,5,5,dtype=int) # 1)-5) bal & unbal

## BCM SPLIT
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_bcm_split_10clients_5seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,10,10,dtype=int) # 1)-4) bal & unbal

## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_dpsgd_10clients_5seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,12,12,dtype=int) # 1)&2) bal&unbal

## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_5seeds_eps2_runs/'
#runs_to_plot = [1] # 1) bal
#runs_to_plot = [2] # 2) bal nat grad
# KALE runs
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/kale_adult_1bnn_lfa_10clients_5seeds_eps2_runs/'
#runs_to_plot =  np.linspace(1,6,6,dtype=int) # 4) unbal
#runs_to_plot =  np.linspace(7,10,4,dtype=int) # 5) bal 30 globals


## LFA fixes
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_5seeds_eps2_fix_runs/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_5seeds_eps2_fix2_runs/'
# KALE runs
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/kale_adult_1bnn_lfa_10clients_5seeds_eps2_fix_runs/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/kale_adult_1bnn_lfa_10clients_5seeds_eps2_fix2_runs/'
#runs_to_plot = [6]
#runs_to_plot = np.linspace(1,6,6,dtype=int) # 
#runs_to_plot = [4,6,7,8]




## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_5seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,12,12,dtype=int) # 1)-2)  bal&unbal
#runs_to_plot = np.linspace(13,36,16,dtype=int) # 3)-4)  bal&unbal
#runs_to_plot = np.linspace(37,40,4,dtype=int) # 5) bal
#runs_to_plot = np.linspace(41,48,8,dtype=int) # 6) bal
#runs_to_plot = np.linspace(49,60,12,dtype=int) # 7) bal
#runs_to_plot = np.linspace(61,72,12,dtype=int) # 8) bal


## LOCAL PVI fixes
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_5seeds_eps2_fix_runs/'
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_5seeds_eps2_fix2_runs/'
#runs_to_plot = [2,3]
#runs_to_plot = np.linspace(1,7,7,dtype=int) # 


### EPS=1 runs
## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_dpsgd_10clients_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,2,2,dtype=int) # 1) bal
#runs_to_plot = np.linspace(3,6,4,dtype=int) # 2) unbal

## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,2,2,dtype=int) # 1) bal

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_5seeds_eps1_runs/'
#runs_to_plot = [1] # 1) bal, hajonnut
#runs_to_plot = np.linspace(2,3,2,dtype=int) # 2) bal, yhä hajonneita
#runs_to_plot = [4] # 3) bal, hajonnut

## LOCAL PVI sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_5seeds_eps1_sec_runs/'
#runs_to_plot = np.linspace(1,16,16,dtype=int) # 1) unbal


### EPS=.5 runs
## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_dpsgd_10clients_5seeds_eps05_runs/'
#runs_to_plot = [1] # 1) bal
#runs_to_plot = np.linspace(2,3,2,dtype=int) # 2) unbal


##########################################################
#### ADULT 1BNN 10 client 1 seed runs:


### only clip runs
## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_1seeds_clip_runs/'
#runs_to_plot = np.linspace(1,144,144,dtype=int) # 1) bal

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_1seeds_clip_runs/'
#runs_to_plot = np.linspace(1,162,162,dtype=int) # 1) bal


### EPS=5 runs
## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_1seeds_eps5_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) bal
#runs_to_plot = np.linspace(73,88,16,dtype=int) # 2) bal

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_1seeds_eps5_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int) # 1) bal

### EPS=2 runs
## COMMS BASELINE
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_comms_baseline_1seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,354,354,dtype=int) # 1)-3) bal 1k,2k,4k runs
#runs_to_plot = [320]

## BCM SAME
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_bcm_same_10clients_1seeds_eps2_runs/'
#runs_to_plot = np.concatenate([np.linspace(1,128,128,dtype=int),np.linspace(407,442,36,dtype=int)]) # 1)-3) bal 1k,2k,4k runs
#runs_to_plot = np.concatenate([np.linspace(191,406,216,dtype=int),[443,444]]) # 4)-6) unbal


## BCM SPLIT
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_bcm_split_10clients_1seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,324,324,dtype=int) # 1)-3) bal&unbal 1k,2k,4k runs


## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_dpsgd_10clients_1seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) bal
#runs_to_plot = np.linspace(73,216,144,dtype=int) # 2) unbal

## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_1seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,48,48,dtype=int) # 1) bal
#runs_to_plot = np.linspace(49,120,72,dtype=int) # 2) bal
#runs_to_plot = np.linspace(121,257,137,dtype=int) # 3) unbal

#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/kale_adult_1bnn_lfa_10clients_1seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,54,54,dtype=int) # 4) bal, 15 globals
#runs_to_plot = np.linspace(55,198,144,dtype=int) # 5) unbal, 15 globals
#runs_to_plot = np.linspace(199,360,162,dtype=int) # 6) bal&unbal, 10 globals
#runs_to_plot = np.linspace(361,464,104,dtype=int) # 7) bal&unbal, 20 globals


## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_1seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int) # 1) bal
#runs_to_plot = np.linspace(109,252,144,dtype=int) # 2) unbal
#runs_to_plot = np.linspace(253,468,216,dtype=int) # 4) bal&unbal
#runs_to_plot = np.linspace(469,522,54,dtype=int) # 5) bal


#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/kale_adult_1bnn_local_pvi_10clients_1seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 3) bal

# 3 seed quick test
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_3seeds_eps2_runs/'
#runs_to_plot = np.linspace(1,360,360,dtype=int) # 1) bal&unbal


### NATGRAD runs, eps=2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/kale_adult_1bnn_lfa_10clients_1seeds_eps2_natgrad_runs/'
#runs_to_plot = np.linspace(1,118,118,dtype=int) # 1)-2) bal+unbal


### EPS=1 runs
## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_dpsgd_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,54,54,dtype=int) # 1)
#runs_to_plot = np.linspace(55,77,24,dtype=int) # 2)
#runs_to_plot = np.linspace(78,94,17,dtype=int) # 3)
#runs_to_plot = np.linspace(95,130,36,dtype=int) # 4) unbal

## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,144,144,dtype=int) # 1)
#runs_to_plot = np.linspace(145,576,432,dtype=int) # 2)
#runs_to_plot = np.linspace(577,648,72,dtype=int) # 3) NOTE: tämä on NON-DP!
#runs_to_plot = np.linspace(649,720,72,dtype=int) # 4)
#runs_to_plot = np.linspace(721,784,64,dtype=int) # 5)
#runs_to_plot = np.linspace(785,800,16,dtype=int) # 6)
#runs_to_plot = np.linspace(801,1088,288,dtype=int) # 7) unbal
#runs_to_plot = np.linspace(1089,1104,16,dtype=int) # 7) unbal pidempi

## LFA sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_lfa_10clients_1seeds_eps1_sec_runs/'
#runs_to_plot = np.linspace(1,288,288,dtype=int) # 1) bal
#runs_to_plot = np.linspace(289,360,72,dtype=int) # 2) bal

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,144,144,dtype=int) # 1)
#runs_to_plot = np.linspace(145,288,144,dtype=int) # 2)
#runs_to_plot = np.linspace(289,396,108,dtype=int) # 3)
#runs_to_plot = np.linspace(397,491,95,dtype=int) # 4)
#runs_to_plot = np.linspace(492,539,48,dtype=int) # 5) bal, onko vakaampi
#runs_to_plot = np.linspace(540,547,8,dtype=int) # 6) bal, vakaampiko taas

## LOCAL PVI sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_local_pvi_10clients_1seeds_eps1_sec_runs/'
#runs_to_plot = np.linspace(1,288,288,dtype=int) # 1)
#runs_to_plot = np.linspace(504,935,432,dtype=int) # 2) unbal

### EPS=.5 runs
## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_1bnn_dpsgd_10clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,36,36,dtype=int) # 1) bal
#runs_to_plot = np.linspace(37,108,72,dtype=int) # 2) unbal



##########################################################
#### ADULT 10 client 5 seed runs:
### EPS=1 runs

## COMMS BASELINE
# same for all number of clients
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_comms_baseline_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,2,2,dtype=int) # 1) bal & unbal: no difference with baseline

## BCM SAME BASELINE
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_bcm_same_10clients_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,6,6,dtype=int) # 1) bal & unbal
#runs_to_plot = [7] # 3) unbal

## BCM SPLIT BASELINE
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_bcm_split_10clients_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,2,2,dtype=int) # 1)-2) bal
#runs_to_plot = np.linspace(3,9,7,dtype=int) # 3)-5) unbal


## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_10clients_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,8,8,dtype=int) # 1) unbal
#runs_to_plot = [9] # 2) bal


## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_5seeds_eps1_runs/'
#runs_to_plot = [1] # 1) bal
#runs_to_plot =  np.linspace(2,9,8,dtype=int) # 2)-3) unbal

## LFA with secure aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_5seeds_eps1_sec_runs/'
#runs_to_plot =  np.linspace(1,2,2,dtype=int) # 1) bal
#runs_to_plot =  np.linspace(3,10,8,dtype=int) # 2) unbal
#runs_to_plot =  np.linspace(11,14,4,dtype=int) # 3) unbal

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_5seeds_eps1_runs/'
#runs_to_plot =  np.linspace(1,2,2,dtype=int) # 1) bal
#runs_to_plot =  np.linspace(3,6,4,dtype=int) # 2) unbal

## LOCAL PVI with secure aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_5seeds_eps1_sec_runs/'
#runs_to_plot =  np.linspace(1,4,4,dtype=int) # 1) bal
#runs_to_plot =  np.linspace(5,8,4,dtype=int) # 2) unbal


### EPS=.5 runs

## COMMS BASELINE
# same for all number of clients
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_comms_baseline_5seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,2,2,dtype=int) # 1) bal & unbal: no difference with baseline

## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_10clients_5seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,8,8,dtype=int) # 1) unbal
#runs_to_plot = [9] # 2) bal

## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_5seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,9,9,dtype=int) # 1) bal
#runs_to_plot = np.linspace(10,13,4,dtype=int) # 2) unbal

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_5seeds_eps05_runs/'
#runs_to_plot = [2] # 1) bal
#runs_to_plot = np.linspace(3,8,6,dtype=int) # 2) unbal


## LFA with secure aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_5seeds_eps05_sec_runs/'
#runs_to_plot = np.linspace(1,6,6,dtype=int) # 1) bal
#runs_to_plot = np.linspace(7,10,4,dtype=int) # 2) unbal

## LOCAL PVI with secure aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_5seeds_eps05_sec_runs/'
#runs_to_plot = [1] # 1) bal
#runs_to_plot = np.linspace(3,8,6,dtype=int) # not done yet


##########################################################
#### ADULT 10 client 1 seed runs:
### EPS=1 runs

## baseline COMMS 
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_comms_baseline_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1)-2) bal
#runs_to_plot = np.concatenate( (np.linspace(1,72,72,dtype=int),np.linspace(109,146,36,dtype=int)))  # 1)-2),4) bal

## baseline BCM SAME
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_bcm_same_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.concatenate([np.linspace(1,216,216,dtype=int),np.linspace(658,765,108,dtype=int)]) # 1)-3)+7)-9) bal
#runs_to_plot = np.concatenate([np.linspace(217,657,441,dtype=int), np.linspace(658,765,108,dtype=int)]) # 4)-6)+7)-9) unbal


## baseline BCM SPLIT
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_bcm_split_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.concatenate([np.linspace(1,216,216,dtype=int), np.linspace(649,756,108,dtype=int)]) # 1)-3)+7)-9) bal
#runs_to_plot = np.concatenate([np.linspace(217,648,432,dtype=int),np.linspace(649,756,108,dtype=int)]) # 4)-7)-9) unbal


## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) bal
#runs_to_plot = np.linspace(73,216,144,dtype=int) # 2) unbal

## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int) # 1) bal
#runs_to_plot = np.linspace(109,216,108,dtype=int) # 2) unbal

## LFA sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_1seeds_eps1_sec_runs/'
#runs_to_plot = np.linspace(1,54,54,dtype=int) # 1) bal
#runs_to_plot = np.linspace(55,162,108,dtype=int) # 2) unbal

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int) # 1)
#runs_to_plot = np.linspace(109,216,108,dtype=int) # 2)
#runs_to_plot = np.linspace(217,288,72,dtype=int) # 3) unbal



## EPS=0.5 runs
## baseline COMMS 
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_comms_baseline_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int) # 1)-3) bal


## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_10clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) bal
#runs_to_plot = np.linspace(73,216,144,dtype=int) # 2) unbal

## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) bal
#runs_to_plot = np.linspace(73,144,72,dtype=int) # 2) unbal
#runs_to_plot = np.linspace(145,216,72,dtype=int) # 3) unbal lisää

## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int) # 1)
#runs_to_plot = np.linspace(109,162,54,dtype=int) # 2)
#runs_to_plot = np.linspace(163,197,36,dtype=int) # 3)

## LOCAL PVI with secure aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_1seeds_eps05_sec_runs/'
#runs_to_plot = np.linspace(1,36,36,dtype=int) # 1) bal
#runs_to_plot = np.linspace(37,144,108,dtype=int) # 2) unbal



# EPS=.2 runs
## DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_dpsgd_10clients_1seeds_eps02_runs/'
#runs_to_plot = np.linspace(1,37,37,dtype=int)#[1] # 1) seq server param testing, 
#runs_to_plot = np.linspace(38,43,6,dtype=int)#[1] # 2) kuten 1) mutta sync server, 100 globals
#runs_to_plot = np.linspace(44,49,6,dtype=int)#[1] # 3) kuten 2) 200 globals
#runs_to_plot = np.linspace(50,55,6,dtype=int)#[1] # 4) 200 globals, seq server


## LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_1seeds_eps02_runs/'
#runs_to_plot = np.linspace(1,216,216,dtype=int)#[1] # 1) 20 globals, liian iso q
#runs_to_plot = np.linspace(217,459,243,dtype=int)#[1] # 2) 10 globals, liian iso q
#runs_to_plot = np.linspace(460,555,96,dtype=int)#[1] # 3) 10 globals, pieni q
#runs_to_plot = np.linspace(556,681,126,dtype=int)#[1] # 4) 20 globals, pieni q
#runs_to_plot = np.linspace(682,713,32,dtype=int)#[1] # 5) unbal data, 10 globals, pieni q
#runs_to_plot = np.linspace(714,761,48,dtype=int)#[1] # 6) puuttuva konfa 3)sta: bal data, 10 globals, pieni q

# nondp runs
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_lfa_10clients_1seeds_nondp_runs/'
#runs_to_plot = np.linspace(1,84,84,dtype=int) # 1) 


## LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_1seeds_eps02_runs/'
##runs_to_plot = np.linspace(1,72,72,dtype=int)#[1] # 1) 20 globals, liian iso q
##runs_to_plot = np.linspace(73,144,72,dtype=int)#[1] # 2) 10 globals, liian iso q
##runs_to_plot = np.linspace(145,249,105,dtype=int)#[1] # 3) 10 globals, pieni q
#runs_to_plot = np.linspace(1,96,96,dtype=int)#[1] # 4) unbal data, 10 globals, pieni q, bugi fiksattu
#runs_to_plot = np.linspace(97,192,96,dtype=int)#[1] # 3) unbal data, 10 globals, pieni q, bugi fiksattu

# nondp runs
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/adult_local_pvi_10clients_1seeds_nondp_runs/'
#runs_to_plot = np.linspace(1,402,402,dtype=int) # 1) 


##########################################################

### MIMIC3 runs:
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



##########################################################
#### MIMIC3 balanced data testing

### nondp bcm baselines
# BCM SAME
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_bcm_same_10clients_1seeds_nondp_runs/'
#runs_to_plot = np.linspace(1,135,135,dtype=int) # 1)& 2) bal


### 10 clients, baselines, 1 seed
## EPS=1 runs

# BCM SAME
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_bcm_same_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.concatenate([np.linspace(1,144,144,dtype=int),np.linspace(433,540,108,dtype=int)]) # 1)-4)+9)-11) bal
#runs_to_plot = np.concatenate([np.linspace(145,432,288,dtype=int),np.linspace(433,540,108,dtype=int)]) # 5)-11) unbal

# BCM SPLIT
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_bcm_split_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.concatenate([np.linspace(1,72,72,dtype=int),np.linspace(217,540,324,dtype=int)]) # 1)-2)+5)-7) bal
#runs_to_plot = np.linspace(73,540,468,dtype=int) # 3)-7) unbal


# COMMS BASELINE
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_comms_baseline_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,48,48,dtype=int) # 1) bal
#runs_to_plot = np.linspace(49,96,48,dtype=int) # 2) bal
#runs_to_plot = np.linspace(97,132,36,dtype=int) # 3) bal
#runs_to_plot = np.linspace(133,168,36,dtype=int) # 4) bal
#runs_to_plot = np.linspace(169,204,36,dtype=int) # 5) bal

### 10 clients, baselines, 5 seed
## EPS=1 runs

# COMMS BASELINE
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_comms_baseline_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,8,8,dtype=int) # 1) bal

# BCM SAME
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_bcm_same_10clients_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,2,2,dtype=int) # 1)-2) bal
#runs_to_plot = np.linspace(3,7,5,dtype=int) # 3)-7) unbal

# BCM SPLIT
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_bcm_split_10clients_5seeds_eps1_runs/'
#runs_to_plot = np.concatenate([np.linspace(1,2,2,dtype=int),[7]]) # 1)-2)+5) bal
#runs_to_plot = np.linspace(3,8,6,dtype=int) # 3)-6) unbal

### 10 clients, 1 seed
## EPS=1 runs

# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_dpsgd_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) bal
#runs_to_plot = np.linspace(73,144,72,dtype=int) # 2) bal, eri q
#runs_to_plot = np.linspace(145,288,144,dtype=int) # 3) unbal

# LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_lfa_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,216,216,dtype=int) # 1) bal
#runs_to_plot = np.linspace(217,264,48,dtype=int) # 2) bal
#runs_to_plot = np.linspace(265,552,288,dtype=int) # 3) unbal
#runs_to_plot = np.linspace(553,576,24,dtype=int) # 4) bal
#runs_to_plot = np.linspace(577,756,180,dtype=int) # 5) unbal

# LFA sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_lfa_10clients_1seeds_eps1_sec_runs/'
#runs_to_plot = np.linspace(1,81,81,dtype=int) # 1) bal
#runs_to_plot = np.linspace(82,339,258,dtype=int) # 2)-3) unbal
#runs_to_plot = np.linspace(340,393,54,dtype=int) # 4) bal nat grad
#runs_to_plot = np.linspace(394,465,72,dtype=int) # 5) unbal nat grad



# LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_local_pvi_10clients_1seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,288,288,dtype=int) # 1) bal
#runs_to_plot = np.linspace(289,432,144,dtype=int) # 2) bal
#runs_to_plot = np.linspace(433,864,432,dtype=int) # 2) unbal

# LOCAL PVI sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_local_pvi_10clients_1seeds_eps1_sec_runs/'
#runs_to_plot = np.linspace(1,108,108,dtype=int) # 1) bal
#runs_to_plot = np.linspace(210,425,216,dtype=int) # 2) unbal

## EPS=.5 runs
# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_dpsgd_10clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,72,72,dtype=int) # 1) bal
#runs_to_plot = np.linspace(73,216,144,dtype=int) # 2) unbal

# LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_lfa_10clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,216,216,dtype=int) # 1) bal
#runs_to_plot = np.linspace(217,504,288,dtype=int) # 2) unbal

# LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_local_pvi_10clients_1seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,144,144,dtype=int) # 1) bal
#runs_to_plot = np.linspace(145,360,216,dtype=int) # 2) unbal


### balanced mimic, 10 clients, 5 seeds
## EPS=1 runs

# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_dpsgd_10clients_5seeds_eps1_runs/'
#runs_to_plot = [1,6] # 1) & 3) bal
#runs_to_plot = np.linspace(2,5,4,dtype=int) # 2) unbal

# LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_lfa_10clients_5seeds_eps1_runs/'
#runs_to_plot = [1] # 1) bal
#runs_to_plot = np.linspace(2,7,6,dtype=int) # 2) & 3) unbal

# LFA sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_lfa_10clients_5seeds_eps1_sec_runs/'
#runs_to_plot = np.linspace(1,4,4,dtype=int) # 1) unbal
#runs_to_plot = np.linspace(5,8,4,dtype=int) # 2) unbal
#runs_to_plot = np.linspace(9,16,8,dtype=int) # 3) unbal
#runs_to_plot = np.linspace(17,24,8,dtype=int) # 4) unbal
#runs_to_plot = np.linspace(25,28,4,dtype=int) # 5) bal

# LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_local_pvi_10clients_5seeds_eps1_runs/'
#runs_to_plot = np.linspace(1,16,16,dtype=int) # 1) unbal
#runs_to_plot = [17,18,19] # 1) bal

# LOCAL PVI sec aggr
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_local_pvi_10clients_5seeds_eps1_sec_runs/'
#runs_to_plot = np.linspace(1,2,2,dtype=int) # 1) unbal
#runs_to_plot = np.linspace(3,14,12,dtype=int) # 2) bal

## EPS=.5 runs
# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_dpsgd_10clients_5seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,16,16,dtype=int) # 1) unbal
#runs_to_plot = [17] # 1) bal

# LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_lfa_10clients_5seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,36,36,dtype=int) # 1) & 2) unbal
#runs_to_plot = [37] # 3) bal


# LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mimic_bal_local_pvi_10clients_5seeds_eps05_runs/'
#runs_to_plot = np.linspace(1,8,8,dtype=int) # 1) unbal
#runs_to_plot = np.linspace(9,10,2,dtype=int) # 2) bal



##########################################################
### MNIST testing
## balanced data, b=1, eps=2
# DPSGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_dpsgd_bal_10clients_b1_eps2_runs/'
#runs_to_plot = np.linspace(1,54,54,dtype=int)#[1] # 1) # 200 globals
#runs_to_plot = np.linspace(55,59,5,dtype=int)#[1] # 2) # 400 globals

## balanced data, b=1, eps=5
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_dpsgd_bal_10clients_b1_eps5_runs/'
#runs_to_plot = np.linspace(1,54,54,dtype=int)#[1] # 1) # 400 globals, 100 hidden units
#runs_to_plot = np.linspace(56,63,8,dtype=int)#[1] # 1) # 200 globals, lr schedule, 200 hidden units


## balanced data, b=1, eps=10
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_dpsgd_bal_10clients_b1_eps10_runs/'
#runs_to_plot = np.linspace(1,54,54,dtype=int)#[1] # 1) # 400 globals, 100 hidden units
#runs_to_plot = np.linspace(55,63,8,dtype=int)#[1] # 1) # 200 globals, 200 hidden units

## balanced data, b=5, eps=2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_dpsgd_bal_10clients_b5_eps2_runs/'
#runs_to_plot = np.linspace(1,18,18,dtype=int)#[1] # 1) # 200-400 globals

# unbalanced data, b=1, eps=2
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_dpsgd_unbal_100clients_b1_eps2_runs/'
#runs_to_plot = np.linspace(1,25,25,dtype=int)#[1] # 1) # 200-400 globals




# NONDP MNIST: balanced data, b=1
# SGD
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_nondp_bal_10clients_b1_runs/'
#runs_to_plot = np.append(np.linspace(25,71,47,dtype=int),73)  # 1) & 2): 400 globals, seq & sync server, full epochs
#runs_to_plot =  np.linspace(76,99,24,dtype=int) # 3) 400 globals, seq server, batches
#runs_to_plot = [72,74,75] # 4) # 400 globals, seq server, batches, freeze var test
#runs_to_plot =  np.linspace(100,111,12,dtype=int) # 5) # 200 globals, seq server, batches, lr scheduler

# LFA
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_lfa_bal_10clients_b1_epsinfty_runs/'
#runs_to_plot = np.linspace(1,28,28,dtype=int)  # 1)
#runs_to_plot = np.linspace(29,64,36,dtype=int)  # 2)
#runs_to_plot = np.linspace(65,80,16,dtype=int)  # 3)

# LOCAL PVI
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_local_pvi_bal_10clients_b1_epsinfty_runs/'
#runs_to_plot = np.linspace(1,36,36,dtype=int)  # 1)
#runs_to_plot = np.linspace(37,72,36,dtype=int)  # 2)


# mixed nondp-sgd
#main_folder = '/Users/mixheikk/Documents/git/DP-PVI/pytorch-code-results/mnist_bnn_mixed_dpsgd_bal_10clients_b1_runs/'
#runs_to_plot = np.linspace(2,2,1,dtype=int)#[1] # 1) # non-dp mixed dpsgd test



#print(runs_to_plot)
#print(len(runs_to_plot))
#sys.exit()


# where to save all plots
fig_folder = 'res_plots/'

restrictions = OD()
restrictions['dp_mode'] = None#['nondp_batches']
restrictions['n_global_updates'] = None#[400]
restrictions['n_steps'] = None#[100]
restrictions['batch_size'] = None#[5]
restrictions['dp_sigma'] = None#[23.15]

restrictions['dp_C'] = None#[20]
restrictions['sampling_frac_q'] = None#[1e-2]
restrictions['learning_rate'] = None#[5e-3]
restrictions['damping_factor'] = None#[5e-3]
restrictions['init_var'] = None#[1e-3]
restrictions['freeze_var_updates'] = None#[5]

plot_legends = 0

# possible balance settings: (0,0), (.7,-3), (.75,.95)
restrictions['data_bal_rho'] = [.7]
restrictions['data_bal_kappa'] = [-3.]

#runs_to_plot = [52,324]
#runs_to_plot = np.delete(runs_to_plot, np.where(runs_to_plot == 281))

# save to disk (or just show)
to_disk = 0

plot_test_error = 1 # if 0 plot training err instead

#dataset_name = 'mimic3_bal'
dataset_name = 'adult'
#dataset_name = 'mnist'
#dataset_name = 'mimic3'
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
#fig_name = "{}_1BNN_dpsgd_10clients_1seeds_bal_b1_eps2.pdf".format(dataset_name)
#'''
#fig_name = "{}_dpsgd_10clients_eps02_bal({},{}).pdf".format(dataset_name, restrictions['data_bal_rho'],restrictions['data_bal_kappa'])
fig_name = "{}_lfa_10clients_nondp_bal({},{})_q{}.pdf".format(dataset_name, restrictions['data_bal_rho'],restrictions['data_bal_kappa'],restrictions['sampling_frac_q'])


# set to None to skip:


if dataset_name == 'mimic3_bal':
    ylims= ((.5,.8),(-.85,-.5), None, None)
    if plot_test_error:
        baseline_acc = 0.761
        baseline_logl = -.517
    else:
        baseline_acc = None
        baseline_logl = None
elif dataset_name == 'mimic3':
    ylims= ((.87,.9),(-.55,-.26), None, None)
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
    #ylims= ((.94,1.),(-.2,-.0), None, None)
    ylims= ((.7,.99),(-.8,-.02), None, None)
    #ylims = (None,None,None,None)
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
add_labels['dp_C'] = 'C'
#add_labels['dp_sigma'] = 'sigma'
#add_labels['n_steps'] = 'steps'
#add_labels['batch_size'] = 'b'
add_labels['sampling_frac_q'] = 'q'
add_labels['learning_rate'] = 'lr'
add_labels['damping_factor'] = 'damping'
add_labels['init_var'] = 'var'
#add_labels['server'] = 'server'
add_labels['freeze_var_updates'] = 'var freeze'
#add_labels['privacy_calculated'] = 'DP epochs'
#add_labels['data_bal_rho'] = 'rho'
#add_labels['data_bal_kappa'] = 'kappa'
add_labels['init_var'] = 'init_var'



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
                pass



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
    #sys.exit()

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


# plot comparisons between given runs: all in same plot
plot_comparisons = True
if plot_comparisons:
    best_acc = 0
    best_logl = -1e6
    best_confs = [None,None]
    best_ids = [-1,-1]

    # check restrictions
    list_to_print = []
    for i_run in runs_to_plot:
        print_this = True
        for k in restrictions:
            if restrictions[k] is not None and all_res['config'][str(i_run)][k] not in restrictions[k]:
                print_this = False
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
            best_ids[0] = i_run
        if np.amax(all_res['validation_res'][str(i_run)]['logl'].mean(-1)) > best_logl:
            best_logl = np.amax(all_res['validation_res'][str(i_run)]['logl'].mean(-1))
            best_confs[1] = config
            best_ids[1] = i_run

        x = np.linspace(1,config['n_global_updates'],config['n_global_updates'])

        # check whats happening
        #print( all_res['validation_res'][str(i_run)]['acc'])
        #print( all_res['validation_res'][str(i_run)]['logl'])

        # testing error plot
        if plot_test_error:
            axs[0,0].errorbar(x, np.nanmean(all_res['validation_res'][str(i_run)]['acc'],-1 ), 
                    yerr= 2*all_res['validation_res'][str(i_run)]['acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    #yerr= 2*np.nanstd(all_res['validation_res'][str(i_run)]['acc'],-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label, 
                    color=colors[i_line%len(colors)],
                    marker='*')
            axs[1,0].errorbar(x, np.nanmean(all_res['validation_res'][str(i_run)]['logl'],-1), 
                    yerr= 2*np.nanstd(all_res['validation_res'][str(i_run)]['logl'],-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                    label=cur_label,
                    color=colors[i_line%len(colors)],
                    marker='*'
                    )

            #posneg_measures = ['avg_prec_score','balanced_acc','f1_score']
            # balanced acc
            #'''
            if dataset_name == 'mnist':
                axs[0,1].errorbar(x, all_res['validation_res'][str(i_run)]['balanced_acc'].mean(-1), 
                        yerr= 2*all_res['validation_res'][str(i_run)]['balanced_acc'].std(-1)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                        label=cur_label,
                        color=colors[i_line%len(colors)]
                        )
            # avg ROC curve at best logl global update
            else:
                try:
                    axs[0,1].plot( 1-all_res['validation_res'][str(i_run)]['TNR'].mean(0), all_res['validation_res'][str(i_run)]['TPR'].mean(0), 
                            #yerr= 2*all_res['validation_res'][str(i_run)]['balanced_acc'].std(0)/np.sqrt(config['n_rng_seeds']), # 2*SEM errorbar over seeds
                            label=cur_label,
                            color=colors[i_line%len(colors)]
                            )
                #plt.plot(1-all_res['validation_res'][str(i_run)]['TNR'][i_seed,:], all_res['validation_res'][str(i_run)]['TPR'][i_seed,:] )
                except:
                    pass

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
                    )

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
            #axs[0,0].legend(loc='lower right')
            #axs[1,0].legend(loc='lower right')
            #axs[0,1].legend(loc='lower right')
            axs[1,1].legend(loc='lower right')
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

        if plot_test_error:
            plt.suptitle(f"{dataset_name} dataset mean with 2*SEM over {config['n_rng_seeds']} runs" + cur_title)
        else:
            plt.suptitle(f"{dataset_name} dataset training err mean with 2*SEM over {config['n_rng_seeds']} runs" + cur_title)

    print(f'\nBest acc: {best_acc:.5f} found with config:\n{best_confs[0]}')
    if best_confs[1] == best_confs[0]:
        best_confs[1] = None
    print(f'\nBest logl: {best_logl:.5f} found with config:\n{best_confs[1]}')
    print(f'Best run ids: {best_ids}')

    if to_disk:
        plt.savefig(fig_folder+fig_name)
    
    else:
        plt.show()


