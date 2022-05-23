"""script for plotting comparisons of parameter traces
"""

import sys

from collections import OrderedDict as OD
from matplotlib import pyplot as plt
import numpy as np



folder = 'res_plots/param_traces/saved_params/'

filenames = { 'nondp_epochs' : "saved_params_nondp_epochs_globals20_steps40_clients10.npz",
              'lfa' : "saved_params_lfa_globals20_steps40_clients10.npz",
              'local pvi' : "saved_params_local_pvi_globals20_steps40_clients10.npz",
            }


# read param traces
res = OD()
for k in filenames:
    res[k] = np.load(folder+filenames[k])
    


# plot
nx = res['nondp_epochs']['loc_params'].shape[0]
x = np.linspace(0,nx-1,nx)
fig,axs = plt.subplots(2,1)
for k in filenames:
    #if k == 'nondp_epochs':
    #    continue
    
    #print(res[k]['loc_params'].shape)
    axs[0].plot(np.linalg.norm(np.sort(res[k]['loc_params'],1)-np.sort(res['nondp_epochs']['loc_params'][-1,:]),ord=2,axis=1), label=k )
    axs[1].plot(np.linalg.norm(np.sort(res[k]['scale_params'],1)-np.sort(res['nondp_epochs']['scale_params'][-1,:]),ord=2,axis=1), label=k )

axs[0].set_ylabel('Loc param distance')
axs[1].set_ylabel('Scale param distance')
axs[1].set_xlabel('Global update')
axs[0].legend()
axs[1].legend()
plt.show()

# eli halutaan y=normi sorted params erotukselle, x=globals, loc ja scale erikseen





