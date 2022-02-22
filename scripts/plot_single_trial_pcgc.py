#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 18:22:28 2022
In this script we plot single trial pairwise GC
@author: guime
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mne.stats import fdr_correction
from src.preprocessing_lib import EcogReader, single_pfc_stat
from src.input_config import args
from scipy.io import loadmat
from scipy.stats import ranksums
from pathlib import Path
from statsmodels.stats.multitest import fdrcorrection

#%% Read pcgc data


# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs'];
# Load functional connectivity matrix
result_path = Path('../results')

fname = 'pairwise_fc.mat'
fc_path = result_path.joinpath(fname)

fc = loadmat(fc_path)
fc = fc['dataset']


#%% Plot z score and  signigicance
#%matplotlib qt
subject = 'DiAs'
z, sig, pval = single_pfc_stat(fc, cohort, subject =subject, single='single_F', 
                    baseline='Rest', alternative='two-sided')


reader = EcogReader(args.data_path, subject=subject)
# Read visual channels 
df_visual = reader.read_channels_info(fname='visual_channels.csv')
populations = df_visual['group']

(n,n,ncomp) = z.shape
f, ax = plt.subplots(ncomp,2)
zmax = np.amax(z)
zmin = np.amin(z)
for icomp in range(ncomp):
    g = sns.heatmap(z[:,:,icomp], vmin=zmin, vmax=zmax, xticklabels=populations,
                        yticklabels=populations, cmap='YlOrBr', ax=ax[icomp,0])
    g.set_yticklabels(g.get_yticklabels(), rotation = 90)
    # Position xticks on top of heatmap
    ax[icomp, 0].xaxis.tick_top()
    ax[icomp, 0].set_ylabel('Target')
    ax[0,0].set_title(' Z score')
    g = sns.heatmap(sig[:,:,icomp], xticklabels=populations,
                        yticklabels=populations, cmap='YlOrBr', ax=ax[icomp,1])
    g.set_yticklabels(g.get_yticklabels(), rotation = 90)
    # Position xticks on top of heatmap
    ax[icomp, 1].xaxis.tick_top()
    ax[0,1].set_title('Significance')





