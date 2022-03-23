#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 22:25:53 2022

@author: guime
"""
from src.input_config import args
from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.plotting_lib import full_stim_multi_pfc, full_stim_multi_gfc
from src.plotting_lib import plot_single_trial_pfc, plot_single_trial_gfc
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#%%

home = Path.home()
figpath = home.joinpath('thesis','overleaf_project','figures')
result_path = Path('../results')
# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
subject = 'DiAs'

#%%

fname = 'multi_trial_bivariate_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
F = 'pGC'
ncdt = 3
populations = ['F','R']
vmax = 2.5
rotation=90 
tau_x=0.5
tau_y=0.8
fig, ax = plt.subplots(ncdt)
for c in range(ncdt): # Consider resting state as baseline
        condition =  fc[0,c]['condition'][0]
        # Functional connectivity matrix
        baseline = fc[0,0][F]['f'][0][0]
        f = fc[0,c][F]['f'][0][0]
        f = f/baseline
        f = np.log(f)
        # Significance
        sig = fc[0,c][F]['sig'][0][0]
        g = sns.heatmap(f, xticklabels=populations, vmin=-vmax, vmax=vmax,
                        yticklabels=populations, cmap='bwr', ax=ax[c])
        g.set_yticklabels(g.get_yticklabels(), rotation = 90)
        # Position xticks on top of heatmap
        ax[c].xaxis.tick_top()
        ax[c].set_ylabel(condition)
        
        # Position xticks on top of heatmap
        ax[c].xaxis.tick_top()
        # Plot statistical significant entries
        for y in range(f.shape[0]):
            for x in range(f.shape[1]):
                if sig[y,x] == 1:
                    ax[c].text(x + tau_x, y + tau_y, '*',
                             horizontalalignment='center', verticalalignment='center',
                             color='k')
                else:
                    continue                 
        plt.tight_layout()
