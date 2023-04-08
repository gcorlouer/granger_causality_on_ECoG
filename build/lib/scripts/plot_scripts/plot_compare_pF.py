#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:44:13 2022

@author: guime
"""

import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path

#%%

cohort = ['AnRa',  'ArLa', 'DiAs']
# Useful paths
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')
fname = 'compare_condition_pGC.mat'
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
F = dataset['pGC']

#%% Plotting functions

def plot_pair_zscore(F, cohort, data_path,
                       vmin = -5, vmax=5, tau_x=0.5, tau_y=0.8):
    """
    We plot Z score from comparing permutation pF in condition 1 with
    condition 2.
    """
    comparisons = ['FvsR', 'PvsR', 'FvsP']
    nsub = len(cohort)
    ncomp = len(comparisons)
    fig, ax = plt.subplots(nsub, ncomp)
    # Loop over subject and comparison to plot Z score heatmap
    for s, subject in enumerate(cohort):
        for c, comparison in enumerate(comparisons):
            # Get visual channels
            reader = EcogReader(data_path, subject=subject)
            df_visual = reader.read_channels_info(fname='visual_channels.csv')
            # Find retinotopic and face channels indices 
            R_idx = df_visual.index[df_visual['group']=='R'].tolist()
            F_idx = df_visual.index[df_visual['group']=='F'].tolist()
            RF_idx = np.array(R_idx + F_idx)
            # Get statistics from matlab analysis
            z = F[subject][0][0][comparison][0][0]['z'][0][0]
            zcrit = F[subject][0][0][comparison][0][0]['zcrit'][0][0]
            sig = F[subject][0][0][comparison][0][0]['sig'][0][0]
            # Pick array of R and F pairs
            z = z[RF_idx, :] 
            z = z[:, RF_idx]
            sig = sig[RF_idx, :] 
            sig = sig[:, RF_idx]
            # Set diagonal elements to 0
            np.fill_diagonal(sig,0)
            # Make ticks label
            group = df_visual['group']
            populations = [group[i] for i in RF_idx]
            # Plot Z score as a heatmap
            g = sns.heatmap(z,  vmin=vmin, vmax=vmax, cmap='bwr', ax=ax[c,s],
            xticklabels=populations, yticklabels=populations)
            g.set_yticklabels(g.get_yticklabels(), rotation = 90)
            ax[c, s].xaxis.tick_top()
            ax[c,0].set_ylabel(f"Z {comparisons[c]}")
            # Plot statistical significant entries
            for y in range(z.shape[0]):
                for x in range(z.shape[1]):
                    if sig[y,x] == 1:
                        ax[c,s].text(x + tau_x, y + tau_y, '*',
                                 horizontalalignment='center', verticalalignment='center',
                                 color='k')
                    else:
                        continue                 
        ax[0,s].set_title(f"{subject}")
    plt.tight_layout()
    print(f"\n Critical Z score is {zcrit}\n")


#%%

plot_pair_zscore(F, cohort, data_path, 
                 vmin = -5, vmax=5, tau_x=0.5, tau_y=0.8)