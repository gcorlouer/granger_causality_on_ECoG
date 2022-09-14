#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:04:30 2022

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
fname = 'compare_condition_gGC.mat'
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
F = dataset['gGC']

#%% Plotting function

def plot_group_zscore(F, cohort,
                       vmin = -5, vmax=5, tau_x=0.5, tau_y=0.8):
    """
    We plot Z score from comparing permutation group F in condition 1 with
    condition 2.
    """
    comparisons = ['FvsR', 'PvsR', 'FvsP']
    nsub = len(cohort)
    ncomp = len(comparisons)
    # xticks
    populations = ['R','O','F']
    fig, ax = plt.subplots(nsub, ncomp)
    # Loop over subject and comparison to plot Z score heatmap
    for s, subject in enumerate(cohort):
        for c, comparison in enumerate(comparisons):
            # Get statistics from matlab analysis
            z = F[subject][0][0][comparison][0][0]['z'][0][0]
            zcrit = F[subject][0][0][comparison][0][0]['zcrit'][0][0]
            sig = F[subject][0][0][comparison][0][0]['sig'][0][0]
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

#%% Plot group MI between conditions

plot_group_zscore(F, cohort, 
                 vmin = -5, vmax=5, tau_x=0.5, tau_y=0.8)