#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:02:26 2022
In this script we plot stochastic dominance of GC/MI between conditions
in each subjects, directions and accross subjects.
@author: guime
"""


import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path

#%% Read single FC data

fname = 'test_singleFC.mat'
path = args.result_path
fpath = path.joinpath(fname)
dataset = loadmat(fpath)
# Read nsubjects x ncomparisons Z scores from single trial GC
per_subject_data = dataset['Subject']
cross_subject_data = dataset['CrossSubject']
# dataset Is of the form dataset[sub_id][0][0][condition][0][0][FC][0][0]

#%% Plotting functions

def plot_single_zscore(per_subject_data, args, FC = 'single_GC',
                       vmin = -3, vmax=3, tau_x=0.5, tau_y=0.8):
    """
    We plot Z score of test comparing single FC distrib in condition 1 with
    single FC in condition 2.
    """
    comparisons = ['FaceRest', 'PlaceRest', 'FacePlace']
    cohort = args.cohort
    nsub = len(cohort)
    ncomp = len(comparisons)
    fig, ax = plt.subplots(nsub, ncomp)
    for s, subject in enumerate(cohort):
        for c, comparison in enumerate(comparisons):
            # Get statistics from matlab analysis
            z = per_subject_data[subject][0][0][comparison][0][0][FC][0][0]['zstat'][0][0]['z'][0][0]
            zcrit = per_subject_data[subject][0][0][comparison][0][0][FC][0][0]['zstat'][0][0]['zcrit'][0][0]
            sig = per_subject_data[subject][0][0][comparison][0][0][FC][0][0]['zstat'][0][0]['sig'][0][0]
            ticks = ['R','O','F']
            # Plot Z score as a heatmap
            g = sns.heatmap(z,  vmin=vmin, vmax=vmax, cmap='bwr', ax=ax[c,s],
            xticklabels=ticks, yticklabels=ticks)
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
    print(f"Critical Z score is {zcrit}")
    
def plot_group_zscore(cross_subject_data, FC = 'single_GC',
                       vmin = -3, vmax=3, tau_x=0.5, tau_y=0.8):
    comparisons = ['FaceRest', 'PlaceRest', 'FacePlace']
    ncomp = len(comparisons)
    fig, ax = plt.subplots(ncomp)
    for c, comparison in enumerate(comparisons):
        # Get statistics from matlab analysis
        z = cross_subject_data[comparison][0][0][FC][0][0]['zstat'][0][0]['z'][0][0]
        zcrit = cross_subject_data[comparison][0][0][FC][0][0]['zstat'][0][0]['zcrit'][0][0]
        sig = cross_subject_data[comparison][0][0][FC][0][0]['zstat'][0][0]['sig'][0][0]
        ticks = ['R','O','F']
        # Plot Z score as a heatmap
        g = sns.heatmap(z,  vmin=vmin, vmax=vmax, cmap='bwr', ax=ax[c],
        xticklabels=ticks, yticklabels=ticks)
        g.set_yticklabels(g.get_yticklabels(), rotation = 90)
        ax[c,].xaxis.tick_top()
        ax[c].set_ylabel(f"Z {comparisons[c]}")
        # Plot statistical significant entries
        for y in range(z.shape[0]):
            for x in range(z.shape[1]):
                if sig[y,x] == 1:
                    ax[c].text(x + tau_x, y + tau_y, '*',
                             horizontalalignment='center', verticalalignment='center',
                             color='k')
                else:
                    continue                 
        ax[0].set_title(f"Group Z score from {FC} distribution")
    plt.tight_layout()
    print(f"Critical Z score is {zcrit}")
#%% Plot Z score per subjects
    
plot_single_zscore(per_subject_data, args, FC = 'single_GC',
                       vmin = -3, vmax=3, tau_x=0.5, tau_y=0.8)

#%% Plot group Z scores

plot_group_zscore(cross_subject_data, FC = 'single_GC',
                       vmin = -3, vmax=3, tau_x=0.5, tau_y=0.8)
 
