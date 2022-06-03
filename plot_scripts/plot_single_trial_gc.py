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

#%% Read dataset

fname = 'compare_condition_fc.mat'
path = args.result_path
fpath = path.joinpath(fname)
gc = loadmat(fpath)
# Read nsubjects x ncomparisons Z scores from single trial GC
gc = gc['GC']

#%% Read single FC data

fname = 'test_singleFC.mat'
path = args.result_path
fpath = path.joinpath(fname)
dataset = loadmat(fpath)
# Read nsubjects x ncomparisons Z scores from single trial GC
dataset = dataset['Subject']
# dataset Is of the form dataset[sub_id][0][0][condition][0][0][FC][0][0]
comparisons_ticks = ['Face/Rest', 'Place/Rest', 'Face/Place']

#%%

cohort = args.cohort
comparisons = ['FaceRest', 'PlaceRest', 'FacePlace']
FC = 'single_GC'
nsub = 3
ncomp =3
fig, ax = plt.subplots(nsub, ncomp)
for s, subject in enumerate(cohort):
    for c, comparison in enumerate(comparisons):
        z = dataset[subject][0][0][comparison][0][0][FC][0][0]['z'][0][0]
        zcrit = dataset[subject][0][0][comparison][0][0][FC][0][0]['zcrit'][0][0]
        sig = np.where(z>zcrit,1,0)
        # Plot Z score as a heatmap
        g = sns.heatmap(z,  vmin=-3, vmax=3, cmap='bwr', ax=ax[c,s])
    
#%% 

def plot_zscore(gc, args, z_connect='zF', tau_x=0.5, tau_y= 0.8, 
                vmin = -3, vmax = 3, cmap = 'bwr'):
    cohort = args.cohort
    comparisons = ['Rest/Face', 'Rest/Place', 'Face/Place']
    for s, subject in enumerate(cohort):
        for c, comparison in enumerate(comparisons):
            z = gc[s,c][z_connect]
            if z_connect == 'zF':
                zcrit = gc[s,c]['zFcrit'][0][0]
            else:
                zcrit = gc[s,c]['zIcrit'][0][0]
            # compute significance matrix
            sig = np.where(z>zcrit,1,0)
            populations = gc[s,c]['populations']
            npop = populations.size
            populations = [populations[i][0][0] for i in range(npop)]
            # Plot Z score as a heatmap
            g = sns.heatmap(z, xticklabels=populations, vmin=vmin, vmax=vmax,
                            yticklabels=populations, cmap=cmap, ax=ax[c,s])
            g.set_yticklabels(g.get_yticklabels(), rotation = 90)
            # Position xticks on top of heatmap
            ax[c,s].xaxis.tick_top()
            ax[c,0].set_ylabel(comparison)
            # Position xticks on top of heatmap
            ax[c, 1].xaxis.tick_top()
            ax[0,s].set_title(f'Z score subject {s}')
            # Plot statistical significant entries
            for y in range(npop):
                for x in range(npop):
                    if sig[y,x] == 1:
                        ax[c,s].text(x + tau_x, y + tau_y, '*',
                                 horizontalalignment='center', verticalalignment='center',
                                 color='k')
                    else:
                        continue                 
            plt.tight_layout()


    
#%% 
# Read Z scores for one subjects and plot them
nsub = len(cohort)
ncomp = len(comparisons)
tau_x, tau_y =(0.5, 0.8)
fig, ax = plt.subplots(nsub, ncomp)
for s, subject in enumerate(cohort):
    for c, comparison in enumerate(comparisons):
        z = gc[s,c]['z']
        zcrit = gc[s,c]['zcrit'][0][0]
        # compute significance matrix
        sig = np.where(z>zcrit,1,0)
        populations = gc[s,c]['populations']
        npop = populations.size
        populations = [populations[i][0][0] for i in range(npop)]
        # Plot Z score as a heatmap
        g = sns.heatmap(z, xticklabels=populations, vmin=-3, vmax=3,
                        yticklabels=populations, cmap='bwr', ax=ax[c,s])
        g.set_yticklabels(g.get_yticklabels(), rotation = 90)
        # Position xticks on top of heatmap
        ax[c,s].xaxis.tick_top()
        ax[c,0].set_ylabel(comparison)
        # Position xticks on top of heatmap
        ax[c, 1].xaxis.tick_top()
        ax[0,s].set_title(f'Z score subject {s}')
        # Plot statistical significant entries
        for y in range(npop):
            for x in range(npop):
                if sig[y,x] == 1:
                    ax[c,s].text(x + tau_x, y + tau_y, '*',
                             horizontalalignment='center', verticalalignment='center',
                             color='k')
                else:
                    continue                 
plt.tight_layout()

#%%








