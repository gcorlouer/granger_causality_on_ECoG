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
comparisons = ['Rest/Face', 'Rest/Place', 'Face/Place']
cohort = args.cohort
#%% 
# Read Z scores for one subjects and plot them
nsub = len(cohort)
ncomp = len(comparisons)
fig, ax = plt.subplots(nsub, ncomp)
for s, subject in enumerate(cohort):
    for c, comparison in enumerate(comparisons):
        z = gc[s,c]['z']
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

plt.tight_layout()

#%%








