#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:25:28 2022
In this script we plot and compare single trial mvgc between Face and Place
conditions
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

conditions = ['Rest', 'Face', 'Place',]
cohort = args.cohort
path = args.result_path
fname = 'compare_condition_fc.mat'
fpath = path.joinpath(fname)
gc = loadmat(fpath)
gc = gc['dataset']

#%% Get z score for mutual information and GC

zI = gc['zI'][0][0]
zF = gc['zF'][0][0]

# Plot Z score in each direction

df_visual = reader.read_channels_info(fname='visual_channels.csv')
# Find retinotopic and face channels indices 
populations = parcellation_to_indices(df_visual, parcellation='group', matlab=False)
populations = list(populations.keys())
R_idx = populations.index('R')
O_idx = populations.index('O')
F_idx = populations.index('F')
sort_idx = [R_idx, O_idx, F_idx]

# Build xticks label
ticks_label = [populations[i] for i in sort_idx]            
# Plot F as heatmap
g = sns.heatmap(f, xticklabels=ticks_label, vmin=vmin, vmax=vmax,
                yticklabels=ticks_label, cmap='bwr', ax=ax[c,s])
g.set_yticklabels(g.get_yticklabels(), rotation = 90)
# Position xticks on top of heatmap
ax[c,s].xaxis.tick_top()
ax[c,0].set_ylabel(condition)

# Position xticks on top of heatmap
ax[c, 1].xaxis.tick_top()
ax[0,s].set_title(f'{F} subject {s}')