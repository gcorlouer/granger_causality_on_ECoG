#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script plot functional connectivity i.e. Mutual information and pairwise
conditional Granger causality.
@author: guime
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.preprocessing_lib import EcogReader, build_dfc
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path


#%% Read ROI and functional connectivity data

reader = EcogReader(args.data_path)
# Read visual channels 
df_visual = reader.read_channels_info(fname='visual_channels.csv')

# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']

# Load functional connectivity matrix
result_path = Path('../results')

fname = 'pairwise_fc.mat'
fc_path = result_path.joinpath(fname)

fc = loadmat(fc_path)
fc = fc['dataset']

#%% Build dataset fc dictionary

dfc = build_dfc(fc)

#%% Plot pairwise conditional gc
#%matplotlib qt
subject = 'DiAs'
conditions = ['baseline', 'Face', 'Place']
ncdt = len(conditions)
condition = 'Face'
subjects = list(dfc['subject'].unique())
nsub = len(subjects)
f, ax = plt.subplots(nsub,2)
for i in range(nsub):
    # Subject
    subject = subjects[i]
    reader = EcogReader(args.data_path, subject= subject)
    # Read visual channels 
    df_visual = reader.read_channels_info(fname='visual_channels.csv')
    # Drop other channels
    df_visual = df_visual[df_visual.group != 'O']
    # Rank channels hierachically
    df_visual = df_visual.sort_values(by='Y')
    ranked_idx = df_visual.index.tolist()
    # Get R, F, P channels
    populations = df_visual['group']
    # Gc baseline
    gcb = dfc['gc'].loc[(dfc['subject']==subject) & (dfc['condition']=='baseline')]
    gcb = gcb.iloc[0]
    # MI baseline
    mib = dfc['mi'].loc[(dfc['subject']==subject) & (dfc['condition']=='baseline')]
    mib = mib.iloc[0]
    gc = dfc['gc'].loc[(dfc['subject']==subject) & (dfc['condition']==condition)]
    gc = gc.iloc[0]
    # Rescale by baseline
    gc = np.log(gc/gcb)
    # Permute gc indices
    gc = gc[ranked_idx, :]
    gc = gc[:, ranked_idx]
    # Remove Other channels indices
    mi = dfc['mi'].loc[(dfc['subject']==subject) & (dfc['condition']==condition)]
    mi = mi.iloc[0]
    mi = np.log(mi/mib)
    # Permute MI indices
    mi = mi[ranked_idx, :]
    mi = mi[:, ranked_idx]
    # Plot MI and GC
    sns.heatmap(gc, xticklabels=populations,
                            yticklabels=populations, cmap='YlOrBr', ax=ax[i,1])
    sns.heatmap(mi, xticklabels=populations,
                            yticklabels=populations, cmap='YlOrBr', ax=ax[i,0])
    ax[0,0].set_title(f'MI (bit) {condition}')
    ax[0,1].set_title(f'Transfer entropy (bit/s) {condition}')
    ax[i, 0].set_ylabel(subject)
    

