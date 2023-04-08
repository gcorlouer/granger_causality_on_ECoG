#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:21:10 2022

@author: guime
"""

from src.input_config import args
from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.plotting_lib import full_stim_multi_pfc, full_stim_multi_gfc
from src.plotting_lib import plot_single_trial_pfc, plot_single_trial_gfc, info_flow_stat
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#%%

plt.style.use('ggplot')
fig_width = 25  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'lines.linewidth': 1.5,
          'axes.labelsize': 15,
          'axes.titlesize': 15,
          'font.size': 15,
          'legend.fontsize': 12,
          'xtick.labelsize': 13,
          'ytick.labelsize': 15,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

#%%

cohort = ['AnRa',  'ArLa', 'DiAs']
# Useful paths
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')
fname = 'GC_effect_size.mat'
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
GC = dataset['GC']
# Plot parameters 

vmax = 4
vmin = - vmax

#%% Select anatomical regions

subject = 'DiAs'
reader = EcogReader(data_path, subject=subject)
df_visual = reader.read_channels_info(fname='visual_channels.csv')
dk = df_visual['DK'].unique().tolist()
dk = [dk[i].replace('ctx-lh-', '') for i in range(len(dk))]
df_sorted = df_visual.copy().sort_values(by='peak_time')
indices = parcellation_to_indices(df_visual,  parcellation='DK', matlab=False)


# %% Plot FC

def plot_GC(GC, cohort,
                       vmin = -5, vmax=5):
    conditions = ['Face', 'Place']
    nsub = len(cohort)
    ncomp = len(conditions)
    # xticks
    populations = ['R','O','F']
    connectivity = GC['connectivity'][0][0]
    fig, ax = plt.subplots(ncomp, nsub)
    # Loop over subject and comparison to plot Z score heatmap
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            # Get visual channels
            reader = EcogReader(data_path, subject=subject)
            df_visual = reader.read_channels_info(fname='visual_channels.csv')
            df_sorted = df_visual.copy().sort_values(by='peak_time')
            indices = parcellation_to_indices(df_visual,  parcellation='DK', matlab=False)
            ls = df_sorted.index.tolist()
            sorted_chan = df_sorted['group'].tolist()
            nchan = len(sorted_chan)
            ordered_F = np.zeros((nchan,nchan))
            # Get statistics from matlab analysis
            F = GC[subject][0][0][condition][0][0]['F'][0][0]
            # Relative to rest
            Fb = GC[subject][0][0]['Rest'][0][0]['F'][0][0]
            #F = (F - Fb)/Fb
            F = np.log(F/Fb)
            #maxF = np.amax(F)
            #vmax = round(maxF)+1
            #vmin = -vmax
            # Plot Z score as a heatmap
            if connectivity == 'pairwise':
                # Hierarchical ordering
                for ix, i in enumerate(ls):
                    for jx, j in enumerate(ls):
                        ordered_F[ix,jx] = F[i,j]
                F = ordered_F
                # Make ticks label
                ticks_labels = sorted_chan
            else:
                ticks_labels = populations
                # Get statistics from matlab analysis
            g = sns.heatmap(F,  vmin=vmin, vmax=vmax, cmap='bwr', ax=ax[c,s],
            xticklabels=ticks_labels, yticklabels=ticks_labels)
            g.set_yticklabels(g.get_yticklabels(), rotation = 90)
            ax[c, s].xaxis.tick_top()
            ax[c,0].set_ylabel(f"{condition}")
            ax[0,s].set_title(f"Subject {s}")
    plt.tight_layout()
    
#%%

plot_GC(GC, cohort, vmin = vmin, vmax=vmax)




















