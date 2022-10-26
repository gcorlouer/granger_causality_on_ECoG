#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:20:03 2022
In this script we plot GC analysis on full trial duration. This include
multitrial GC and single trial GC
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
fname = 'null_fc.mat'
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
F = dataset['FC']


# %% Plot FC

def plot_functional_connectivity(F, cohort, function='GC',
                       vmin = -5, vmax=5, tau_x=0.5, tau_y=0.8):
    conditions = ['Face', 'Place']
    nsub = len(cohort)
    ncomp = len(conditions)
    # xticks
    populations = ['R','O','F']
    connectivity = F['connectivity'][0][0]
    fig, ax = plt.subplots(ncomp, nsub)
    # Loop over subject and comparison to plot Z score heatmap
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            # Get visual channels
            reader = EcogReader(data_path, subject=subject)
            df_visual = reader.read_channels_info(fname='visual_channels.csv')
            df_sorted = df_visual.copy().sort_values(by='peak_time')
            l = df_visual.index.tolist()
            ls = df_sorted.index.tolist()
            sorted_chan = df_sorted['group'].tolist()
            nchan = len(sorted_chan)
            ordered_stat = np.zeros((nchan,nchan))
            ordered_sig = np.zeros((nchan,nchan))
            # Get statistics from matlab analysis
            stat = F[subject][0][0][condition][0][0][function][0][0]['F'][0][0]
            sig = F[subject][0][0][condition][0][0][function][0][0]['sig'][0][0]
            # Relative to rest
            statb = F[subject][0][0]['Rest'][0][0][function][0][0]['F'][0][0]
            stat = (stat - statb)/statb 
            # Plot Z score as a heatmap
            if connectivity == 'pairwise':
                # Hierarchical ordering
                for ix, i in enumerate(ls):
                    for jx, j in enumerate(ls):
                        ordered_stat[ix,jx] = stat[i,j]
                        ordered_sig[ix,jx] = sig[i,j]
                stat = ordered_stat
                sig = ordered_sig
                # Make ticks label
                ticks_labels = sorted_chan
            else:
                ticks_labels = populations
                # Get statistics from matlab analysis
                np.fill_diagonal(sig,0)
            g = sns.heatmap(stat,  vmin=vmin, vmax=vmax, cmap='bwr', ax=ax[c,s],
            xticklabels=ticks_labels, yticklabels=ticks_labels)
            g.set_yticklabels(g.get_yticklabels(), rotation = 90)
            ax[c, s].xaxis.tick_top()
            ax[c,0].set_ylabel(f"{function } {condition}")
            ax[0,s].set_title(f"Subject {s}")
            # Plot statistical significant entries
            for y in range(stat.shape[0]):
                for x in range(stat.shape[1]):
                    if sig[y,x] == 1:
                        ax[c,s].text(x + tau_x, y + tau_y, '*',
                                 horizontalalignment='center', verticalalignment='center',
                                 color='k')
                    else:
                        continue                 
    plt.tight_layout()
    
#%%

plot_functional_connectivity(F, cohort, function='GC',
                       vmin = -5, vmax=5, tau_x=0.5, tau_y=0.8)

#%% Plot multitrial pair FC
# Load functional connectivity matrix
# fc_path = result_path.joinpath(fname)
# fc = loadmat(fc_path)
# fc = fc['dataset']
# vmax = 3
# #vmax = [11, 15, 12]
# (ncdt, nsub) = fc.shape

# full_stim_multi_pfc(fc, cohort, args, F='pGC',vmin=-vmax,vmax=vmax,
#                                  rotation=90, tau_x=0.5, tau_y=0.8)


# #%% Plot multitrial pair MI

# full_stim_multi_pfc(fc, cohort, args, F='pMI', vmin=-vmax,vmax=vmax,
#                                  rotation=90, tau_x=0.5, tau_y=0.8)

# #%% Plot multitrial groupwise GC
# vmin = 3
# full_stim_multi_gfc(fc, cohort, args, F='gGC', vmin=vmin,vmax=-vmin,
#                                  rotation=90, tau_x=0.5, tau_y=0.8)

# #%% Plot  multitrial groupwise MI
# vmin = 4
# full_stim_multi_gfc(fc, cohort, args, F='gMI', vmin=vmin,vmax=-vmin,  
#                                  rotation=90, tau_x=0.5, tau_y=0.8)




















