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
fig_width = 20  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = 0.9*fig_width*golden_mean      # height in inches
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
# Plot parameters
function = 'MI'
vmax = 0.01
vmin = 0
# %% Plot FC

def plot_functional_connectivity(F, cohort, function='GC', vmax=5, vmin=0, 
                                 tau_x=0.5, tau_y=0.8):
    conditions = ['Rest','Face', 'Place']
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
            df_sorted = df_visual.copy().sort_values(by='latency')
            ls = df_sorted.index.tolist()
            df_sorted = df_sorted.reset_index()
            sorted_chan = df_sorted['group'].tolist()
            # Find retinotopic, others and face channels indices 
            R_idx = df_sorted.index[df_sorted['group']=='R'].tolist()
            F_idx = df_sorted.index[df_sorted['group']=='F'].tolist()
            RF_idx = np.array(R_idx + F_idx)
            nchan = len(sorted_chan)
            ordered_stat = np.zeros((nchan,nchan))
            ordered_sig = np.zeros((nchan,nchan))
            # Get statistics from matlab analysis
            stat = F[subject][0][0][condition][0][0][function][0][0]['F'][0][0]
            sig = F[subject][0][0][condition][0][0][function][0][0]['sig'][0][0]
            # Relative to rest
            #statb = F[subject][0][0]['Rest'][0][0][function][0][0]['F'][0][0]
            #stat = (stat - statb)
            #stat = (stat - statb)/statb
            #stat = np.log(stat/statb)
            #max_stat = np.amax(stat)
            #vmax = round(max_stat)
            # Plot Z score as a heatmap
            if connectivity == 'pairwise':
                # Hierarchical ordering
                for ix, i in enumerate(ls):
                    for jx, j in enumerate(ls):
                        ordered_stat[ix,jx] = stat[i,j]
                        ordered_sig[ix,jx] = sig[i,j]
                ordered_stat = ordered_stat[RF_idx,:]
                ordered_stat = ordered_stat[:,RF_idx]
                stat = ordered_stat
                sig = ordered_sig
                ticks_labels = [sorted_chan[i] for i in RF_idx]
                # Make ticks label
            else:
                ticks_labels = populations
                # Get statistics from matlab analysis
                np.fill_diagonal(sig,0)
            #cbar_kws={"ticks":[0,2.5*1e-3,5*1e-3]}
            g = sns.heatmap(stat,  vmin=0, vmax=vmax, cmap='YlOrBr', ax=ax[c,s],
            xticklabels=ticks_labels, yticklabels=ticks_labels)
            g.set_yticklabels(g.get_yticklabels(), rotation = 90)
            ax[c, s].xaxis.tick_top()
            ax[c,0].set_ylabel(f"{function } {condition}")
            ax[0,s].set_title(f"S{s}")
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

plot_functional_connectivity(F, cohort, function=function, vmax=vmax, vmin=vmin,
                             tau_x=0.5, tau_y=0.8)




















