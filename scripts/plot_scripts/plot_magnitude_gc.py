#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:56:42 2022

@author: guime
"""

from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.plotting_lib import info_flow_stat
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#%%

plt.style.use('ggplot')
fig_width = 24  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 10
tick_size = 8
params = {'backend': 'ps',
          'lines.linewidth': 1.5,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'font.size': label_size,
          'legend.fontsize': tick_size,
          'xtick.labelsize': tick_size,
          'ytick.labelsize': tick_size,
          'text.usetex': False,
          'figure.figsize': fig_size,
          "font.weight": "bold",
          "axes.labelweight": "bold"}
plt.rcParams.update(params)

eeg_bands = {"[1 4]":"δ", "[4 7]":"θ", "[ 8 12]": "α", "[13 30]": "β",
             "[32 60]": "γ", "[ 60 120]":"hγ", "[ 0 62]": "hfa"}

eeg_bands = ["[4 7]", "[8 12]", "[13 30]", 
                    "[32 60]", "[60 120]", "[0 62]"] # EEG bands

eeg_bands_fname_dic = {"[1 4]":"delta", "[4 7]":"theta", "[8 12]": "alpha", "[13 30]": "beta",
             "[32 60]": "gamma", "[60 120]":"hgamma", "[0 62]": "hfa"} # to safe fig file name

eeg_bands_fig_title_dic = {"[1 4]":"δ", "[4 7]":"θ", "[8 12]": "α", "[13 30]": "β",
             "[32 60]": "γ", "[60 120]":"hγ", "[0 62]": "hfa"} # To write fig titles
#%%
# Variable inputs of the script
connect = "groupwise" # pairwise or groupwise
cohort = ['AnRa',  'ArLa', 'DiAs']
signal = 'hfa'
connectivity = 'groupwise'
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')

#%%

def plot_gc(F, cohort, vmax = 0.1):
    conditions = ['Rest','Face', 'Place']
    nsub = len(cohort)
    ncomp = len(conditions)
    # xticks
    populations = ['R','O','F']
    connectivity = F['connect'][0][0][0]
    fig, ax = plt.subplots(ncomp, nsub, sharex=False, sharey=False)
    cbar_ax = fig.add_axes([0.91, 0.2, .01, .6])
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
            stat = F[subject][0][0][condition][0][0]
            # Plot Z score as a heatmap
            if connectivity == 'pairwise':
                # Hierarchical ordering
                for ix, i in enumerate(ls):
                    for jx, j in enumerate(ls):
                        ordered_stat[ix,jx] = stat[i,j]
                ordered_stat = ordered_stat[RF_idx,:]
                ordered_stat = ordered_stat[:,RF_idx]
                stat = ordered_stat
                ticks_labels = [sorted_chan[i] for i in RF_idx]
                g = sns.heatmap(stat, vmin=0, vmax=vmax, cmap='YlOrBr', ax=ax[c,s],
                            cbar_ax=cbar_ax, xticklabels=ticks_labels, yticklabels=ticks_labels)
                g.set_yticklabels(g.get_yticklabels(), rotation = 90)
                ax[c,s].set_xticklabels(ticks_labels)
                ax[c,s].set_yticklabels(ticks_labels)
                ax[c,s].xaxis.set_ticks_position('top')
                ax[c,s].xaxis.set_label_position('top')
                ax[c,0].set_ylabel(f"GC {condition}")
                ax[0,s].set_title(f"Subject {s}")
                # Make ticks label
            else:
                ticks_labels = populations
                # Get statistics from matlab analysis
                g = sns.heatmap(stat, vmin=0, vmax=vmax, cmap='YlOrBr', ax=ax[c,s],
                            cbar_ax=cbar_ax)
                g.set_yticklabels(g.get_yticklabels(), rotation = 90)
                ax[c,s].set_xticklabels(ticks_labels)
                ax[c,s].set_yticklabels(ticks_labels)
                ax[c,s].xaxis.set_ticks_position('top')
                ax[c,s].xaxis.set_label_position('top')
                ax[c,0].set_ylabel(f"GC {condition}")
                fband = eeg_bands_fig_title_dic[band]
                ax[0,s].set_title(f"S{s}, {fband}")
    plt.suptitle(f"Single subject {connectivity} conditional GC")
    
#%%
vmax = 0.01
for band in eeg_bands:
    # Useful paths
    bandstr = eeg_bands_fname_dic[band]
    fname = "_".join(['magnitude', connect, band, 'GC.mat'])
    path = result_path
    fpath = path.joinpath(fname)
    # Read dataset
    dataset = loadmat(fpath)
    F = dataset['GC']
    connectivity = F['connect'][0][0][0]
    plot_gc(F, cohort, vmax=vmax)
    figpath = Path('~','thesis','overleaf_project', 'figures','results_figures').expanduser()
    bandstr = eeg_bands_fname_dic[band]
    fname =  "_".join(["magnitude", connectivity, bandstr,"GC.pdf"])
    figpath = figpath.joinpath(fname)
    plt.savefig(figpath)