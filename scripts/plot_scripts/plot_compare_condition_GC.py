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
connect = "groupwise"
band = "[0 62.5]"
fname = "compare_condition_GC_" + connect + '_' band + "Hz.mat"
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
F = dataset['GC']

vmin = -1
vmax = 1

#%% Plotting function

def plot_compare_condition_GC(F, cohort,
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
    connectivity = F['connectivity'][0][0]
    band = F['band'][0][0]
    fig, ax = plt.subplots(nsub, ncomp)
    # Loop over subject and comparison to plot Z score heatmap
    for s, subject in enumerate(cohort):
        for c, comparison in enumerate(comparisons):
            # Get visual channels
            reader = EcogReader(data_path, subject=subject)
            df_visual = reader.read_channels_info(fname='visual_channels.csv')
            # Order channel indices along hierarchy
            R_idx = df_visual.index[df_visual['group']=='R'].tolist()
            F_idx = df_visual.index[df_visual['group']=='F'].tolist()
            O_idx = df_visual.index[df_visual['group']=='O'].tolist()
            vis_idx = np.array(R_idx + O_idx + F_idx)
            # Get statistics from matlab analysis
            z = F[subject][0][0][comparison][0][0]['z'][0][0]
            zcrit = F[subject][0][0][comparison][0][0]['zcrit'][0][0]
            sig = F[subject][0][0][comparison][0][0]['sig'][0][0]
            # Plot Z score as a heatmap
            if connectivity == 'pairwise':
                ticks_labels = vis_idx
            else:
                ticks_labels = populations
                # Get statistics from matlab analysis
                np.fill_diagonal(sig,0)
            g = sns.heatmap(z,  vmin=vmin, vmax=vmax, cmap='bwr', ax=ax[c,s],
            xticklabels=ticks_labels, yticklabels=ticks_labels)
            g.set_yticklabels(g.get_yticklabels(), rotation = 90)
            ax[c, s].xaxis.tick_top()
            ax[c,0].set_ylabel(f"Z; {comparisons[c]}")
            # Plot statistical significant entries
            for y in range(z.shape[0]):
                for x in range(z.shape[1]):
                    if sig[y,x] == 1:
                        ax[c,s].text(x + tau_x, y + tau_y, '*',
                                 horizontalalignment='center', verticalalignment='center',
                                 color='k')
                    else:
                        continue                 
        ax[0,s].set_title(f"S{s}, [{band[0][0]} {band[0][1]}]Hz")
    plt.tight_layout()
    print(f"\n Critical Z score is {zcrit}\n")

#%% Plot GC comparison between conditions

plot_compare_condition_GC(F, cohort, 
                 vmin = vmin, vmax=vmax, tau_x=0.5, tau_y=0.8)