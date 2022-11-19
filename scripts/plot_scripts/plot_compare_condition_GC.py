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
fig_width = 16  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 10
params = {'backend': 'ps',
          'lines.linewidth': 1.5,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'font.size': label_size,
          'legend.fontsize': label_size,
          'xtick.labelsize': label_size,
          'ytick.labelsize': label_size,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)


#eeg_bands = {"delta":[1, 4], "theta":[4, 7], "alpha": [8, 12], "beta": [13,30],
#             "gamma": [32, 60], "hgamma":[60, 120], "hfa": [0, 62]}

eeg_bands = {"[1 4]":"δ", "[4 7]":"θ", "[ 8 12]": "α", "[13 30]": "β",
             "[32 60]": "γ", "[ 60 120]":"hγ", "[ 0 62]": "hfa"}

#%%

cohort = ['AnRa',  'ArLa', 'DiAs']
# Useful paths
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')
connect = "pairwise"
band = "[0 62]"
fname = "compare_condition_GC_" + connect + '_' + band + "Hz.mat"
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
    bandstr = np.array2string(band[0])
    fig, ax = plt.subplots(nsub, ncomp)
    cbar_ax = fig.add_axes([0.91, 0.2, .01, .6])
    # Loop over subject and comparison to plot Z score heatmap
    for s, subject in enumerate(cohort):
        for c, comparison in enumerate(comparisons):
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
            ordered_z = np.zeros((nchan,nchan))
            ordered_sig = np.zeros((nchan,nchan))
            # Get statistics from matlab analysis
            z = F[subject][0][0][comparison][0][0]['z'][0][0]
            zcrit = F[subject][0][0][comparison][0][0]['zcrit'][0][0]
            sig = F[subject][0][0][comparison][0][0]['sig'][0][0]
            np.fill_diagonal(sig, 0)
            # Plot Z score as a heatmap
            if connectivity == 'pairwise':
                # Hierarchical ordering
                for ix, i in enumerate(ls):
                    for jx, j in enumerate(ls):
                        ordered_z[ix,jx] = z[i,j]
                        ordered_sig[ix,jx] = sig[i,j]
                ordered_z = ordered_z[RF_idx,:]
                ordered_z = ordered_z[:,RF_idx]
                ordered_sig = ordered_sig[RF_idx,:]
                ordered_sig = ordered_sig[:,RF_idx]
                z = ordered_z
                sig = ordered_sig
                # Plot z score
                ticks_labels = [sorted_chan[i] for i in RF_idx]
                g = sns.heatmap(z, vmin=vmin, vmax=vmax, cmap='bwr', ax=ax[c,s],
                            cbar_ax=cbar_ax, cbar_kws={"ticks":[-1,-0.5,0,0.5,1]},
                            xticklabels=ticks_labels, yticklabels=ticks_labels)
                g.set_yticklabels(g.get_yticklabels(), rotation = 90)
                if c>=1:
                    ax[c,s].set_xticks([]) # (turn off xticks)
                if s>=1:
                    ax[c,s].set_yticks([]) # (turn off xticks)
                ax[c,s].xaxis.set_ticks_position('top')
                ax[c,s].xaxis.set_label_position('top')
                ax[c,0].set_ylabel(f"{comparisons[c]}")
                # Plot statistical significant entries
                for y in range(z.shape[0]):
                    for x in range(z.shape[1]):
                        if sig[y,x] == 1:
                            ax[c,s].text(x + tau_x, y + tau_y, '*',
                                     horizontalalignment='center', verticalalignment='center',
                                     color='k')
                        else:
                            continue                 
            else:
                ticks_labels = populations
                # Get statistics from matlab analysis
                np.fill_diagonal(sig,0)
                g = sns.heatmap(z,  vmin=vmin, vmax=vmax, cmap='bwr', ax=ax[c,s],
                                cbar_ax=cbar_ax, cbar_kws={"ticks":[-1,-0.5,0,0.5,1]})
                g.set_yticklabels(g.get_yticklabels(), rotation = 90)
                ax[0,s].set_xticklabels(ticks_labels)
                ax[c,0].set_yticklabels(ticks_labels)
                ax[0,s].xaxis.set_ticks_position('top')
                ax[0,s].xaxis.set_label_position('top')
                g.set_yticklabels(g.get_yticklabels(), rotation = 90)
                ax[c,0].set_ylabel(f"{comparisons[c]}")
                if c>=1:
                        ax[c,s].set_xticks([]) # (turn off xticks)
                if s>=1:
                        ax[c,s].set_yticks([]) # (turn off xticks)
                # Plot statistical significant entries
                for y in range(z.shape[0]):
                    for x in range(z.shape[1]):
                        if sig[y,x] == 1:
                            ax[c,s].text(x + tau_x, y + tau_y, '*',
                                     horizontalalignment='center', verticalalignment='center',
                                     color='k')
                        else:
                            continue                 
            fband = eeg_bands[bandstr]
            ax[0,s].set_title(f"S{s}, {fband}")
    print(f"\n Critical Z score is {zcrit}\n")

#%% Plot GC comparison between conditions

plot_compare_condition_GC(F, cohort, 
                 vmin = vmin, vmax=vmax, tau_x=0.5, tau_y=0.8)