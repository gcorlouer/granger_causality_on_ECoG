#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 11:26:38 2022

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
fig_width = 28  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 14
tick_size = 12
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

#eeg_bands = {"delta":[1, 4], "theta":[4, 7], "alpha": [8, 12], "beta": [12,30],
#            "gamma": [32, 60], "hgamma":[60, 120]}

eeg_bands = {"[1 4]":"δ", "[4 7]":"θ", "[ 8 12]": "α", "[13 30]": "β",
             "[32 60]": "γ", "[ 60 120]":"hγ", "[ 0 62]": "hfa"}

eeg_bands = ["[4 7]", "[8 12]", "[13 30]", 
                    "[32 60]", "[60 120]", "[0 62]"] # EEG bands

eeg_bands_fname_dic = {"[1 4]":"delta", "[4 7]":"theta", "[8 12]": "alpha", "[13 30]": "beta",
             "[32 60]": "gamma", "[60 120]":"hgamma", "[0 62]": "hfa"} # to safe fig file name

eeg_bands_fig_title_dic = {"[1 4]":"δ", "[4 7]":"θ", "[ 8 12]": "α", "[13 30]": "β",
             "[32 60]": "γ", "[ 60 120]":"hγ", "[ 0 62]": "hfa"} # To write fig titles

#%%

def plot_TD_BU_zscore(F, cohort, data_path, cmap = 'PuOr',
                       vmin = -5, vmax=5, tau_x=0.5, tau_y=0.8):
    """
    We plot Z score from comparing top down with bottom up pairwise unconditional
    GC in a given condition
    """
    conditions = ['Rest', 'Face', 'Place']
    nsub = len(cohort)
    ncomp = len(conditions)
    band = F['band'][0][0]
    bandstr = np.array2string(band[0])
    fig, ax = plt.subplots(nsub, ncomp)
    cbar_ax = fig.add_axes([0.91, 0.2, .01, .6])
    # Loop over subject and comparison to plot Z score heatmap
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            # Get statistics from matlab analysis
            z = F[subject][0][0][condition][0][0]['z'][0][0]
            zcrit = F[subject][0][0][condition][0][0]['zcrit'][0][0]
            sig = F[subject][0][0][condition][0][0]['sig'][0][0]
            # Make ticks label
            (nR,nF) = z.shape
            xticklabels = ['F']*nF
            yticklabels = ['R']*nR
            # Plot Z score as a heatmap
            g = sns.heatmap(z,  vmin=vmin, vmax=vmax, cmap=cmap, ax=ax[c,s],
            xticklabels=xticklabels, yticklabels=yticklabels, cbar_ax=cbar_ax)
            g.set_yticklabels(g.get_yticklabels(), rotation = 90)
            # if c>=1:
            #         ax[c,s].set_xticks([]) # (turn off xticks)
            # if s>=1:
            #         ax[c,s].set_yticks([]) # (turn off xticks)
            ax[c,s].xaxis.set_ticks_position('top')
            ax[c,s].xaxis.set_label_position('top')
            ax[c,0].set_ylabel(f"Z {condition}")
            # Plot statistical significant entries
            for y in range(z.shape[0]):
                for x in range(z.shape[1]):
                    if sig[y,x] == 1:
                        ax[c,s].text(x + tau_x, y + tau_y, '*',
                                 horizontalalignment='center', verticalalignment='center',
                                 color='k')
                    else:
                        continue                 
            fband = eeg_bands_fig_title_dic[bandstr]
            if fband == "hfa":
                ax[0,s].set_title(f"Subject {s}, HFA",fontweight="bold")
            else:
                ax[0,s].set_title(f"Subject {s}, {fband}-band",fontweight="bold")
    #plt.suptitle(f"Top-down relative to bottom-up GC")
    print(f"\n Critical Z score is {zcrit}\n")

#%% Plot results

cohort = ['AnRa',  'ArLa', 'DiAs']
cmap ='PuOr_r'
vmax = 10

for band in eeg_bands:
    # Useful paths
    cifar_path = Path('~','projects','cifar').expanduser()
    data_path = cifar_path.joinpath('data')
    result_path = cifar_path.joinpath('results')
    fname = "compare_TD_BU_GC_"+ band + "Hz.mat"
    path = result_path
    fpath = path.joinpath(fname)
    # Read dataset
    dataset = loadmat(fpath)
    F = dataset['GC']
    
    vmin = -vmax
    
    plot_TD_BU_zscore(F, cohort, data_path,  cmap=cmap,
                     vmax=vmax, vmin=vmin, tau_x=0.5, tau_y=0.8)
    figpath = Path('~','thesis','overleaf_project', 'figures','results_figures').expanduser()
    bandstr = eeg_bands_fname_dic[band]
    fname =  "_".join(["compare", "td_vs_bu","condition", bandstr,"GC.pdf"])
    figpath = figpath.joinpath(fname)
    #plt.savefig(figpath)