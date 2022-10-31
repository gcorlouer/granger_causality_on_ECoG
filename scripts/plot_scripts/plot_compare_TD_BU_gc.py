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
fig_width = 20  # figure width in cm
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

# Bands

eeg_bands = {"delta":[1, 4], "theta":[4, 7], "alpha": [8, 12], "beta": [12,30],
             "gamma": [32, 60], "hgamma":[60, 120]}
# bands.delta = [1 4];

#%%
cohort = ['AnRa',  'ArLa', 'DiAs']
# Useful paths
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')
connect = "groupwise"
band = "[0 62]"
fname = "compare_TD_BU_GC_"+ band + "Hz.mat"
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
F = dataset['GC']

#%%

def plot_TD_BU_zscore(F, cohort, data_path,
                       vmin = -5, vmax=5, tau_x=0.5, tau_y=0.8):
    """
    We plot Z score from comparing top down with bottom up pairwise unconditional
    GC in a given condition
    """
    conditions = ['Rest', 'Face', 'Place']
    nsub = len(cohort)
    ncomp = len(conditions)
    band = F['band'][0][0]
    fig, ax = plt.subplots(nsub, ncomp)
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
            g = sns.heatmap(z,  vmin=vmin, vmax=vmax, cmap='bwr', ax=ax[c,s],
            xticklabels=xticklabels, yticklabels=yticklabels)
            g.set_yticklabels(g.get_yticklabels(), rotation = 90)
            ax[c, s].xaxis.tick_top()
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
        ax[0,s].set_title(f"S{s}, [{band[0][0]} {band[0][1]}]Hz")
    plt.tight_layout()
    print(f"\n Critical Z score is {zcrit}\n")

#%%

plot_TD_BU_zscore(F, cohort, data_path, 
                 vmin = -5, vmax=5, tau_x=0.5, tau_y=0.8)