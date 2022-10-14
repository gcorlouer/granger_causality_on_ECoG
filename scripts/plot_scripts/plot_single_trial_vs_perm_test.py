#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:43:38 2022
Plot histogram of single trial distrib vs permutation testing
@author: guime
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from pathlib import Path
from src.input_config import args


#%% Style parameters

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

#%%
# Paths
result_path = Path('~','projects','cifar', 'results').expanduser()
figpath = Path('~','thesis','overleaf_project', 'figures','method_figure').expanduser()
fname = 'permtest_vs_single_trial.mat'
fpath = Path.joinpath(result_path, fname)
dataset = loadmat(fpath)
stat = dataset['stat']

# Input
nbins = 50
#%% Get statistics

perm_test = stat['permutation'][0][0]['t'][0][0]
perm_z = stat['permutation'][0][0]['z'][0][0]
perm_z = perm_z[0][0]
perm_z = round(perm_z, 2)
obs = stat['permutation'][0][0]['obs'][0][0]
single_F = stat['single_trial'][0][0]['t'][0][0]
single_z = stat['single_trial'][0][0]['z'][0][0]
single_z = single_z[0][0]
single_z = round(single_z,2)

#%% Plot histogram


fig, ax = plt.subplots(2,1)
ax[0].hist(perm_test, bins=nbins, density=True, facecolor='g', alpha=0.75)
ax[0].axvline(x=obs, color='k', label='Observed difference')
ax[0].text(0.02,20,f'z = {perm_z}')
ax[0].set_xlabel('Permutation test statistic')
ax[0].set_ylabel('Density')

ax[1].hist(single_F[0,:], bins=nbins, density=True, facecolor='b', alpha=0.75, 
           label='condition 1')
ax[1].hist(single_F[1,:], bins=nbins, density=True, facecolor='r', alpha=0.75,
           label='condition 2')
ax[1].text(1.2,3,f'z = {single_z}')
ax[1].set_xlabel('Single trial distribution ')
ax[1].set_ylabel('Density')

plt.tight_layout()
plt.legend()

#%% Save figure

fname = 'compare_permutation_single_trial.png'
figpath = figpath.joinpath(fname)
plt.savefig(figpath)