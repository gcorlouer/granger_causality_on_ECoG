#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:49:15 2022
In this script we plot validation of 2 chans GC.
@author: guime
"""

import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

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

#%%
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')

eeg_bands = {"[4 7]":"θ", "[8 12]": "α", "[13 30]": "β",
             "[32 60]": "γ", "[60 120]":"hγ"}
bands = list(eeg_bands.keys())
conditions = ['Rest', 'Face', 'Place']
directions = ["BU", "TD"]
nbands = len(bands)
ymax = 0.0006

fig, ax = plt.subplots(nbands,1)
for i, band in enumerate(bands):
    fname = "validate_GC_"+ band + "Hz.mat"
    path = result_path
    fpath = path.joinpath(fname)
    # Read dataset
    dataset = loadmat(fpath)
    F = dataset['GC']
    xticks = []
    gc = []
    for condition in conditions:
            z = F[condition][0][0]
            bu = z[0,1] 
            td = z[1,0]
            bu_td = [bu, td]
            for j, direction in enumerate(directions):
                xticks.append(condition + ' ' +  direction)
                gc.append(bu_td[j])
    ax[i].bar(xticks, gc, width=0.5)
    ax[i].set_ylim(0, ymax)