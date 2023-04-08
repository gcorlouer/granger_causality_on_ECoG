#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:40:00 2022

@author: guime
"""

import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

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
          'lines.linewidth': 0.5,
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

#%%


cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')

eeg_bands = {"[4 7]": "θ", "[8 12]": "α", "[13 30]": "β",
             "[32 60]": "γ", "[60 120]":"hγ", "[0 62]":"hfa"}
bands = list(eeg_bands.keys())
comparisons = ['FvsR','PvsR', 'FvsP']
comparisons_dic = {'FvsR':'Face/Rest', 'PvsR':'Place/Rest', 'FvsP':'Face/Place'}
directions = {"TD": [0,1],"BU":[1,0]} # TD and BU directions
directions_dic = {"TD":"Top-down", "BU":"Bottom-up"}
ndir = len(directions)
ncomp = len(comparisons)
nbands = len(bands)
ymax = 0.2
ymin = -0.2
xticks = []
gc = []
colors = []
fig, ax = plt.subplots(ncomp,ndir)
for idir, direction in enumerate(list(directions.keys())):
    source = directions[direction][1]
    target = directions[direction][0]
    for c, comparison in enumerate(comparisons):
        xticks = []
        gc = []
        colors = []
        for i, band in enumerate(bands):
                fname = "two_chans_compare_gc_"+ band + "Hz.mat"
                path = result_path
                fpath = path.joinpath(fname)
                # Read dataset
                dataset = loadmat(fpath)
                F = dataset['GC']
                F = F["DiAs"][0][0] #We only look at channels from DiAs
                band_name = eeg_bands[band]
                z = F[comparison][0][0]['z'][0][0]
                z = z[target, source]
                sig = F[comparison][0][0]['sig'][0][0][0][0]
                z_crit_plus =1.96
                z_crit_minus = -z_crit_plus
                xticks.append(f'{band_name}')
                gc.append(z) # condition, band specific gc z score
                if z>=0:
                    color = 'orange'
                elif z<=0:
                    color = 'purple'
                pcrit = F[comparison][0][0]['pval'][0][0][0][0]
                zcrit = F[comparison][0][0]['zcrit'][0][0][0][0]
                print(f"pval={pcrit}\n")
                print(f"z={z}\n")
                print(f"zcrit={zcrit}\n")
                colors.append(color)
        ax[c,idir].bar(xticks, gc, width=0.1, color=colors)
        ax[c,idir].set_ylim(ymin, ymax)
        ax[c,idir].set_ylabel(f'{comparisons_dic[comparison]}')
        ax[0,idir].set_title(f'Z-score, {directions_dic[direction]}')
        rects = ax[c,idir].patches
        ax[c,idir].axhline(y=z_crit_plus, color='r')
        ax[c,idir].axhline(y=z_crit_minus, color='r')