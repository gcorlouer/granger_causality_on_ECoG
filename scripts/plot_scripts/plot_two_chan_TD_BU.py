#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:09:59 2022

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

eeg_bands = {"[8 12]": "α", "[13 30]": "β",
             "[32 60]": "γ", "[60 120]":"hγ"}
bands = list(eeg_bands.keys())
conditions = ['Face', 'Place']
nbands = len(bands)
ymax = 4
ymin = -4
xticks = []
gc = []
color = []
fig, ax = plt.subplots(1,1)
for i, band in enumerate(bands):
    fname = "two_chans_TD_BU_GC_"+ band + "Hz.mat"
    path = result_path
    fpath = path.joinpath(fname)
    # Read dataset
    dataset = loadmat(fpath)
    F = dataset['GC']
    band_name = eeg_bands[band]
    for condition in conditions:
        z = F[condition][0][0]['z'][0][0][0][0]
        sig = F[condition][0][0]['sig'][0][0][0][0]
        
        xticks.append(condition + f' {band_name}')
        gc.append(z)
        if z>=0:
            c = 'r'
        elif z<=0:
            c = 'b'
        color.append(c)
        ax.bar(xticks, gc, width=0.1, color=color)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('Z-score')
        rects = ax.patches
        label = '*'
        if sig==1:
            pcrit = F[condition][0][0]['pval'][0][0][0][0]
            zcrit = F[condition][0][0]['zcrit'][0][0][0][0]
            print(f"pval={pcrit}\n")
            print(f"z={z}\n")
            print(f"zcrit={zcrit}\n")
            for rect in rects:
                height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height + 0.1, label, ha="center", va="bottom"
            )
        else:
             continue 