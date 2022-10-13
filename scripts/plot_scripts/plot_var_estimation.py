#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:59:39 2021
This script plot var estimation for all subjects. 
@author: guime
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata as ud

from pathlib import Path
from src.input_config import args

#%% Style parameters

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
# Paths
result_path = Path('~','projects','cifar', 'results')
fname = 'rolling_var_ecog_estimation.csv'
fpath = Path.joinpath(result_path, fname)
df = pd.read_csv(fpath)
# Parameters
cohort = ['AnRa',  'ArLa', 'DiAs']
nsub = len(cohort)
ncdt = 3
ic = ["aic", "hqc", "bic"]
cdt = list(df["condition"].unique())
# Model order max
pmax =20
# Range of spectral radius to plot
rho_min = 0.92
rho_max = 1
# HFA or LFP
signal = 'HFA'
# Ticks
pticks = [0, 5, 10]
#%% Plot varmodel order

#%matplotlib qt

f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)

for c in range(ncdt):
    for s in range(nsub):
        for i in ic:
            time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
            morder = df[i].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
            t0 = round(time[0], 2)
            tmax = round(time[-1], 2)
            xticks = [t0, 0, 0.5, 1, tmax]
            ax[c,s].plot(time, morder, label=i)
            ax[c,s].set_ylim(0,pmax)
            ax[0,s].set_title(f'Subject {s}')
            ax[c,0].set_ylabel(f'{cdt[c]}')
            ax[c,0].set_yticks(pticks,)
            ax[c,0].set_xticks(xticks)
            ax[2,s].set_xlabel("Time (s)")
            ax[c,s].axvline(x=0, color='k')
handles, labels = ax[c,s].get_legend_handles_labels()
f.legend(handles, labels, loc='upper right')
f.suptitle('Rolling model order', )
plt.tight_layout()
fpath = Path('~','thesis','overleaf_project', 'figures','method_figure').expanduser()
fname = signal + '_rolling_var.png'
figpath = fpath.joinpath(fname)
plt.savefig(figpath)
#%% Plot Spectral radius

f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)

#ypos = np.arange(0.95, 1, 0.01)
for c in range(ncdt):
    for s in range(nsub):
        time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
        rho = df["rho"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
        ax[c,s].plot(time, rho)
        ax[c,s].set_ylim(rho_min,rho_max)
        ax[c,0].set_xticks(xticks)
        ax[0,s].set_title(f'Subject {s}')
        ax[c,0].set_ylabel(rf'{cdt[c]}')
        ax[2,s].set_xlabel("Time (s)")
        ax[c,s].axvline(x=0, color='k')
        #ax[c,s].tick_params(labelsize=20)
            
handles, labels = ax[c,s].get_legend_handles_labels()
f.legend(handles, labels, loc='upper right')
f.suptitle('Rolling spectral radius', fontsize=22)
plt.tight_layout()
fname = signal + '_rolling_specrad.png'
figpath = fpath.joinpath(fname)
plt.savefig(figpath)



