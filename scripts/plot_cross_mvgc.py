#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:39:09 2021
This script plot GC results across all subjects

@author: guime
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

#%%

result_path = Path('~','projects','CIFAR','data', 'results').expanduser()
fname = 'cross_sliding_mvgc.csv'
fpath = Path.joinpath(result_path, fname)
df = pd.read_csv(fpath)

#%% 

pairs = ['R->F', 'F->R']
#%matplotlib qt

cohort = list(df["subject"].unique())
nsub = len(cohort)
ncdt = 3
cdt = list(df["condition"].unique())
Fmax = df['F'].max()
Fmax = 0.05
f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
for s in range(nsub):
    for c in range(ncdt):
        for pair in pairs:
            baseline = df["Fb"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
            & (df["pair"]==pair)].to_numpy()
            time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
            & (df["pair"]==pair)].to_numpy()
            F = df['F'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
            & (df["pair"]==pair)].to_numpy()
            # Baseline rescale
            F = F/baseline
            ax[c,s].plot(time, F, label=pair)
            ax[c,s].axhline(y=1, color='k')
            ax[c,s].axvline(x=0, color ='k')
            #ax[c,s].set_ylim((0, 0.05))
            ax[0,s].set_title(cohort[s], fontsize = 20)
            ax[c,0].set_ylabel(f"GC {cdt[c]}", fontsize = 22)
            ax[2,s].set_xlabel("Time (s)", fontsize = 22)
            ax[c,s].tick_params(labelsize=20)
            
handles, labels = ax[c,s].get_legend_handles_labels()
f.legend(handles, labels, loc='upper right', fontsize = 22)
f.suptitle('MVGC across subjects', fontsize=22)

#%% 

pairs = ['R->R', 'F->F']
Fmax = df['F'].max()
Fmax = 0.004
f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
for c in range(ncdt):
    for s in range(nsub):
        for pair in pairs:
            time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
            & (df["pair"]==pair)].to_numpy()
            F = df['F'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
            & (df["pair"]==pair)].to_numpy()
            ax[c,s].plot(time, F, label=pair)
            ax[c,s].set_ylim(0,Fmax)
            ax[0,s].set_title(cohort[s], fontsize = 20)
            ax[c,0].set_ylabel(f"GC {cdt[c]}", fontsize = 22)
            ax[2,s].set_xlabel("Time (s)", fontsize = 22)
            ax[c,s].tick_params(labelsize=20)
            
handles, labels = ax[c,s].get_legend_handles_labels()
f.legend(handles, labels, loc='upper right', fontsize = 22)
f.suptitle('Causal density across subjects', fontsize=22)