#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:09:02 2021
This script plot the results of mvgc bootstrapp analysis. 
@author: guime
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from scipy.io import savemat, loadmat


#%% Read data
result_path = Path('~', 'projects', 'CIFAR', 'data', 'results').expanduser()
#result_path = Path('~','neocortex', 'results').expanduser()
fname = 'cross_sliding_mvgc_bootstrapp_test.csv'
fpath = Path.joinpath(result_path, fname)
df = pd.read_csv(fpath)

#%% Define plotting functions:

#
def plot_mvgc(df, pairs = ['R->F', 'F->R'], ncdt =3):
    
    cohort = list(df["subject"].unique())
    nsub = len(cohort)
    cdt = list(df["condition"].unique())
    f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
    for c in range(ncdt):
        for s in range(nsub):
            for pair in pairs:
                time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
                & (df["pair"]==pair)].to_numpy()
                Fbm = df['Fbm'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
                & (df["pair"]==pair)].to_numpy()
                baseline = df['Fbb'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
                & (df["pair"]==pair)].to_numpy()
                F = Fbm/baseline 
                ax[c,s].plot(time, F, label=pair)
                ax[0,s].set_title(cohort[s], fontsize = 20)
                ax[c,0].set_ylabel(f"GC {cdt[c]}", fontsize = 22)
                ax[2,s].set_xlabel("Time (s)", fontsize = 22)
                ax[c,s].tick_params(labelsize=20)
                ax[c,s].axvline(x=0, color='k')
                ax[c,s].axhline(y=1, color='k')
                
    handles, labels = ax[c,s].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper right', fontsize = 22)
    f.suptitle('MVGC along 400ms rolling window', fontsize=22)

def plot_zscore(df, pairs= ['R->F', 'F->R'], zmax = 15, ncdt = 3):
    
    cohort = list(df["subject"].unique())
    nsub = len(cohort)
    cdt = list(df["condition"].unique())
    zmin = -zmax
    zcrit = df['zcrit'][0]
    f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
    for c in range(ncdt):
        for s in range(nsub):
            for pair in pairs:
                time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
                & (df["pair"]==pair)].to_numpy()
                z = df['z'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
                & (df["pair"]==pair)].to_numpy()
                ax[c,s].plot(time, z, label=pair)
                ax[c,s].axhline(y=zcrit, color='r')
                ax[c,s].axhline(y=-zcrit, color='r')
                ax[c,s].axvline(x=0, color='k')
                ax[c,s].set_ylim(zmin,zmax)
                ax[0,s].set_title(cohort[s], fontsize = 20)
                ax[c,0].set_ylabel(f"Z score {cdt[c]}", fontsize = 22)
                ax[2,s].set_xlabel("Time (s)", fontsize = 22)
                ax[c,s].tick_params(labelsize=20)
                
    handles, labels = ax[c,s].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper right', fontsize = 22)
    f.suptitle('MVGC bootsrapp along 400ms rolling window', fontsize=22)

#%% Plot mvgc
#%matplotlib qt
plot_mvgc(df, pairs = ['R->F', 'F->R'])
plot_mvgc(df, pairs = ['R->R', 'F->F'])

#%% Plot Z score

pairs = ['R->F', 'F->R']
plot_zscore(df, pairs=pairs)
pairs = ['R->R', 'F->F']
plot_zscore(df, pairs=pairs)

#%% Plot Bootstrapp distribution

fname = 'cross_sliding_mvgc_distribution.mat'
fpath = Path.joinpath(result_path, fname)
fb = loadmat(fpath)
#fb= fb['Fb']
fb = fb['mFbb']
print(fB.shape)
(ng, ng, sample, nwin, ncdt) = fB.shape

win_idx = np.arange(0, nwin, 3)
win_time = time[win_idx]
f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)

#%% Plot median bootstrapp with statistical significance

#%matplotlib qt

pairs = ['R->F', 'F->R']

cohort = list(df["subject"].unique())
nsub = len(cohort)
ncdt = 3
cdt = list(df["condition"].unique())
Fmax = df['Fbm'].max()
Fmax = 0.08
f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
for c in range(ncdt):
    for s in range(nsub):
        for pair in pairs:
            time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
            & (df["pair"]==pair)].to_numpy()
            nwin = len(time)
            Fbm = df['Fbm'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
            & (df["pair"]==pair)].to_numpy()
            baseline = df['Fbbm'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
            & (df["pair"]==pair)].to_numpy()
            F = Fbm/baseline 
            
#            sig =  df['sig'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
#            & (df["pair"]==pair)].to_numpy()
#            for w in range(nwin):
#                if sig[w] == 1:
#                    ax[c,s].plot(time[w], Fbm[w], '*', color = 'k')
            ax[c,s].plot(time, F, label=pair)
            #ax[c,s].set_ylim(0,Fmax)
            ax[0,s].set_title(cohort[s], fontsize = 20)
            ax[c,0].set_ylabel(f"GCb {cdt[c]}", fontsize = 22)
            ax[2,s].set_xlabel("Time (s)", fontsize = 22)
            ax[c,s].tick_params(labelsize=20)
            ax[c,s].axvline(x=0, color='k')
            ax[c,s].axhline(y=1, color='k')
            
handles, labels = ax[c,s].get_legend_handles_labels()
f.legend(handles, labels, loc='upper right', fontsize = 22)
f.suptitle('MVGC bootsrapp for each subjects along 300ms rolling window', fontsize=22)

#%% Plot z score

pairs = ['R->F', 'F->R']
nsample = 100
nwin = len(time)

cohort = list(df["subject"].unique())
nsub = len(cohort)
ncdt = 3
cdt = list(df["condition"].unique())
zmax = 15
zmin = -zmax
zcrit = df['zcrit'][0]
pcrit = df['pcrit'][0]
f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
for c in range(ncdt):
    for s in range(nsub):
        for pair in pairs:
            time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
            & (df["pair"]==pair)].to_numpy()
            z = df['z'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
            & (df["pair"]==pair)].to_numpy()
            #sig =  df['sig'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
            #& (df["pair"]==pair)].to_numpy()
            ax[c,s].plot(time, z, label=pair)
            ax[c,s].axhline(y=zcrit, color='r')
            ax[c,s].axhline(y=-zcrit, color='r')
            ax[c,s].axvline(x=0, color='k')
            ax[c,s].set_ylim(zmin,zmax)
            ax[0,s].set_title(cohort[s], fontsize = 20)
            ax[c,0].set_ylabel(f"Z score {cdt[c]}", fontsize = 22)
            ax[2,s].set_xlabel("Time (s)", fontsize = 22)
            ax[c,s].tick_params(labelsize=20)
            
handles, labels = ax[c,s].get_legend_handles_labels()
f.legend(handles, labels, loc='upper right', fontsize = 22)
f.suptitle('MVGC bootsrapp along 400ms rolling window', fontsize=22)

#%% Plot Causal density 

pairs = ['F->F', 'R->R']

cohort = list(df["subject"].unique())
nsub = len(cohort)
ncdt = 3
cdt = list(df["condition"].unique())
Fmax = df['Fbm'].max()
Fmax = 0.08
f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
for c in range(ncdt):
    for s in range(nsub):
        for pair in pairs:
            time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
            & (df["pair"]==pair)].to_numpy()
            nwin = len(time)
            Fbm = df['Fbm'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
            & (df["pair"]==pair)].to_numpy()
            baseline = df['Fbbm'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
            & (df["pair"]==pair)].to_numpy()
            F = Fbm/baseline 
            
#            sig =  df['sig'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
#            & (df["pair"]==pair)].to_numpy()
#            for w in range(nwin):
#                if sig[w] == 1:
#                    ax[c,s].plot(time[w], Fbm[w], '*', color = 'k')
            ax[c,s].plot(time, Fbm, label=pair)
            ax[c,s].set_ylim(0,Fmax)
            ax[0,s].set_title(cohort[s], fontsize = 20)
            ax[c,0].set_ylabel(f"GCb {cdt[c]}", fontsize = 22)
            ax[2,s].set_xlabel("Time (s)", fontsize = 22)
            ax[c,s].tick_params(labelsize=20)
            
handles, labels = ax[c,s].get_legend_handles_labels()
f.legend(handles, labels, loc='upper right', fontsize = 22)
f.suptitle('MVGC bootsrapp for each subjects along 300ms rolling window', fontsize=22)


#%% Plot pvalues

pairs = ['R->F', 'F->R']
nsample = 100
nwin = len(time)

cohort = list(df["subject"].unique())
nsub = len(cohort)
ncdt = 3
cdt = list(df["condition"].unique())
pmax = 0.05
f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
for c in range(ncdt):
    for s in range(nsub):
        for pair in pairs:
            time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
            & (df["pair"]==pair)].to_numpy()
            pval = df["pval"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c]) 
            & (df["pair"]==pair)].to_numpy()
            sig =  df['sig'].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])
            & (df["pair"]==pair)].to_numpy()
            ax[c,s].plot(time, pval, label=pair)
            ax[c,s].set_ylim(0,pmax)
            ax[0,s].set_title(cohort[s], fontsize = 20)
            ax[c,0].set_ylabel(f"p value {cdt[c]}", fontsize = 22)
            ax[2,s].set_xlabel("Time (s)", fontsize = 22)
            ax[c,s].tick_params(labelsize=20)
            
handles, labels = ax[c,s].get_legend_handles_labels()
f.legend(handles, labels, loc='upper right', fontsize = 22)
f.suptitle('MVGC bootsrapp for each subjects along 400ms rolling window', fontsize=22)

