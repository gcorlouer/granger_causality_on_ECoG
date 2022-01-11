#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 15:25:33 2021

@author: guime
"""


import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat, loadmat

#%% Plot mvgc

peak = '_post_peak.mat'
result_path = Path('~','projects', 'CIFAR','data', 'results').expanduser()
fname = args.subject + '_GC' + peak
gc_path = result_path.joinpath(fname)

gc = loadmat(gc_path)
# 

f = gc['F']
fb = gc['Fb']
fbm = gc['Fbm']
sig = gc['sig']
z = gc['z']
pval = gc['pval']
pval = np.around(pval, 5)
cdt_pair = np.array([[1,2],[1,3],[2,3]])
z = z/(np.sqrt(1000)) # or 2128 if taking number of observations instead of bootsrapp sample
z = np.round(z,1)
(npop, npop, ncdt, nwin, nsample) = fb.shape
#%% 

populations = list(functional_indices.keys())
npop = len(populations)
pairs = np.zeros((npop, npop), dtype=list)

for i in range(npop):
    for j in range(npop):
        pairs[i,j] = f"{populations[j]} -> {populations[i]}"
            
#%% Drop diagonal elements and reshape:
#%matplotlib qt

nbin = 50

plt.rcParams['font.size'] = '11'
f, ax = plt.subplots(npop, npop,sharex=False, sharey=True)
for i in range(npop):
    for j in range(npop):
        for c in range(ncdt):
            ax[i,i].set_visible(False)
            ax[i,j].hist(fb[i,j,c,0,:], bins=nbin, density=True)
            xmax = np.max(fb[i,j,c,0,:])
            ax[i,j].set_title(pairs[i,j])
            ax[i,j].set_xlim((0, xmax))
            
fig = plt.gcf()
fig.suptitle('MVGC histogram,' + peak + ' single subject', fontsize=14)
fig.legend(['Rest','Face','Place'])

#%% Plot boxplot


plt.rcParams['font.size'] = '14'
f, ax = plt.subplots(npop, npop,sharex=True, sharey=False)
for i in range(npop):
    for j in range(npop):
        fbl = []
        for c in range(ncdt):
            fbl.append(fb[i,j,c,0,:])
        ax[i,i].set_visible(False)
        ax[i,j].boxplot(fbl, notch=True,showfliers=False)
        ax[i,j].set_title(pairs[i,j])
        x1, x2, x3 = 1, 2, 3
        y = np.max(fb[i,j,1:2,0,:])
        h = 0.001
        ax[i,j].plot([x2, x2, x3, x3], [y, y + h, y+ h, y], lw=1.5, c='k')
        if sig[i,j,2]== 1:
            ax[i,j].text((x2 + x3)*0.5, y+h, f"*, z={z[i,j,2]}", ha='center', va='bottom', color='k')            
#        y = np.max(fb[i,j,[0, 2],0,:])
#        h2 = 30*h
#        ax[i,j].plot([x1, x1, x3, x3], [y, y+h2, y+h2, y], lw=1.5, c='k')
#        ax[i,j].text((x1 + x3)*0.5, y+h2, "ns", ha='center', va='bottom', color='k')
        y = np.max(fb[i,j, 0:1,0,:])
        ax[i,j].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        if sig[i,j,0]==1:
            ax[i,j].text((x1 + x2)*0.5, y+h, f"*, z={z[i,j,0]}", ha='center', va='bottom', color='k')
        #plt.xticks([])
        ym = np.max(fbl) + 10*h
        ax[i,j].set_ylim((0,ym))
        ax[i,j].set_ylabel("GCb")
        ax[i,j].set_xticks([1,2,3])
        ax[i,j].set_xticklabels(("Rest", "Face", "Place"))
f.suptitle('Bootstrapp GC Box plot,' + peak + ' single subject', fontsize=14)
# plt.xticks(["Rest", "Face", "Place"])
