#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:22:17 2021

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


#%% Read GC results

peak = '_pre_peak.mat'
result_path = Path('~','projects', 'CIFAR','data', 'results').expanduser()
fname = args.subject + '_GC' + peak
gc_path = result_path.joinpath(fname)

gc = loadmat(gc_path)
# 

f_pre = gc['F']
fb_pre = gc['Fb']
fbm_pre = gc['Fbm']
sig_pre = gc['sig']
z_pre = gc['z']
pval_pre = gc['pval']

#%% Read prepeak data
peak = '_post_peak.mat'
result_path = Path('~','projects', 'CIFAR','data', 'results').expanduser()
fname = args.subject + '_GC' + peak
gc_path = result_path.joinpath(fname)

gc = loadmat(gc_path)
# 

f_post = gc['F']
fb_post = gc['Fb']
fbm_post = gc['Fbm']
sig_post = gc['sig']
z_post = gc['z']
pval_post = gc['pval']

#%%

f = np.stack([f_pre , f_post],axis=-1)
fb = np.stack([fb_pre , fb_post], axis=-1)
fbm = np.stack([fbm_pre, fbm_post],axis=-1)
sig = np.stack([sig_pre, sig_post], axis=-1)
z = np.stack([z_pre, z_post],axis=-1)
pval = np.stack([pval_pre, pval_post],axis=-1)


#%% 

populations = list(functional_indices.keys())
npop = len(populations)
pairs = np.zeros((npop, npop), dtype=list)

for i in range(npop):
    for j in range(npop):
        pairs[i,j] = f"{populations[j]} -> {populations[i]}"

#%% Concatenate face and retinotopic channels

iF = 1
iR = 2
nsample = 1000 

f_pair = np.stack([fb[iF,iR,...],fb[iR,iF,...]], axis=0)
f_pair = np.squeeze(f_pair, axis = 2)
npair = f_pair.shape[0]

sig_pair = np.stack([sig[iF,iR,...], sig[iR,iF,...]], axis=0)
z_pair = np.stack([z[iF,iR,...], z[iR,iF,...]], axis=0)
z_pair = np.round(z_pair/np.sqrt(nsample),1)
pair = ["R -> F", "F -> R"]

(npair, ncdt, nsample, nwin) = f_pair.shape
#%% Compare histograms from face to retinotopic channels

nbin = 50

plt.rcParams['font.size'] = '18'
fig, ax = plt.subplots(2, 2,sharex=False, sharey=True)
for w in range(nwin):
    for i in range(npair):
        for c in range(ncdt):
            ax[i,w].hist(f_pair[i,c,:,w], bins=nbin, density=True)
            #xmax = np.max(f_pair[i,c,:,:])
            ax[i,w].set_title(pair[i])
            ax[i,w].set_xlim((0, 0.10))

fig = plt.gcf()
fig.suptitle('MVGC between R and F pre-peak and post peak single subject', fontsize=14)
fig.legend(['Rest','Face','Place'])


#%% Compare boxplot from face to retinotopic channels
#%matplotlib qt

sns.set(font_scale=2)

plt.rcParams['font.size'] = '18'

fig, ax = plt.subplots(2, 2,sharex=True, sharey=True)
for w in range(nwin):
    for i in range(npair):
        fl = []
        for c in range(ncdt):
            fl.append(f_pair[i,c,:,w])
        ax[i,w].boxplot(fl, notch=True,showfliers=False)
        ax[i,w].set_title(pair[i])
        x1, x2, x3 = 1, 2, 3
        y = np.max(f_pair[i,c,:,w])
        h = 0.001
        ax[i,w].plot([x2, x2, x3, x3], [y, y + h, y+ h, y], lw=1.5, c='k')
        if sig_pair[i,2,w]== 1:
            ax[i,w].text((x2 + x3)*0.5, y+h, f"*, z={z_pair[i,2,w]}", ha='center', va='bottom', color='k')            
#        y = np.max(fb[i,j,[0, 2],0,:])
#        h2 = 30*h
#        ax[i,j].plot([x1, x1, x3, x3], [y, y+h2, y+h2, y], lw=1.5, c='k')
#        ax[i,j].text((x1 + x3)*0.5, y+h2, "ns", ha='center', va='bottom', color='k')
        y = np.max(fb[i,j, 0:1,0,:])
        ax[i,w].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        if sig_pair[i,0,w]==1:
            ax[i,w].text((x1 + x2)*0.5, y+h, f"*, z={z_pair[i,0,w]}", ha='center', va='bottom', color='k')
        #plt.xticks([])
        ym = np.max(fl) + 20*h
        ax[i,w].set_ylim((0,ym))
        ax[i,w].set_ylabel("GCb")
        ax[i,w].set_xticks([1,2,3])
        ax[i,w].set_xticklabels(("Rest", "Face", "Place"))
fig.suptitle('Bootstrapp GC Box plot, preeak (left), post peak (right) single subject', fontsize=14)
# plt.xticks(["Rest", "Face", "Place"])
