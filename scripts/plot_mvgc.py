#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:39:16 2021

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

#%% Concatenate pre and post peak mvgc

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
        if i==j:
            pairs[i,j] = None
        else:
            pairs[i,j] = [f"{populations[j]} to {populations[i]}"]

#%% Reshape GC

(npop, npop, ncdt, nwin) = f.shape
newshape = (npop*npop, ncdt, nwin)
f = np.reshape(f, newshape)
fbm = np.reshape(fbm, newshape)
z = np.reshape(z, newshape)
pval = np.reshape(pval, newshape)
sig = np.reshape(sig, newshape)

nsample = fb.shape[-2]
newshape = (npop*npop, ncdt, nwin, nsample)
fb = np.reshape(fb, newshape)

newshape = (npop*npop)
pairs = np.reshape(pairs, newshape)
#%% Drop row with 0 elements
 
mask = np.all(f==0, axis=(1,2))
idx = np.where(mask==False)[0]

f = f[idx,...]
fb = fb[idx, ...]
fbm = fbm[idx, ...]
sig = sig[idx,...]
z = z[idx, ...]
pval = pval[idx, ...]
pairs = pairs[idx]

#%% Permute indices to make compatible with subplot
#npair = len(pairs)
#fb = np.transpose(fb, (2,0,1,3))
#newshape = (npair, nwin, ncdt, nsample)
#fb = np.reshape(fb, newshape)
#
#newPair = np.asarray([pairs, pairs])
##newPair = np.transpose(newPair)
#newPair = np.reshape(newPair, (npair, nwin))
#%% Plot histogram
#%matplotlib qt
nbin = 100
npair = 6;

plt.rcParams['font.size'] = '11'

for i in range(npair):
    for c in range(ncdt):
        plt.subplot(3,2,i+1)
        plt.hist(fb[i,c,0,:], bins = nbin, density=True, stacked=True)
        plt.title(pairs[i][0])
        plt.xlim((0, 0.10))
        
fig = plt.gcf()
fig.suptitle('MVGC histogram,' + peak + ' single subject', fontsize=14)
fig.legend(['Rest','Face','Place'])
#%% Plot boxplot
# Note estimate IQR with bootstrapp
#%matplotlib qt

for i in range(npair):
    fbl = []
    for c in range(ncdt):
        fbl.append(fb[i,c,0,:])
    plt.subplot(3,2,i+1)
    plt.boxplot(fbl, vert=1, notch=True,showfliers=False)
    plt.title(pairs[i][0])
    x2, x3 = 2, 3
    y = np.max(fbl)
    h = 0.001
    plt.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c='k')
    plt.text((x2 + x3)*0.5, y+h, "s", ha='center', va='bottom', color='k')
    #plt.xticks([])
    plt.ylim((0,0.05))
    
fig = plt.gcf()
fig.suptitle('MVGC box plot,' + peak + ' single subject', fontsize=14)
#fig.legend(['Rest','Face','Place'])

