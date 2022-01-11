#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:58:19 2021

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

subject = "DiAs"

# Anatomical info
ecog = hf.Ecog(args.cohort_path, subject=subject, proc=args.proc, 
                           stage = args.stage, epoch=args.epoch)
df_visual = ecog.read_channels_info(fname=args.channels)
functional_group = df_visual["group"].unique()
functional_indices = hf.parcellation_to_indices(df_visual, 'group', matlab=True)
visual_chan = df_visual['chan_name'].to_list()

# Read results

result_path = Path('~','neocortex', 'results').expanduser()
fname = subject + '_GC_sliding.mat'
gc_path = result_path.joinpath(fname)

gc = loadmat(gc_path)

fb = gc['Fb']
sig = gc['sig']
z = gc['z']
pval = gc['pval']
win_time = gc['win_time']

pval = np.around(pval, 5)
(npop, npop, nwin, ncdt, nsample) = fb.shape
z = z/(np.sqrt(nsample)) 
z = np.round(z,2)
#%% 

populations = list(functional_indices.keys())
npop = len(populations)
pairs = np.zeros((npop, npop), dtype=list)

for i in range(npop):
    for j in range(npop):
        pairs[i,j] = f"{populations[j]} -> {populations[i]}"
            


#%% Plot boxplot of gc between R->F and F->R during face presentation
#%matplotlib qt
vpop = list(functional_indices.keys())
iF = vpop.index("F")
iR = vpop.index("R")
npair = 2

apair = np.array([[iF,iR],[iR,iF]])
f, ax = plt.subplots(ncdt, npair,sharex=True, sharey=False)
for c in range(ncdt):
    for i in range(npair):
        fbl = []
        for w in range(nwin):
            fbl.append(fb[apair[i,0],apair[i,1],w,c,:])
        ax[c,i].boxplot(fbl, notch=True,showfliers=False)
        ax[c,i].set_title(f"{vpop[apair[i,1]]} -> {vpop[apair[i,0]]}")
f.suptitle('Bootstrapp GC Box plot single subject', fontsize=14)

#%% Plot mvgc time series
    
sfreq = args.sfreq
fbm = np.median(fb, axis=-1)
f, ax = plt.subplots(ncdt, 1,sharex=True, sharey=False)
for c in range(ncdt):
    for i in range(npair):
        ax[c].plot(win_time[:,-1], fbm[apair[i,0],apair[i,1], :, c], 
          label=f"{vpop[apair[i,1]]} -> {vpop[apair[i,0]]}")
        ax[c].legend()
f.suptitle('Sliding GC plot', fontsize=14)
