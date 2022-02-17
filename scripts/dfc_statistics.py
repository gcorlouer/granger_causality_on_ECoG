#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:40:03 2022
In this script we estimate the statistics of pairwise MI and GC.
@author: guime
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.preprocessing_lib import EcogReader, build_dfc
from src.input_config import args
from scipy.io import loadmat
from scipy.stats import ranksums
from pathlib import Path
from statsmodels.stats.multitest import fdrcorrection

#TODO: take same number of sample as baseline
#%%%


reader = EcogReader(args.data_path)
# Read visual channels 
df_visual = reader.read_channels_info(fname='visual_channels.csv')

# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']

# Load functional connectivity matrix
result_path = Path('../results')

fname = 'pairwise_fc.mat'
fc_path = result_path.joinpath(fname)

fc = loadmat(fc_path)
fc = fc['dataset']

#%% Build dataset fc dictionary

dfc = build_dfc(fc)

#%% Compute z score between conditions

baseline = 'baseline'
comparisons = [(baseline,'Face'), (baseline, 'Place'), ('Place','Face')]
subject = 'DiAs'
gc1 = dfc['sgc'].loc[(dfc['subject']==subject) & (dfc['condition']==baseline)]
gc1 = gc1.iloc[0]
(n,n,N) = gc1.shape
ncomp = len(comparisons)

z = np.zeros((n,n,ncomp))
pval = np.zeros((n,n,ncomp))

for icomp in range(ncomp):
    gc1 = dfc['sgc'].loc[(dfc['subject']==subject) & (dfc['condition']==comparisons[icomp][0])]
    gc1 = gc1.iloc[0]
    gc2 =  dfc['sgc'].loc[(dfc['subject']==subject) & (dfc['condition']==comparisons[icomp][1])]
    gc2 = gc2.iloc[0]
    # Test wether gc2 stochastically dominate gc1
    for i in range(n):
        for j in range(n):
            z[i,j,icomp], pval[i,j,icomp] = ranksums(gc2[i,j,:], gc1[i,j,:], alternative='two-sided')
pval = np.ndarray.flatten(pval)
rejected, pval_corrected = fdrcorrection(pval,alpha=0.05)
#%% Restrict pcgc over R to F channels
icomp = 0
subject = 'DiAs'

# Read visual channels
reader = EcogReader(args.data_path, subject=subject)
df_visual = reader.read_channels_info(fname='visual_channels.csv')
group = df_visual['group']

# Prepare gc in two distinct conditions
gc1 = dfc['sgc'].loc[(dfc['subject']==subject) & (dfc['condition']==comparisons[icomp][0])]
gc1 = gc1.iloc[0]
gc2 =  dfc['sgc'].loc[(dfc['subject']==subject) & (dfc['condition']==comparisons[icomp][1])]
gc2 = gc2.iloc[0]


#%% Plot distribution of pgc 

# Get R and F indices
indices = dfc['visual_idx'].loc[(dfc['subject']==subject)].iloc[0]
F_idx = indices['F']
R_idx = indices['R']
RF_idx = sorted(F_idx + R_idx)
groupRF = [group[i] for i in RF_idx]

# Restrict gc to R and F pairs
gc1 = gc1[RF_idx, :,:]
gc1 = gc1[:, RF_idx,:]
gc2 = gc2[RF_idx, :,:]
gc2 = gc2[:, RF_idx,:]

nbins = 20
nRF = len(RF_idx)
alpha = 0.6
f, ax = plt.subplots(nRF, nRF)
for i in range(nRF):
    for j in range(nRF):
        f1 = gc1[i,j,:]
        f2 = gc2[i,j,:]
        ax[i,j].hist(f1, bins = nbins, density=True, label='Baseline', 
          alpha=alpha, log=True)
        ax[i,j].set_label('baseline')
        ax[i,j].hist(f2, bins = nbins, density=True, label='Face',alpha=alpha,
          log=True)
        ax[i,j].set_label('face')
        
for i in range(nRF):
    ax[i,0].set_ylabel(f'{groupRF[i]}')
for j in range(nRF):
    ax[-1,j].set_xlabel(f'{groupRF[j]}')
    
plt.legend()

#%% Plot distrib of pcgc accross all pairs
nbins = 20

sns.set_theme()
sns.set(font_scale=2)  # crazy big

gc1 = np.ndarray.flatten(gc1)
gc2 = np.ndarray.flatten(gc2)


plt.hist(gc1, bins = nbins, density=True, label='Baseline', 
          alpha=alpha, log=True)
plt.hist(gc2, bins=nbins, density=True, label='Face', 
          alpha=alpha, log=True)
plt.legend()

z, pval = ranksums(gc2, gc1, alternative='less')
z = round(z,2)
pval = round(pval,3)
plt.title(f'Dias pairwise GC across all pairs, z score={z}, p={pval}')

#%% Plot distrib of pcmi accross all pairs
nbins = 20
icomp = 0 
# Prepare gc in two distinct conditions
mi1 = dfc['smi'].loc[(dfc['subject']==subject) & (dfc['condition']==comparisons[icomp][0])]
mi1 = mi1.iloc[0]
mi2 =  dfc['smi'].loc[(dfc['subject']==subject) & (dfc['condition']==comparisons[icomp][1])]
mi2 = mi2.iloc[0]

sns.set_theme()
sns.set(font_scale=2)  # crazy big

mi1 = np.ndarray.flatten(mi1)
mi2 = np.ndarray.flatten(mi2)


plt.hist(mi1, bins = nbins, density=True, label='Baseline', 
          alpha=alpha, log=True)
plt.hist(mi2, bins=nbins, density=True, label='Face', 
          alpha=alpha, log=True)
plt.legend()

z, pval = ranksums(mi1, mi2, alternative='less')
z = round(z,2)
pval = round(pval,3)
plt.title(f'Dias pairwise MI across all pairs, z score={z}, p={pval}')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#
#
#
##
#
##
#
#
#
#
#
#
#
###
#
#
#
#
#
#
#
#
#
###
#
#
#
#
#
#
#
#
#

#
#
#
