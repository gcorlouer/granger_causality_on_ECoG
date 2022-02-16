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
gc1 = dfc['sgc'].loc[(dfc['subject']==subject) & (dfc['condition']=='baseline')]
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
#%% 
    
    