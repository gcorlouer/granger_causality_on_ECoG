#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:05:48 2022
In this script we plot MVGC on multitrial
@author: guime
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.preprocessing_lib import EcogReader, plot_pmvgc_null
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path

#%% Read data

# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs'];
# Load functional connectivity matrix
result_path = Path('../results')

fname = 'mvgc.mat'
fpath = result_path.joinpath(fname)

mvgc = loadmat(fpath)
mvgc = mvgc['dataset']

#%%

(ncdt, nsub) = mvgc.shape
    fig, ax = plt.subplots(ncdt,2, figsize=(15,15))
    populations = df_visual['group'].to_list()
    for c in range(ncdt):
        condition =  mvgc[c,s]['condition'][0]
        # Granger causality matrix
        f = mvgc[c,s]['F']
        sig_gc = mvgc[c,s]['sigF']
        # Plot GC as heatmap
        g = sns.heatmap(f, xticklabels=populations,
                        yticklabels=populations, cmap='YlOrBr', ax=ax[c,1])
        g.set_yticklabels(g.get_yticklabels(), rotation = rotation)
        # Position xticks on top of heatmap
        ax[c, 1].xaxis.tick_top()
        ax[c, 1].set_ylabel('Target')
        ax[0,1].set_title('Transfer entropy (bit/s)')
        # Plot statistical significant entries
        for y in range(f.shape[0]):
            for x in range(f.shape[1]):
                if sig_mi[y,x] == 1:
                    ax[c,0].text(x + tau_x, y + tau_y, '*',
                             horizontalalignment='center', verticalalignment='center',
                             color='k')
                else:
                    continue
                if sig_gc[y,x] == 1:
                    ax[c,1].text(x + tau_x, y + tau_y, '*',
                             horizontalalignment='center', verticalalignment='center',
                             color='k')
                else:
                    continue


