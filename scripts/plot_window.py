#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:11:14 2021

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

#%% Read condition specific time series

subject = 'DiAs'
fname = subject + '_condition_ts_visual.mat'
result_path = Path('~','projects','CIFAR','data', 'results').expanduser()
fpath = result_path.joinpath(fname)
ts = loadmat(fpath)
findices = ts['functional_indices']
X = ts['data']

(n,m,N,nc) = X.shape
#%% Create windows


lw = 60
shift = 60
nw = int(np.floor((m - lw)/shift + 1))
W = np.zeros((n,lw,N,nc,nw))

for w in range(nw):
    o = w * shift
    W[:,:,:,:,w] = X[:,o:o+lw,:,:]

#%% Plot windows trials
# TODO create win_time and plot along time
    
#%matplotlib qt
iF = list(findices['F'][0][0][0] - 1)
cdt = ['rest', 'face', 'place']
mW = np.mean(W, axis = 2)
time = ts['time']

wtime = np.zeros((nw, lw))

for w in range(nw):
    o = w * shift
    wtime[w,:] = time[:,o:o+lw]
    
for w in range(nw):
    for c in range(nc):
        x = np.mean(mW[iF, :, c, w], axis=0)
        plt.subplot(3,3,w+1)
        plt.plot(wtime[w,:], x, label = f'{cdt[c]}')
        wt = np.round(wtime[w,-1],1)
        plt.xlabel(f"{wt}s")
        plt.ylabel("HFA face")
        plt.ylim((-0.5,3))
        plt.figlegend(cdt, loc = 'upper right')

