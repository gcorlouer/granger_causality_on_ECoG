#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:14:03 2022

@author: guime
"""
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import mne 

from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path

#%%

cohort = ['AnRa',  'ArLa', 'DiAs']
# Useful paths
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')
fname = 'time_frequency_gc.mat'
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
F = dataset['GC']
subject = 'DiAs'
condition = 'Face'
# Plot parameters
populations = ['R','O', 'F']
vmax = 0.05
vmin = -vmax
#%%

gF = F[subject][0][0][condition][0][0]['gF'][0][0]
# Baseline GC
gFb = F[subject][0][0]['Rest'][0][0]['gF'][0][0]
time = F[subject][0][0][condition][0][0]['time'][0][0]
freqs = F['freqs'][0][0]
(ng, ng, nobs, nfreqs) = gF.shape

# Zscoring of F by resting state
mFb = np.mean(gFb, axis=2)
mFb = mFb[:,:,np.newaxis,:]
sFb = np.std(gFb, axis=2)
sFb = sFb[:,:,np.newaxis,:]
z = np.zeros_like(F)
z = gF - mFb
#%%

x = time[:,0]
y = freqs[:,0]

fig, ax = plt.subplots(ng, ng)
for i in range(ng):
    for j in range(ng):
        mesh = ax[i,j].pcolormesh(x, y, z[i,j,:,:].T, cmap='RdBu_r', 
                                  vmin=vmin, vmax=vmax, shading='auto')
for i in range(ng):
    ax[0,i].set_xlabel(populations[i])
for i in range(ng):
    ax[0,i].xaxis.set_label_position('top') 
    ax[i,0].set_ylabel(populations[i])
ax[-1,-1].set_xlabel('Time (s)')
ax[-1,-1].set_ylabel('Frequency (Hz)')
fig.colorbar(mesh, ax = ax[-1,-1])
plt.tight_layout()
plt.show()

