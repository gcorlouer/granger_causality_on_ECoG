#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 19:33:11 2021
Check recall data
@author: guime
"""


import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from config import args

#%%

sub_id = 'DiAs'
stim = 'faces'
run = 1

#%%

path = Path('~', 'projects', 'CIFAR', 'data', 'CIFAR_RECALL_DATA', 'DiAs', 
             'EEGLAB_datasets', 'raw_signal').expanduser()
fname = sub_id + '_freerecall' + '_' + stim + '_' + str(run) + '_preprocessed.set'
fpath = path.joinpath(fname)

# Read dataset

raw = mne.io.read_raw_eeglab(fpath, preload=True)

#%%

raw.plot(duration = 50, n_channels= 10, scalings = 1e-4)

#%%

stim = 'stimuli'
path = Path('~', 'projects', 'CIFAR', 'data', 'source_data', 'iEEG_10', 'subjects', 
            'DiAs', 'EEGLAB_datasets', 'bipolar_montage').expanduser()
fname = sub_id + '_freerecall' + '_' + stim + '_' + str(run) + '_preprocessed_BP_montage.set'
fpath = path.joinpath(fname)

# Read dataset

raw = mne.io.read_raw_eeglab(fpath, preload=True)

#%%

raw.plot(duration = 50, n_channels= 10, scalings = 1e-4)

