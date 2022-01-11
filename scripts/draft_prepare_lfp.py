#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:22:17 2021
@author: guime
"""

#%% Import libraries

import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat

#%% Import dataset:

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
lfp = ecog.read_dataset(task = 'rest_baseline', run=1)

# Pick visual channels

df_visual = ecog.read_channels_info(fname=args.channels)
visual_chan = df_visual['chan_name'].to_list()
lfp = lfp.pick_channels(visual_chan)
#%% Plot spectral lfp 

%matplotlib qt

lfp.plot_psd(xscale='log')

#%% Plot time series

lfp.plot(duration = 10, scalings=5e-4, n_channels=14)

#%% Epoch time series:

events = mne.make_fixed_length_events(lfp, start=5, stop=205, duration=4, 
                                      first_samp=False, overlap=1)
epochs= mne.Epochs(lfp, events,
                            tmin=0, tmax=4, baseline= None, preload=True)
# Downsample
epochs.decimate(2)

time = epochs.times
epochs.plot_image()

#%% Save time series

X = epochs.get_data()
X = np.transpose(X, (1,2,0))
ts_dict = {"data": X, "time": time}
fname = args.subject + "_rest_epochs.mat"
fpath = Path('~','projects','CIFAR','data', 'test').expanduser()
fpath = fpath.joinpath(fname)
savemat(fpath, ts_dict)