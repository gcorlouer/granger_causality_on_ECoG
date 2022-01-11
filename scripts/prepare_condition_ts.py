#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:43:20 2021
This script prepare condition specific time series for further analysis in 
mvgc toolbox
@author: guime
"""

#%%

import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat


#%% Read baseline time series

tmin_crop = -0.5
tmax_crop = 0

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
hfb = ecog.read_dataset()

# Read ROI info for mvgc
df_visual = ecog.read_channels_info(fname=args.channels)
df_electrodes = ecog.read_channels_info(fname='electrodes_info.csv')
functional_indices = hf.parcellation_to_indices(df_visual, 'group', matlab=True)
visual_chan = df_visual['chan_name'].to_list()

#%% Read condition specific time serieschannels pre peak

# Read visual hfb
if args.stage == '_hfb_extracted_raw.fif':
    ts_type = 'hfb'
    ts, time = hf.test_category_ts(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=tmin_crop, 
                              tmax_crop=tmax_crop)
# Read visual lfp
else:
    ts_type = 'lfp'
    ts, time = hf.test_category_lfp(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                              tmax_crop=args.tmax_crop)

#%% Save time series for GC analysis

ts_dict = {'rest': ts[0], 'face': ts[1], 'place': ts[2], 'sfreq': args.sfreq, 
           'time': time, 'sub_id': args.subject, 
           'functional_indices': functional_indices, 'ts_type':ts_type}
fname = args.subject + '_condition_ts_visual_baseline_test.mat'
result_path = Path('~','projects','CIFAR','data', 'results').expanduser()
fpath = result_path.joinpath(fname)
savemat(fpath, ts_dict)

#%% Read time series

tmin_crop = 0
tmax_crop = 1.5

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
hfb = ecog.read_dataset()

#%% Read condition specific time serieschannels pre peak

# Read visual hfb
if args.stage == '_hfb_extracted_raw.fif':
    ts_type = 'hfb'
    ts, time = hf.category_ts(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=tmin_crop, 
                              tmax_crop=tmax_crop)
# Read visual lfp
else:
    ts_type = 'lfp'
    ts, time = hf.category_lfp(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                              tmax_crop=args.tmax_crop)

#%% Save time series for GC analysis

ts_dict = {'data': ts, 'sfreq': args.sfreq, 'time': time, 'sub_id': args.subject, 
           'functional_indices': functional_indices, 'ts_type':ts_type}
fname = args.subject + '_condition_ts_visual.mat'
result_path = Path('~','projects','CIFAR','data', 'results').expanduser()
fpath = result_path.joinpath(fname)
savemat(fpath, ts_dict)
