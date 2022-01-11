#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:21:52 2021
Try some spectral analyses
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


#%% Loading data parameters

cohort = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

# Path to subjects repository. Enter your own path in local machine
cohort_path = Path('~','projects', 'CIFAR', 'data', 'derivatives', 'iEEG_10', 
                   'subjects').expanduser()

parser = argparse.ArgumentParser()
parser.add_argument("--cohort_path", type=list, default=cohort_path)
parser.add_argument("--cohort", type=list, default=cohort)
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--proc", type=str, default='preproc')
parser.add_argument("--stage", type=str, default='_hfb_extracted_raw.fif')
parser.add_argument("--epoch", type=bool, default=False)
parser.add_argument("--channels", type=str, default='visual_channels.csv')


#%% Create category specific time series

parser.add_argument("--sfreq", type=float, default=150)
parser.add_argument("--tmin_crop", type=float, default=0.3)
parser.add_argument("--tmax_crop", type=float, default=1.5)

#%% Functional connectivity parameters

parser.add_argument("--nfreq", type=float, default=1024)
parser.add_argument("--roi", type=str, default="functional")

# Types of roi:
# functional
# anatomical

#%%

args = parser.parse_args()

#%% Prepare condition time series

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
hfb = ecog.read_dataset()

# Read ROI info for mvgc
df_visual = ecog.read_channels_info(fname=args.channels)
df_electrodes = ecog.read_channels_info(fname='electrodes_info.csv')
functional_indices = hf.parcellation_to_indices(df_visual, 'group', matlab=True)
visual_chan = df_visual['chan_name'].to_list()

# Read condition specific time series
# Read visual hfb
if args.stage == '_hfb_extracted_raw.fif':
    ts_type = 'hfb'
    ts, time = hf.category_ts(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                              tmax_crop=args.tmax_crop)
# Read visual lfp
else:
    ts_type = 'lfp'
    ts, time = hf.category_lfp(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                              tmax_crop=args.tmax_crop)

#%% Explore MNE spectral estimation:


#%% Epoch hfb

hfbDb = hf.Hfb_db()
hfb_db = hfbDb.hfb_to_db(hfb)

hfb_db.plot_psd(fmin=2., fmax=40., average=True, spatial_colors=False)

psds, freqs = mne.time_frequency.psd_welch(hfb_db, fmin=0, fmax=200, n_per_seg=100,
                                           tmin=0, tmax=1.5, n_fft=700, n_overlap= 50)

#%% Plot psd

psd = np.average(psds, 0)
for i in range(nchan):
    plt.plot(freqs, psd[i,:])
    plt.yscale("log")
    plt.xscale("log")
    #%% Plot cpsd from hFb
nchan = 14
psd, freqs = mne.time_frequency.psd_welch(hfb, fmin=0, fmax=100, n_fft=1024)
for i in range(nchan):
    plt.plot(freqs, psd[i,:])
    plt.yscale("log")
    plt.xscale("log")

#%% Plot cpsd from hfb_db
nchan = 14
psd, freqs = mne.time_frequency.psd_welch(hfb_db, average='mean', fmin=0, fmax=100, n_fft=1024)
for i in range(nchan):
    plt.plot(freqs, psd[1,i,:])
    plt.yscale("log")
    plt.xscale("log")

#%% plot cpsd from condition ts
    
ncdt = 2
X = ts[...,2]
X = np.transpose(X, (0,2,1))
psd, freqs = mne.time_frequency.psd_array_welch(X, sfreq=100,  
                                                fmin=0, fmax=100, n_overlap=50, n_fft=100 )

for i in range(nchan):
    plt.plot(freqs, psd[1,i,:])
    plt.yscale("log")
    plt.xscale("log")

#%%

import os
from datetime import timedelta
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.crop(tmax=60).load_data()