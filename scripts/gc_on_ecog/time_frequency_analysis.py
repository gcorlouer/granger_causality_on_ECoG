#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 22:19:26 2022
In this script we run time frequency analysis of ECoG data
@author: guime
"""

#%%
import mne 
import matplotlib.pyplot as plt
import numpy as np 

from src.input_config import args
from src.preprocessing_lib import EcogReader, Epocher
from mne.time_frequency import tfr_morlet

#%% Parameters
channel = ['LTo1-LTo2']
condition = 'Face'
band = 'alpha'
freq_bands = {'delta' : [1, 3], 'theta':[4,7], 'alpha':[8, 12], 'beta':[13,30],
              'gamma':[30, 70], 'high_gamma':[70,124], 'spectrum':[0.1, 124]}

freqs = np.arange(1.,120.,0.1)
n_cycles = freqs/2
mode = 'zscore'
vmax =50
vmin = -vmax
baseline = (-0.4, 0)

#%% Epoch continuous ecog

reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                     preprocessed_suffix=args.preprocessed_suffix,
                     epoch=args.epoch)
raw = reader.read_ecog()
df_visual = reader.read_channels_info(fname='visual_channels.csv')
visual_chans = df_visual['chan_name'].to_list()
raw = raw.pick_channels(visual_chans)

epocher = Epocher(condition=condition, t_prestim=args.t_prestim, t_postim = args.t_postim, 
                         baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                         tmax_baseline=args.tmax_baseline, mode=args.mode)
epochs = epocher.epoch(raw)
epochs = epochs.pick(channel)
epochs = epochs.filter(l_freq=0.01, h_freq=None)
epochs = epochs.copy().decimate(args.decim)


#%% Estimate time frequency baseline rescaling when plotting

power = tfr_morlet(epochs, freqs, n_cycles, return_itc=False)

power.plot([0], baseline=baseline, vmin=vmin, mode = mode,
                   vmax=vmax)

power.apply_baseline(baseline=baseline, mode=mode)














