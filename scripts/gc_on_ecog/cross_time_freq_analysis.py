#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:29:20 2022
In this script we run cross subjects time frequency analysis
@author: guime
"""


import mne 
import matplotlib.pyplot as plt
import numpy as np 

from src.input_config import args
from src.preprocessing_lib import EcogReader, Epocher, parcellation_to_indices
from src.preprocessing_lib import prepare_condition_scaled_ts
from mne.time_frequency import tfr_morlet


#%% Functions

def visual_indices(args, subject='DiAs'):
    """
    Return indices of each functional group for a given subject
    Input: 
        - data_path (string): where data of cifar project is stored 
        - subject (string): subject name
    Output:
        - indices (dict): indices of each functional group
    """
    # Read visual channel dataframe
    reader = EcogReader(args.data_path, subject=subject)
    df_visual = reader.read_channels_info(fname=args.channels)
    # Return indices of functional groups from visual channel dataframe
    indices = parcellation_to_indices(df_visual, parcellation='group', matlab=False)
    return indices 

def compute_power(args, freqs, condition = 'Face', l_freq = 0.01,
                 mode = 'zscore'):
    """Compute power of visually responsive epochs for time freq analysis"""
    # Read ECoG
    reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                         preprocessed_suffix=args.preprocessed_suffix,
                         epoch=args.epoch)
    raw = reader.read_ecog()
    # Read visually responsive channels
    df_visual = reader.read_channels_info(fname='visual_channels.csv')
    visual_chans = df_visual['chan_name'].to_list()
    raw = raw.pick_channels(visual_chans)
    # Epoch raw ECoG
    epocher = Epocher(condition=condition, t_prestim=args.t_prestim, t_postim = args.t_postim, 
                             baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                             tmax_baseline=args.tmax_baseline, mode=args.mode)
    epochs = epocher.epoch(raw)
    # High pass filter
    # epochs = epochs.filter(l_freq=l_freq, h_freq=None)
    # Downsample
    #epochs = epochs.decimate(args.decim)
    # Compute time frequency with Morlet wavelet 
    n_cycles = freqs/2
    power = tfr_morlet(epochs, freqs, n_cycles, return_itc=False)
    # Apply baseline correction
    baseline = (args.tmin_baseline, args.tmax_baseline)
    print(f"\n Computing power from morlet wavelet: rescale with {mode}")
    print(f"\n Condition is {condition}\n")
    power.apply_baseline(baseline=baseline, mode=mode)
    return power
    
#%% Parameters
condition = 'Face'
freq_bands = {'delta' : [1, 3], 'theta':[4,7], 'alpha':[8, 12], 'beta':[13,30],
              'gamma':[30, 70], 'high_gamma':[70,124], 'spectrum':[0.1, 124]}

freqs = np.arange(1.,120.,0.1)
n_cycles = freqs/2
mode = 'zscore'
vmax =50
vmin = -vmax
baseline = (-0.4, 0)


#%% Epoch continuous ecog



power = compute_power(args, freqs) 
    














