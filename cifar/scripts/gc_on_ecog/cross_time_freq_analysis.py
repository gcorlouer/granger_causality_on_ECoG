#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:29:20 2022
In this script we run cross subjects time frequency analysis
@author: guime
"""

#%%
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

def compute_group_power(args, freqs, group='F', condition = 'Face', l_freq = 0.01,
                 baseline=(-0.5, 0), mode = 'zscore'):
    """Compute power of visually responsive in a specific group epochs 
    for time freq analysis"""
    # Read ECoG
    reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                         preprocessed_suffix=args.preprocessed_suffix,
                         epoch=args.epoch)
    raw = reader.read_ecog()
    # Read visually responsive channels
    df_visual = reader.read_channels_info(fname='visual_channels.csv')
    visual_chans = df_visual['chan_name'].to_list()
    raw = raw.pick_channels(visual_chans)
    # Get visual channels from functional group
    indices = visual_indices(args)
    group_indices = indices[group]
    group_chans = [visual_chans[i] for i in group_indices]
    print(f'\n {group} channels are {group_chans} \n')
    # Epoch raw ECoG
    epocher = Epocher(condition=condition, t_prestim=args.t_prestim, t_postim = args.t_postim, 
                             baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                             tmax_baseline=args.tmax_baseline, mode=args.mode)
    epochs = epocher.epoch(raw)
    # High pass filter
    epochs = epochs.filter(l_freq=l_freq, h_freq=None)
    # Downsample
    epochs = epochs.decimate(args.decim)
    times = epochs.times
    # Pick channels
    epochs = epochs.pick(group_chans)
    # Compute time frequency with Morlet wavelet 
    n_cycles = freqs/2
    power = tfr_morlet(epochs, freqs, n_cycles, return_itc=False)
    # Apply baseline correction
    baseline = (args.tmin_baseline, args.tmax_baseline)
    print(f"\n Computing group power from morlet wavelet: rescale with {mode}")
    print(f"\n Condition is {condition}\n")
    power.apply_baseline(baseline=baseline, mode=mode)
    power = power.data
    power = np.average(power,axis=0)
    return power, times
    
#%% Parameters
conditions = ['Rest', 'Face', 'Place']
groups = ['R','O','F']
ngroup = len(groups)
ncdt = len(conditions)
# Frequency space
fn = 500/args.decim #Nyquist
nfreqs = 2**9
fmin = 0.5
fmax = fn/2 - 1
fres = (fmax + fmin -1)/nfreqs
freqs = np.arange(fmin,fmax,fres)
n_cycles = freqs/2
# Baseline correction 
mode = 'zscore'
baseline = (-0.5, 0)
#%% For a given subject, run time frequency analysis across groups and 
# conditions
power_dict = {}
for group in groups:
    for condition in conditions:
        power, time = compute_group_power(args, freqs, group=group, 
        condition=condition, l_freq=0.01, baseline=baseline, mode=mode)
        power_dict['condition']['group'] = power

















