#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:12:56 2022
In this script we compute the power spectral density of the HFA and ECoG time
series from condition specific time series. After computation we plot the psd
of the HFA.
@author: guime
"""

import mne 
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing_lib import prepare_condition_scaled_ts, EcogReader, parcellation_to_indices
from src.input_config import args
from mne.time_frequency import psd_array_multitaper, psd_array_welch, csd_array_multitaper


#%% functions

def ts_to_psd(ts, indices, condition='Face', group='F',  
              sfreq = 250, fmax=100, bandwidth=6):
    """
    Compute spectral density average over functional groups 
    from condition specific time series
    Input: ts (dict), indices (list)
    output: s (array), freqs (array)
    """
    # Take condition specific time series
    x = ts[condition]
    # Reshapre time series in suitable MNE format
    (n,m,N) = x.shape
    x = np.transpose(x, (2,0,1))
    # Compute spectral density with multitaper method
    psds, freqs = psd_array_multitaper(x, sfreq = sfreq, fmax=fmax, bandwidth=bandwidth)
    # Average over functional group 
    idx = indices[group]
    # Return condition specifc psd
    s = psds[0]
    # Average over functional group
    s = np.mean(s[idx,:],0)
    return s, freqs

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

def average_psd(args, condition='Face', group = 'F', fmax=50, bandwidth=8,
                matlab = False):
    """
    Compute average of psd over subjects in each condition and functional group
    Input: 
        - args (parser): arguments to prepare condition specific ts
        - condition (string): condition on which to plot psd
        - fmax (float): Max frequency for psd
        - bandwitdth (float): control the number of DPSS tapers
    Output:
        - s (array): condition specific psd of a functional group averaged over subjects 
        - freqs (array): array of frequencies
    """
    nsub = len(args.cohort)
    s_average = [0]*nsub
    # Loop over subjects
    for i, subject in enumerate(args.cohort):
        # Prepare condition specific scaled HFA
        ts = prepare_condition_scaled_ts(args.data_path, subject=subject, stage=args.stage, matlab = matlab,
                     preprocessed_suffix=args.preprocessed_suffix, decim=args.decim,
                     epoch=args.epoch, t_prestim=args.t_prestim, t_postim=args.t_postim, 
                     tmin_baseline = args.tmin_baseline, tmax_baseline = args.tmax_baseline,
                     tmin_crop=args.tmin_crop, tmax_crop=args.tmax_crop)
        # Return indices of functional group
        indices = visual_indices(args, subject=subject)
        s, freqs = ts_to_psd(ts, indices, condition=condition, group =group,
                             fmax=fmax, bandwidth=bandwidth)
        s_average[i] = s
    #     
    s = np.stack(s_average, axis=-1)
    s = np.average(s, axis =-1)
    return s, freqs

def compute_freq_loss(s, freqs, frac=0.95) :
    """
    Compute frequency after which psd lose more than 95% of its value
    """
    smax = np.amax(s)
    s_neglect = smax-frac*smax
    d = s - s_neglect
    d = np.abs(d)
    dmin = np.min(d)
    idx_neglect = np.where(d==dmin)[0][0]
    freq_loss = freqs[idx_neglect]
    return freq_loss


#%% Plot psd
fmax = 100
bandwidth = 8
conditions = ['Rest', 'Face', 'Place']
visual_group = ['R','F']
ng = len(visual_group)
frac = 0.95 #Fraction of PSD loss
f, ax = plt.subplots(2,1, figsize=(5,6))
for i, group in enumerate(visual_group):
    for condition in conditions:
        s, freqs = average_psd(args, condition=condition, group=group, 
                               fmax=fmax, bandwidth=bandwidth)
        freq_loss = compute_freq_loss(s, freqs, frac=frac)
        ax[i].plot(freqs, s, label=condition)
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel('frequency (Hz)')
        ax[i].set_ylabel(f'PSD {group}')
    ax[i].axvline(x=freq_loss, color = 'k')
plt.tight_layout()
plt.legend()
#%%

##%% Test individual functions
##%matplotlib qt
#subject = 'DiAs'
#
#ts = prepare_condition_scaled_ts(args.data_path, subject=subject, stage=stage, matlab = matlab,
#                     preprocessed_suffix=preprocessed_suffix, decim=decim,
#                     epoch=epoch, t_prestim=args.t_prestim, t_postim=args.t_postim, 
#                     tmin_baseline = args.tmin_baseline, tmax_baseline = args.tmax_baseline,
#                     tmin_crop=args.tmin_crop, tmax_crop=args.tmax_crop)
#
#indices = visual_indices(args, subject='DiAs')
#
#s, freqs = ts_to_psd(ts, indices, condition='Face', group='F',  
#              sfreq = 250, fmax=100, bandwidth=8)
#plt.plot(freqs, s)
#plt.yscale('log')
#plt.xscale('log')
##%% Test average function
#
#s, freqs = average_psd(args, condition='Face', group = 'F', fmax=100, bandwidth=10,
#                matlab = False)
#plt.plot(freqs, s)
#plt.yscale('log')
#plt.xscale('log')
