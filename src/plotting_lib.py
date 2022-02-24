#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:55:11 2022
This script contain plotting functions for the project
@author: guime
"""

import mne 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.preprocessing_lib import EcogReader, Epocher
from pathlib import Path


#%% Style parameters

plt.style.use('ggplot')
fig_width = 16  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'lines.linewidth': 1.5,
          'axes.labelsize': 12,
          'axes.titlesize': 8,
          'font.size': 12,
          'legend.fontsize': 8,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

#%% Preprocessing plots

#def plot_stimuli_presentation():

def plot_narrow_broadband(args, fpath, fname = 'DiAs_narrow_broadband_stim.pdf', 
                          chan = ['LTo1-LTo2'], tmin=500, tmax=506):
    """
    Plot the narrowband envelope and broadband hfa between (tmin, tmax) for 
    one subject and one reperesentative channel
    """
    # Read EcoG data
    reader = EcogReader(args.data_path, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                 epoch=False)
    ecog = reader.read_ecog()

    #Extract narrowband envelope
    raw_band = ecog.copy().filter(l_freq=args.l_freq, h_freq=args.l_freq+args.band_size,
                                         phase=args.phase, filter_length=args.filter_length,
                                         l_trans_bandwidth= args.l_trans_bandwidth, 
                                         h_trans_bandwidth= args.h_trans_bandwidth,
                                             fir_window=args.fir_window) 
    envelope = raw_band.copy().apply_hilbert(envelope=True)
    # Filtered signal
    X_filt = raw_band.copy().pick_channels(chan).crop(tmin=tmin, tmax=tmax).get_data()
    X_filt = X_filt[0,:]
    # muV
    X_filt =X_filt*1e6
    print(f"Filtered signal shape is {X_filt.shape}\n")
    # Signal narrow band envelope
    narrow_envelope = envelope.copy().pick_channels(chan).crop(tmin=tmin, tmax=tmax).get_data()
    narrow_envelope = narrow_envelope[0,:]
    narrow_envelope = narrow_envelope*1e6
    print(f"narrow band envelope shape is {narrow_envelope.shape}\n")
    # Time axis
    time = raw_band.copy().pick_channels(chan).crop(tmin=tmin, tmax=tmax).times
    
    #Rread hfb
    reader = EcogReader(args.data_path, stage=args.stage,
                     preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                     epoch=False)
    
    ecog = reader.read_ecog()
    # Extract single channel broadband envelope
    hfb = reader.read_ecog()
    hfb = hfb.copy().pick(chan)
    hfb = hfb.copy().crop(tmin=tmin, tmax=tmax)
    broadband_envelope = hfb.copy().get_data()
    broadband_envelope = broadband_envelope[0,:]
    broadband_envelope = broadband_envelope*1e6
    print(f"broadband_envelope shape is {narrow_envelope.shape}\n")
    # Plot narrowband and broadband envelope
    f, ax = plt.subplots(2,1)
    # Plot narrow band
    ax[0].plot(time, X_filt, label='ECoG')
    ax[0].plot(time, narrow_envelope, label='Narrow band envelope')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Signal (V)')
    ax[0].legend(bbox_to_anchor=(0.3,1.02), loc='lower left')
    ax[0].set_title('a)', loc='left')
    # Plot broadband
    ax[1].plot(time, broadband_envelope)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Amplitude (muV)')
    ax[1].set_title('b)', loc='left')
    plt.tight_layout()
    # Save figure
    fpath = fpath.joinpath(fname)
    plt.savefig(fpath)

# Plot log vs non log trials
    
def plot_log_trial(args, fpath, fname = 'DiAs_log_trial.eps', 
                          chan = ['LTo1-LTo2'], itrial=2, nbins=50):
    
    #
    reader = EcogReader(args.data_path, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                 epoch=False)
    # Read visually responsive channels
    hfa = reader.read_ecog()
    epocher = Epocher(condition='Face', t_prestim=args.t_prestim, t_postim = args.t_postim, 
                                 baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                                 tmax_baseline=args.tmax_baseline, mode=args.mode)
    # Epoch HFA
    epoch = epocher.epoch(hfa)
    # Downsample by factor of 2 and check decimation
    epoch = epoch.copy().decimate(args.decim)
    time = epoch.times
    # Get epoch data
    epoch = epoch.copy().pick(chan)
    X = epoch.copy().get_data()
    # Results in muV
    X = X*1e6
    # Trial
    trial = X[itrial,0, :]
    # Log epoch HFA
    l_epoch = epocher.log_epoch(hfa)
    l_epoch = l_epoch.copy().decimate(args.decim)
    l_epoch = l_epoch.copy().pick(chan)
    l_X = l_epoch.copy().get_data()
    # Log trial
    l_trial = l_X[itrial,0, :]
    # Get array for histogram
    x =  np.ndarray.flatten(X)
    l_x = np.ndarray.flatten(l_X)
    amax = np.amax(x)
    # Plot log vs non log trial and histogram
    f, ax = plt.subplots(2,2)
    # Plot trial
    ax[0,0].plot(time, trial)
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel('HFA (muV)')
    # Plot log trial
    ax[0,1].plot(time, l_trial)
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel('Log HFA (muV)')
    # Plot hfa histogram
    sns.histplot(x, stat = 'probability', bins=nbins, kde=True, ax=ax[1,0])
    ax[1,0].set_xlabel('HFA (muV)')
    ax[1,0].set_ylabel('Probability')
    ax[1,0].set_xlim(left=0, right=amax)

    # Plot log hfa histogram
    sns.histplot(l_x, stat = 'probability', bins=nbins, kde=True, ax=ax[1,1])
    ax[1,1].set_xlabel('Log HFA')
    ax[1,1].set_ylabel('Probability')
    #ax[1,1].set_xlim(left=0, right=amax)
    plt.tight_layout()
    # Save figure
    fpath = fpath.joinpath(fname)
    plt.savefig(fpath)
#%%








































