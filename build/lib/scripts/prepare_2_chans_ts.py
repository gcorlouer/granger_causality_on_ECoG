#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:23:11 2022

@author: guime
"""


from src.preprocessing_lib import Epocher, EcogReader, parcellation_to_indices
from pathlib import Path
from scipy.io import savemat

import argparse
import numpy as np

#%% Parameters
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs']

# Paths (Change before running. Run from root.)
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
derivatives_path = data_path.joinpath('derivatives')
result_path = cifar_path.joinpath('results')

chans = ['LTo1-LTo2', 'LGRD60-LGRD61']

signal = 'lfp'
# Signal dic
hfa_dic = {'suffix': '_hfb_continuous_raw.fif', 'log_transf': True, 
           'l_freq':0.1, 'decim': 4}
lfp_dic = {'suffix': '_bad_chans_removed_raw.fif', 'log_transf': False,
           'l_freq':1, 'decim': 2}
if signal == 'hfa':
    signal_dic = hfa_dic
elif signal == 'lfp':
    signal_dic = lfp_dic

# Parser arguments
parser = argparse.ArgumentParser()
# Dataset parameters
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--sfeq", type=float, default=500.0)
parser.add_argument("--stage", type=str, default='preprocessed')
parser.add_argument("--preprocessed_suffix", type=str, default= signal_dic['suffix'])
parser.add_argument("--epoch", type=bool, default=False)
parser.add_argument("--channels", type=str, default='visual_channels.csv')

# Epoching parameters
parser.add_argument("--condition", type=str, default='Stim') 
parser.add_argument("--t_prestim", type=float, default=-0.5)
parser.add_argument("--t_postim", type=float, default=1.5)
parser.add_argument("--baseline", default=None) # No baseline from MNE
parser.add_argument("--preload", default=True)
parser.add_argument("--tmin_baseline", type=float, default=-0.5)
parser.add_argument("--tmax_baseline", type=float, default=0)

# Wether to log transform the data
parser.add_argument("--log_transf", type=bool, default=signal_dic['log_transf'])
# Mode to rescale data (mean, logratio, zratio)
parser.add_argument("--mode", type=str, default='logratio')
# Pick visual chan
parser.add_argument("--pick_visual", type=bool, default=True)
# Create category specific time series
parser.add_argument("--l_freq", type=float, default=signal_dic['l_freq'])
parser.add_argument("--decim", type=float, default=signal_dic['decim'])
parser.add_argument("--tmin_crop", type=float, default=0.2)
parser.add_argument("--tmax_crop", type=float, default=1.5)
parser.add_argument("--matlab", type=bool, default=True)

args = parser.parse_args()

#%% 

def input_fname(subject, signal='hfa'):
    fname = subject + '_condition_' + 'two_chans_' + signal +'.mat'
    return fname

def prepare_ts(raw, subject='DiAs', stage='preprocessed', matlab = True,
                     preprocessed_suffix='_hfb_continuous_raw.fif', decim=2,
                     l_freq = 0.1,
                     epoch=False, t_prestim=-0.5, t_postim=1.75, tmin_baseline = -0.5,
                     tmax_baseline = 0, tmin_crop=0, tmax_crop=1, condition='Face',
                     mode = 'logratio', log_transf=True):
    """
    Return category-specific time series as a dictionary 
    """
    conditions = ['Rest', 'Face', 'Place', 'baseline']
    ts = dict.fromkeys(conditions, [])
    for condition in conditions:
        # Epoch visually responsive HFA
        if condition == 'baseline':
            # Return prestimulus baseline
            epocher = Epocher(condition='Stim', t_prestim=t_prestim, t_postim = t_postim, 
                            baseline=None, preload=True, tmin_baseline=tmin_baseline, 
                            tmax_baseline=tmax_baseline, mode=mode)
            if log_transf == True:
                epoch = epocher.log_epoch(raw)
            else:
                epoch = epocher.epoch(raw)
                # Downsample by factor of 2 and check decimation
            epoch = epoch.copy().crop(tmin = -0.5, tmax=0)
            # Low pass filter
            epoch = epoch.copy().filter(l_freq=l_freq, h_freq=None)
            epoch = epoch.copy().decimate(decim)
        else:
            # Return condition specific epochs
            epocher = Epocher(condition=condition, t_prestim=t_prestim, t_postim = t_postim, 
                                baseline=None, preload=True, tmin_baseline=tmin_baseline, 
                                tmax_baseline=tmax_baseline, mode=mode)
            #Epoch condition specific hfb and log transform to approach Gaussian
            if log_transf == True:
                epoch = epocher.log_epoch(raw)
            else:
                epoch = epocher.epoch(raw)
            
            epoch = epoch.copy().crop(tmin = tmin_crop, tmax=tmax_crop)
            # Low pass filter
            epoch = epoch.copy().filter(l_freq=l_freq, h_freq=None)
                # Downsample by factor of 2 and check decimation
            epoch = epoch.copy().decimate(decim)
            time = epoch.times

        # Prerpare time series for MVGC
        X = epoch.copy().get_data()
        (N, n, m) = X.shape
        X = np.transpose(X, (1,2,0))
        ts[condition] = X
        # Add category specific channels indices to dictionary
        indices = {'R': 2, 'F': 1}
        ts['indices']= indices
        
        # Add time
        ts['time'] = time
        
        # Add subject
        ts['subject'] = subject
        
        # Add sampling frequency
        ts['sfreq'] = 500/decim
    
    return ts


#%% Prepare condition for subject DiAs with appropriate channels
# Read continuous HFA
reader = EcogReader(data_path, subject=args.subject, stage=args.stage,
                     preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                     epoch=False)
raw = reader.read_ecog()
# Read visually responsive channels
df_visual = reader.read_channels_info(fname=args.channels)
visual_chans = df_visual['chan_name'].to_list()
raw = raw.pick_channels(chans)
ts = prepare_ts(raw, subject=args.subject, stage=args.stage, matlab = args.matlab,
                    preprocessed_suffix=args.preprocessed_suffix, decim=args.decim,
                    epoch=args.epoch, t_prestim=args.t_prestim, t_postim=args.t_postim, 
                    tmin_baseline = args.tmin_baseline, tmax_baseline = args.tmax_baseline,
                    tmin_crop=args.tmin_crop, tmax_crop=args.tmax_crop, 
                    mode = args.mode, log_transf=args.log_transf)

    #%% Save condition ts as mat file
# Save file as  _condition_visual_ts.mat or _condition_ts.mat
fname = input_fname(args.subject, signal=signal)
fpath = result_path.joinpath(fname)
print(f"\n Saving in {fpath}")
savemat(fpath, ts)
print(f"\n Sampling rate is {500/args.decim}Hz")
print(f"\n Stimulus is during {args.tmin_crop} and {args.tmax_crop}s")

#%% To verify position of indices

# import matplotlib.pyplot as plt

# reader = EcogReader(data_path, subject=args.subject, stage=args.stage,
#                      preprocessed_suffix=args.preprocessed_suffix, preload=True, 
#                      epoch=False)
# raw = reader.read_ecog()
# # Read visually responsive channels
# df_visual = reader.read_channels_info(fname=args.channels)
# visual_chans = df_visual['chan_name'].to_list()
# raw = raw.pick_channels(chans)
# condition = 'Face'
# epocher = Epocher(condition=condition, t_prestim=args.t_prestim, t_postim = args.t_postim, 
#                             baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
#                             tmax_baseline=args.tmax_baseline, mode=args.mode)
# epoch = epocher.epoch(raw)
# epoch = epoch.copy().filter(l_freq=65, h_freq=120,
#                                      phase='minimum', filter_length='auto',
#                                      l_trans_bandwidth= 10, 
#                                      h_trans_bandwidth= 10,
#                                          fir_window='blackman')
# epoch = epoch.copy().apply_hilbert(envelope=True)
# X = epoch.copy().get_data()

# populations = []
# chans = epoch.info['ch_names']
# for chan in chans:
#     populations.append(df_visual['group'].loc[df_visual['chan_name']==chan].to_list()[0])
# indices = {'R': 2, 'F': 1}

# times = epoch.times
# evok = np.average(X, axis=0)
# evok = np.abs(evok)
# for i in range(2):
#     plt.plot(times, evok[i,:], label=populations[i])
# plt.legend()