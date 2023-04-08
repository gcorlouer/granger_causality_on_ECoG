#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this script we plot the high frequency narrow and broad envelope.
@author: guime
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

from src.preprocessing_lib import EcogReader, Epocher
from scipy.stats import sem
from pathlib import Path


#%%

# Parameters
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs']

# Paths (Change before running. Run from root.)
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
derivatives_path = data_path.joinpath('derivatives')
result_path = cifar_path.joinpath('results')

parser = argparse.ArgumentParser()

# Dataset parameters
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--sfeq", type=float, default=500.0)
parser.add_argument("--stage", type=str, default='preprocessed')
parser.add_argument("--preprocessed_suffix", type=str, default= '_bad_chans_removed_raw.fif')
parser.add_argument("--signal", type=str, default= 'lfp') # correspond to preprocessed_suffix
parser.add_argument("--epoch", type=bool, default=False)
parser.add_argument("--channels", type=str, default='visual_channels.csv')

#%z Filtering parameters

parser.add_argument("--l_freq", type=float, default=70.0)
parser.add_argument("--band_size", type=float, default=20.0)
parser.add_argument("--nband", type=float, default=5)
parser.add_argument("--l_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--h_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--filter_length", type=str, default='auto')
parser.add_argument("--phase", type=str, default='minimum')
parser.add_argument("--fir_window", type=str, default='blackman')


# Epoching parameters
parser.add_argument("--condition", type=str, default='Stim') 
parser.add_argument("--t_prestim", type=float, default=-0.5)
parser.add_argument("--t_postim", type=float, default=1.5)
parser.add_argument("--baseline", default=None) # No baseline from MNE
parser.add_argument("--preload", default=True)
parser.add_argument("--tmin_baseline", type=float, default=-0.5)
parser.add_argument("--tmax_baseline", type=float, default=0)

# Wether to log transform the data
parser.add_argument("--log_transf", type=bool, default=False)
# Mode to rescale data (mean, logratio, zratio)
parser.add_argument("--mode", type=str, default='logratio')
# Pick visual chan
parser.add_argument("--pick_visual", type=bool, default=True)

args = parser.parse_args()

#%% 
chan = ['LTo1-LTo2']
reader = EcogReader(data_path, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                 epoch=False)

ecog = reader.read_ecog()


#%% Extract narrowband HFB

(tmin, tmax) = (100, 102)
raw_band = ecog.copy().filter(l_freq=args.l_freq, h_freq=args.l_freq+args.band_size,
                                     phase=args.phase, filter_length=args.filter_length,
                                     l_trans_bandwidth= args.l_trans_bandwidth, 
                                     h_trans_bandwidth= args.h_trans_bandwidth,
                                         fir_window=args.fir_window)

envelope = raw_band.copy().apply_hilbert(envelope=True)

X = raw_band.copy().pick_channels(chan).crop(tmin=tmin, tmax=tmax).get_data()
H = envelope.copy().pick_channels(chan).crop(tmin=tmin, tmax=tmax).get_data()

time = raw_band.copy().pick_channels(chan).crop(tmin=tmin, tmax=tmax).times
#%% Plot narrow band hfb

#%matplotlib qt
linewidth=1.5
sns.set(font_scale=2)
plt.figure(figsize=(10,8),dpi=80)
plt.plot(time, X[0,:], label='ECoG', linewidth=linewidth)
plt.plot(time, H[0,:], label='narrow-band envelope', linewidth=linewidth)
plt.xlabel('Time (s)')
plt.ylabel('Signal (V)')
plt.legend()

#%% Rread hfb

reader = EcogReader(args.data_path, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                 epoch=False)

ecog = reader.read_ecog()

hfb = reader.read_ecog()
hfb = hfb.copy().pick(chan)
hfb = hfb.copy().crop(tmin=500, tmax=506)

#%% Plot broadband HFA
sns.set(font_scale=2)
time = hfb.times
plt.figure(figsize=(10,8),dpi=80)
X = hfb.copy().get_data()
X = X[0,:]
plt.plot(time, X, linewidth=linewidth)
plt.xlabel('Time (s)')
plt.ylabel('HFA (V)')

#%% Plot representative trial

itrial = 2
# Read ECoG
reader = EcogReader(args.data_path, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                 epoch=False)

# Read visually responsive channels
df= reader.read_channels_info(fname='visual_channels.csv')
latency = df['latency'].loc[df['chan_name']==chan[0]].tolist()
latency = latency[0]*1e-3

hfa = reader.read_ecog()

# Epoch HFA
epocher = Epocher(condition='Face', t_prestim=args.t_prestim, t_postim = args.t_postim, 
                             baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                             tmax_baseline=args.tmax_baseline, mode=args.mode)
epoch = epocher.epoch(hfa)

# Downsample by factor of 2 and check decimation
epoch = epoch.copy().decimate(args.decim)
time = epoch.times

# Get data
epoch = epoch.copy().pick(chan)
X = epoch.copy().get_data()
trial = X[itrial,0, :]

# Get evoked data

evok = np.mean(X, axis=0)
evok = evok[0,:]
sm = sem(X, axis=0)
sm = sm[0,:]
up_ci = evok + 1.96*sm
down_ci = evok - 1.96*sm

# Get histogram of prestimulus

prestim = epoch.copy().crop(tmin=args.tmin_prestim, tmax=args.tmax_prestim)
prestim = prestim.get_data()
prestim = np.ndarray.flatten(prestim)
baseline = np.mean(prestim)
# Get histogram of postimulus

postim = epoch.copy().crop(tmin=args.tmin_postim, tmax=args.tmax_postim)
postim = postim.get_data()
postim = np.ndarray.flatten(postim)
amax = np.amax(postim)
# Plot trial, evoked response and pre/post stimulus histogram
sns.set(font_scale=2)
f, ax = plt.subplots(2,2, figsize=(10,8),dpi=80)

# Plot representative trial
ax[0,0].plot(time, trial)
ax[0,0].set_xlabel('Time (s)')
ax[0,0].set_ylabel('HFA (V)')
ax[0,0].axvline(x=0, color='k')
ax[0,0].axvline(x=latency, color='r', label='latency response')
ax[0,0].axhline(y=baseline, color='k')
ax[0,0].legend()
# Plot evoked response
ax[1,0].plot(time, evok, color='b')
ax[1,0].fill_between(time, down_ci, up_ci, alpha=0.6)
ax[1,0].set_xlabel('Time (s)')
ax[1,0].set_ylabel('HFA (V)')
ax[1,0].axvline(x=0, color='k')
ax[1,0].axvline(x=latency, color='r', label='latency response')
ax[1,0].axhline(y=baseline, color='k')
ax[1,0].legend()

# Plot postimulus histogram
sns.histplot(postim, stat = 'probability', bins=50, kde=True, ax=ax[0,1])
ax[0,1].set_xlabel('Postimulus Amplitude (V)')
ax[0,1].set_ylabel('Probability')
ax[0,1].set_xlim(left=0, right=amax)

# Plot prestimulus histogram

sns.histplot(prestim, stat = 'probability', bins=50, kde=True, ax=ax[1,1])
ax[1,1].set_xlabel('Prestimulus Amplitude (V)')
ax[1,1].set_ylabel('Probability')
ax[1,1].set_xlim(left=0, right=amax)

plt.legend()
#%%
# Plot trial
sns.set(font_scale=2)
plt.figure(figsize=(10,8),dpi=80)
plt.plot(time, X)
plt.xlabel('Time (s)')
plt.ylabel('HFA (V)')
plt.axvline(x=0, color='k')
plt.axvline(x=latency, color='r', label='latency response')
