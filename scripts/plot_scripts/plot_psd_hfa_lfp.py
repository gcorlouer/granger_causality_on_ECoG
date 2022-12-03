#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 18:18:04 2022
Plot psd of HFA and LFP
@author: guime
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

from src.preprocessing_lib import prepare_condition_ts
from pathlib import Path
from scipy.stats import sem
from mne.time_frequency import psd_array_multitaper

#%% Plot parameters

plt.style.use('ggplot')
fig_width = 22  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 12
tick_size = 8
params = {'backend': 'ps',
          'lines.linewidth': 1.2,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'font.size': label_size,
          'legend.fontsize': tick_size,
          'xtick.labelsize': tick_size,
          'ytick.labelsize': tick_size,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

#%%

cohort = ['AnRa',  'ArLa', 'DiAs']
# Path to source data, derivatives and results. Enter your own path in local machine
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
derivatives_path = data_path.joinpath('derivatives')
result_path = cifar_path.joinpath('results')
fig_path = cifar_path.joinpath('results/figures')

signal = 'hfa'
# Signal dic
hfa_dic = {'suffix': '_hfb_continuous_raw.fif', 'log_transf': False, 
           'l_freq':0.1, 'decim': 4}
lfp_dic = {'suffix': '_bad_chans_removed_raw.fif', 'log_transf': False,
           'l_freq':1, 'decim': 2}
if signal == 'hfa':
    signal_dic = hfa_dic
elif signal == 'lfp':
    signal_dic = lfp_dic

parser = argparse.ArgumentParser()

# Dataset parameters
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--sfeq", type=float, default=500.0)
parser.add_argument("--stage", type=str, default='preprocessed')
parser.add_argument("--preprocessed_suffix", type=str, default= signal_dic['suffix'])
parser.add_argument("--signal", type=str, default= signal) # correspond to preprocessed_suffix
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
parser.add_argument("--l_freq", type=float, default=0.1)
parser.add_argument("--decim", type=float, default=signal_dic['decim'])
parser.add_argument("--tmin_crop", type=float, default=0)
parser.add_argument("--tmax_crop", type=float, default=1.5)
parser.add_argument("--matlab", type=bool, default=False)

args = parser.parse_args()

# Suffix
#'_bad_chans_removed_raw.fif'
# '_hfb_continuous_raw.fif' 
# signal hfa/lfp

#%%

def plot_psd(args, cohort, signal='hfa',fmin=0, fmax=100, bandwidth=2, adaptive=True):
    # Prepare inputs for plotting
    conditions = ['Rest','Face','Place']
    populations = ['R','O','F']
    nsub = len(cohort)
    fig, ax = plt.subplots(3,nsub, sharex=False, sharey=False)
    # Prepare condition ts
    for s, subject in enumerate(cohort):
        ts = prepare_condition_ts(data_path, subject=subject, stage=args.stage, matlab = args.matlab,
                        preprocessed_suffix=args.preprocessed_suffix, decim=args.decim,
                        epoch=args.epoch, t_prestim=args.t_prestim, t_postim=args.t_postim, 
                        tmin_baseline = args.tmin_baseline, tmax_baseline = args.tmax_baseline,
                        tmin_crop=args.tmin_crop, tmax_crop=args.tmax_crop, 
                        mode = args.mode, log_transf=args.log_transf, 
                        pick_visual=args.pick_visual, channels = args.channels)
        sfreq = ts['sfreq']
        time = ts['time']
        # Plot condition ts
        for p, pop in enumerate(populations):
            for c, cdt in enumerate(conditions):
                # Condition specific neural population
                X = ts[cdt]
                pop_idx = ts['indices'][pop]
                X = X[pop_idx,:,:]
                X = np.transpose(X, (0,2,1)) #nchan x ntrial x ntimes
                psd, freqs = psd_array_multitaper(X, sfreq, fmin=fmin, fmax=fmax,
                                                 bandwidth=bandwidth, adaptive=adaptive)
                psd = np.mean(psd, axis=(0,1)) #average over populations
                max_psd = np.max(psd)
                # Plot condition-specific evoked HFA
                ax[p,s].plot(freqs, psd, label=cdt)
                ax[p,0].set_ylabel(f'{pop}, PSD')
                ax[0,s].set_title(f'subject S{s}')
                ax[p,s].set_xscale('log')
                ax[p,s].set_yscale('log')
                # if p<=1:
                #         ax[p,s].set_xticks([]) # (turn off xticks)
                # if s>=1:
                #         ax[p,s].set_yticks([]) # (turn off xticks)
                handles, labels = ax[p,s].get_legend_handles_labels()
                #xticks = [fmin, (fmax+fmin)/2 ,fmax]
                #ax[2,s].get_xticks()
                #ax[2,s].set_xticks(xticks)
                #ax[2,s].set_xticklabels(xticks) 
                ax[p,s].set_xlabel("Frequency (Hz)")
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f'PSD of {signal}', )
    plt.show()

#%%

fmin = 0
fmax = 100
bandwidth = 2
plot_psd(args, cohort, signal=signal, fmin=fmin, fmax=fmax, bandwidth=bandwidth, adaptive=True)
figpath = Path('~','thesis','overleaf_project', 'figures','supplementary_methods_figures').expanduser()
fname = signal + '_psd.pdf'
figpath = figpath.joinpath(fname)
plt.savefig(figpath)