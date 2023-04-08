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
import argparse

from src.preprocessing_lib import prepare_condition_scaled_ts, EcogReader, parcellation_to_indices
from mne.time_frequency import psd_array_multitaper, psd_array_welch, csd_array_multitaper
from scipy.stats import sem
from pathlib import Path


#%% Parameters
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs']

# Paths (Change before running. Run from root.)
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
derivatives_path = data_path.joinpath('derivatives')
result_path = cifar_path.joinpath('results')

parser = argparse.ArgumentParser()


# Dataset parameters
parser.add_argument("--cohort", type=str, default=cohort)
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--sfeq", type=float, default=500.0)
parser.add_argument("--stage", type=str, default='preprocessed')
parser.add_argument("--preprocessed_suffix", type=str, default= '_hfb_continuous_raw.fif')
parser.add_argument("--epoch", type=bool, default=False)
parser.add_argument("--channels", type=str, default='visual_channels.csv')

# Epoching parameters
parser.add_argument("--condition", type=str, default='Stim') 
parser.add_argument("--t_prestim", type=float, default=-0.5)
parser.add_argument("--t_postim", type=float, default=1.75)
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
# Create category specific time series
parser.add_argument("--decim", type=float, default=2)
parser.add_argument("--tmin_crop", type=float, default=0)
parser.add_argument("--tmax_crop", type=float, default=1.5)
parser.add_argument("--matlab", type=bool, default=True)

args = parser.parse_args()


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
    reader = EcogReader(data_path, subject=subject)
    df_visual = reader.read_channels_info(fname=args.channels)
    # Return indices of functional groups from visual channel dataframe
    indices = parcellation_to_indices(df_visual, parcellation='group', matlab=False)
    return indices 

def average_hfa(args, cohort, condition='Face', group = 'F', matlab = False):
    """
    Compute grand average of hfa accross subjects in each conditions and functional
    group.
    Input: 
        - args (parser): arguments to prepare condition specific ts
        - condition (string): condition on which to plot psd
        - fmax (float): Max frequency for psd
        - bandwitdth (float): control the number of DPSS tapers
    Output:
        - hfa (array): condition specific hfa of a functional group averaged over subjects 
    """
    nsub = len(cohort)
    hfa_population = [0]*nsub
    # Loop over subjects
    for i, subject in enumerate(cohort):
        # Prepare condition specific scaled HFA
        ts = prepare_condition_scaled_ts(data_path, subject=subject, stage=args.stage, matlab = matlab,
                     preprocessed_suffix=args.preprocessed_suffix, decim=args.decim,
                     epoch=args.epoch, t_prestim=args.t_prestim, t_postim=args.t_postim, 
                     tmin_baseline = args.tmin_baseline, tmax_baseline = args.tmax_baseline,
                     tmin_crop=args.tmin_crop, tmax_crop=args.tmax_crop, mode=args.mode)
        # Return time array of trial
        time =ts['time']
        # Return baseline
        baseline = ts['baseline']
        baseline = np.average(baseline)
        # Return indices of functional group
        indices = visual_indices(args, subject=subject)
        # Take condition specific time series
        x = ts[condition]
        # Average over functional group 
        idx = indices[group]
        # Average over populations 
        hfa_population[i] = np.average(x[idx,:,:],axis=0)
        
    # Cross subjects array
    hfa_population = np.stack(hfa_population, axis=-1) # shape is npopxmxNxnsub
    (m,N,nsub) = hfa_population.shape
    # Average across subjects and trials to get grand evoked hfa
    evoked_hfa = np.average(hfa_population, axis=(1,2)) 
    # Reshape cross subject population hfa to compute standard error of mean
    hfa_population = np.transpose(hfa_population, axes=(1,2,0))
    hfa_population = np.reshape(hfa_population, newshape=(N*nsub,m)) 
    # Compute standard error of mean           
    grand_sem = sem(hfa_population, axis =0)
    return evoked_hfa, grand_sem, time, baseline
    
def average_psd(args, data_path, condition='Face', group = 'F', fmax=50, bandwidth=8,
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
    nsub = len(cohort)
    s_average = [0]*nsub
    # Loop over subjects
    for i, subject in enumerate(cohort):
        # Prepare condition specific scaled HFA
        ts = prepare_condition_scaled_ts(data_path, subject=subject, stage=args.stage, matlab = matlab,
                     preprocessed_suffix=args.preprocessed_suffix, decim=args.decim,
                     epoch=args.epoch, t_prestim=args.t_prestim, t_postim=args.t_postim, 
                     tmin_baseline = args.tmin_baseline, tmax_baseline = args.tmax_baseline,
                     tmin_crop=args.tmin_crop, tmax_crop=args.tmax_crop, mode=args.mode)
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

#%% Plot average psd and hfa

fmax = 100
bandwidth = 8
conditions = ['Rest', 'Face', 'Place']
visual_group = ['R','F']
colors = ['b', 'r', 'brown']
ng = len(visual_group)


fig_name = 'cross_psd_hfa.jpg'
fig_dir = Path('~','thesis', 'overleaf_project', 'figures', 'results_figures').expanduser()
fig_path = fig_dir.joinpath(fig_name)
frac = 0.95 #Fraction of PSD loss
f, ax = plt.subplots(2,2, figsize=(5,6))
for i, group in enumerate(visual_group):
    for c, condition in enumerate(conditions):
        color = colors[c]
        # Plot grand averaged psd
        s, freqs = average_psd(args, data_path, condition=condition, group=group, 
                               fmax=fmax, bandwidth=bandwidth)
        #freq_loss = compute_freq_loss(s, freqs, frac=frac)
        ax[i,0].plot(freqs, s, label=condition, color=color)
        ax[i,0].set_xscale('log')
        ax[i,0].set_yscale('log')
        ax[i,0].set_xlabel('frequency (Hz)')
        ax[i,0].set_ylabel(f'PSD {group} channels')
        # Plot grand average HFA
        evoked_hfa, sem_hfa, time, baseline = average_hfa(args, cohort, condition=condition, group = group)
        up_ci = evoked_hfa + 1.96*sem_hfa
        low_ci = evoked_hfa - 1.96*sem_hfa
        ax[i,1].plot(time, evoked_hfa, label=condition, color=color)
        ax[i,1].fill_between(time, low_ci, up_ci, alpha=0.3, color=color)
        ax[i,1].axvline(x=0, color ='k')
        ax[i,1].axhline(y=baseline, color='k')
        ax[i,1].set_ylabel(f'Evoked HFA {group} channels (dB)')
        ax[i,1].set_xlabel('Time (s)')
    #ax[i,0].axvline(x=freq_loss, color = 'k')
    
plt.tight_layout()
plt.legend()

plt.savefig(fig_path)


