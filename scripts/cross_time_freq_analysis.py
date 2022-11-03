#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:29:20 2022
In this script we run cross subjects time frequency analysis
@author: guime
"""

#%%
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
# Import classes
from src.preprocessing_lib import EcogReader, Epocher
# Import functions
from src.preprocessing_lib import visual_indices
from mne.time_frequency import tfr_morlet

#%% Parameters
cohort = ['AnRa',  'ArLa', 'DiAs']
conditions = ["Rest", "Face", "Place"]
groups = ["R", "O", "F"]
ngroup = len(groups)
ncdt = len(conditions)
power_dict = {"subject": [], "condition": [], "group": [], "power": [],
              "time": [], "freqs": []}
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
derivatives_path = cifar_path.joinpath('derivatives')
result_path = cifar_path.joinpath('results')
# Command arguments
parser = argparse.ArgumentParser()
# Path
parser.add_argument("--data_path", type=list, default=data_path)
parser.add_argument("--derivatives_path", type=list, default=derivatives_path)
parser.add_argument("--result_path", type=list, default=result_path)
parser.add_argument("--fig_path", type=list, default=result_path)
# Dataset
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--sfreq", type=float, default=500.0)
parser.add_argument("--stage", type=str, default='preprocessed')
parser.add_argument("--preprocessed_suffix", type=str, default= '_bad_chans_removed_raw.fif')
parser.add_argument("--epoch", type=bool, default=False)
parser.add_argument("--channels", type=str, default='visual_channels.csv')

# Frequency space
parser.add_argument("--decim", type=float, default=2)
parser.add_argument("--nfreqs", type=float, default=2**9) # 2**9 for more resolution
parser.add_argument("--fmin", type=float, default=1)
parser.add_argument("--l_freq", type=float, default=1)
# Epoching parameters
parser.add_argument("--condition", type=str, default='Stim') 
parser.add_argument("--t_prestim", type=float, default=0)
parser.add_argument("--t_postim", type=float, default=0.1)
parser.add_argument("--baseline", default=None) # No baseline from MNE
parser.add_argument("--preload", default=True)
parser.add_argument("--tmin_baseline", type=float, default=-0.4)
parser.add_argument("--tmax_baseline", type=float, default=0)
# Wether to log transform the data
parser.add_argument("--log_transf", type=bool, default=False)
# Mode to baseline rescale data (mean, logratio, zratio)
parser.add_argument("--mode", type=str, default='mean')
# Baseline correction
# Do  not confuse with mode and baseline of epoching
parser.add_argument("--tf_mode", type=str, default="zscore")
parser.add_argument("--tf_tmin_baseline", type=float, default=-0.5)
parser.add_argument("--tf_tmax_baseline", type=float, default=0)

args = parser.parse_args()

#%% Functions

def compute_group_power(args, subject='DiAs', group='R', 
                        condition='Face'):
    """
    Compute power of visually responsive in a specific group epochs 
    for time freq analysis
    Input: 
        args: arguments from input_config
        tf_args: arguments to run  cross_time_freq_analysis
        subject: subject name
        group: group of channels name
        condition: condition name
    """
    # Read ECoG
    reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
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
    epochs = epochs.filter(l_freq=args.l_freq, h_freq=None)
    # Downsample
    epochs = epochs.decimate(args.decim)
    times = epochs.times
    # Pick channels
    epochs = epochs.pick(group_chans)
    # Compute time frequency with Morlet wavelet 
    sfreq = 500/args.decim
    freqs = get_freqs(args, sfreq)
    n_cycles = freqs/2
    power = tfr_morlet(epochs, freqs, n_cycles, return_itc=False)
    # Apply baseline correction
    baseline = (args.tf_tmin_baseline, args.tf_tmax_baseline)
    print(f"\n Morlet wavelet: rescaled with {args.mode}")
    print(f"\n Condition is {condition}\n")
    power.apply_baseline(baseline=baseline, mode=args.tf_mode)
    power = power.data
    power = np.average(power, axis=0)
    return power, times, freqs


def get_freqs(args, sfreq):
    """Get frequencies for time frequency analysis"""
    fmin = args.fmin
    nfreqs = args.nfreqs
    fmax = sfreq/2
    fres = (fmax + fmin - 1)/nfreqs
    freqs = np.arange(fmin, fmax, fres)
    return freqs

#%% For a given subject, run time frequency analysis across groups and
# conditions


def cross_tf_analysis(args, cohort):
    for subject in cohort:
        for group in groups:
            for condition in conditions:
                power, time, freqs = compute_group_power(
                    args, subject=subject, group=group, condition=condition
                )
                power_dict["subject"].append(subject)
                power_dict["condition"].append(condition)
                power_dict["group"].append(group)
                power_dict["power"].append(power)
                power_dict["freqs"].append(freqs)
                power_dict["time"].append(time)
    df_power = pd.DataFrame(power_dict)
    return df_power


def main():
    fname = "tf_power_dataframe.pkl"
    df_power = cross_tf_analysis(args, cohort)
    fpath = args.result_path
    fpath = fpath.joinpath(fname)
    df_power.to_pickle(fpath)


if __name__ == "__main__":
    main()
