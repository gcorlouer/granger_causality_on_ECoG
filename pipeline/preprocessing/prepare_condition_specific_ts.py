#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:37:05 2022
This script prepare condition specific time series for rolling mvgc analysis 
or pairwise conditional GC, depending on the tmine range during post stimulus
one considers.

Note we restrict to visually responsive channels here.

@author: guime
"""


from src.preprocessing_lib import EcogReader, Epocher, parcellation_to_indices
from src.input_config import args
from pathlib import Path
from scipy.io import savemat

import numpy as np
#%%

conditions = ['Rest', 'Face', 'Place', 'baseline']
# Original sampling rate
sfreq = 500
log_transf = False
decim = args.decim
sfreq = sfreq/decim
min_postim = args.tmin_crop
max_postim = args.tmax_crop

ts = dict.fromkeys(conditions, [])

for subject in args.cohort:
    for condition in conditions:
        # Read continuous HFA
        reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                         preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                         epoch=False)
        hfb = reader.read_ecog()
        # Read visually responsive channels
        df_visual = reader.read_channels_info(fname='visual_channels.csv')
        visual_chans = df_visual['chan_name'].to_list()
        # Pick visually responsive HFA
        hfb = hfb.pick_channels(visual_chans)
        # Epoch visually responsive HFA
        if condition == 'baseline':
            # Return prestimulus baseline
            epocher = Epocher(condition='Stim', t_prestim=args.t_prestim, t_postim = args.t_postim, 
                         baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                         tmax_baseline=args.tmax_baseline, mode=args.mode)
            if log_transf == True:
                epoch = epocher.log_epoch(hfb)
            else:
                epoch = epocher.epoch(hfb)
             # Downsample by factor of 2 and check decimation
            epoch = epoch.copy().crop(tmin = -0.5, tmax=0)
            epoch = epoch.copy().decimate(args.decim)
        else:
            # Return condition specific epochs
            epocher = Epocher(condition=condition, t_prestim=args.t_prestim, t_postim = args.t_postim, 
                             baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                             tmax_baseline=args.tmax_baseline, mode=args.mode)
            #Epoch condition specific hfb and log transform to approach Gaussian
            if log_transf == True:
                epoch = epocher.log_epoch(hfb)
            else:
                epoch = epocher.epoch(hfb)
            
            epoch = epoch.copy().crop(tmin = args.tmin_crop, tmax=args.tmax_crop)
             # Downsample by factor of 2 and check decimation
            epoch = epoch.copy().decimate(args.decim)
            time = epoch.times
    
        # Prerpare time series for MVGC
        X = epoch.copy().get_data()
        (N, n, m) = X.shape
        X = np.transpose(X, (1,2,0))
        ts[condition] = X
    # Add category specific channels indices to dictionary
    indices = parcellation_to_indices(df_visual,  parcellation='group', matlab=True)
    # Pick populations and order them
    ordered_keys = ['R','O','F']
    ordered_indices = {k: indices[k] for k in ordered_keys}
    ts['indices']= ordered_indices
    
    # Add time
    ts['time'] = time
    
    # Add subject
    ts['subject'] = subject
    
    # Add sampling frequency
    ts['sfreq'] = sfreq/args.decim
    
    # Save condition ts as mat file
    result_path = args.result_path
    fname = subject + '_condition_visual_ts.mat'
    fpath = result_path.joinpath(fname)
    savemat(fpath, ts)

print(f"\n Sampling frequency is {sfreq}Hz\n")
print(f"\n Stimulus is during {min_postim} and {max_postim}s\n")
