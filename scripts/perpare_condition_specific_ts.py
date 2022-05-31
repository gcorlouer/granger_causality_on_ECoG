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
ts = dict.fromkeys(conditions, [])

for subject in  args.cohort:
    for condition in conditions:
        # Read continuous HFA
        reader = EcogReader(args.data_path, subject=subject, stage='preprocessed',
                         preprocessed_suffix='_hfb_continuous_raw.fif', preload=True, 
                         epoch=False)
        hfb = reader.read_ecog()
        df_visual = reader.read_channels_info(fname='visual_channels.csv')
        visual_chans = df_visual['chan_name'].to_list()
        hfb = hfb.pick_channels(visual_chans)
        # Epoch HFA
        if condition == 'baseline':
            # Return prestimulus baseline
            epocher = Epocher(condition='Stim', t_prestim=args.t_prestim, t_postim = args.t_postim, 
                         baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                         tmax_baseline=args.tmax_baseline, mode=args.mode)
            epoch = epocher.log_epoch(hfb)
             # Downsample by factor of 2 and check decimation
            epoch = epoch.copy().crop(tmin = -0.5, tmax=0)
            epoch = epoch.copy().decimate(args.decim)
        else:
            # Return condition specific epochs
            epocher = Epocher(condition=condition, t_prestim=args.t_prestim, t_postim = args.t_postim, 
                             baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                             tmax_baseline=args.tmax_baseline, mode=args.mode)
            epoch = epocher.log_epoch(hfb)
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
    result_path = Path('../results')
    fname = subject + '_condition_visual_ts.mat'
    fpath = result_path.joinpath(fname)
    savemat(fpath, ts)
