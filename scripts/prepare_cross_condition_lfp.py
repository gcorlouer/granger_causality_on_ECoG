#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 01:17:38 2021

@author: guime
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:27:18 2021

@author: guime
"""

#%%

import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat

cohort = ['AnRa',  'ArLa',  'BeFe',  'DiAs',  'JuRo']
#result_path = Path('~','neocortex', 'results').expanduser()
result_path = Path('~','projects', 'CIFAR', 'data', 'results').expanduser()

stage = '_bad_chans_removed_raw.fif'
#%% Check who has face and retinotopic channels

for subject in cohort :
    
    ecog = hf.Ecog(args.cohort_path, subject=subject, proc=args.proc, 
                           stage = stage, epoch=args.epoch)
    df_visual = ecog.read_channels_info(fname=args.channels)
    functional_group = df_visual["group"].unique()
    functional_indices = hf.parcellation_to_indices(df_visual, 'group', matlab=True)
    visual_chan = df_visual['chan_name'].to_list()
    hfb = ecog.read_dataset()

#%% Prepare condition baseline

    tmin_crop = -0.5
    tmax_crop = 0
    
    baseline, timeb = hf.category_lfp(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                                  tmax_crop=args.tmax_crop)

    #%% Prepare condition time series
    
    tmin_crop = -0.5
    tmax_crop = 1.5
    
    # Read visual hfb
    ts, time = hf.category_lfp(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                                  tmax_crop=args.tmax_crop)

    #%% Save time series for GC analysis
    
    ts_dict = {'rest': ts[0], 'face': ts[1], 'place': ts[2], 'sfreq': args.sfreq, 
               'time': time, 'sub_id': subject, 
               'functional_indices': functional_indices, 'ts_type':'lfp', 
               'rest_baseline': baseline[0], 'face_baseline': baseline[1], 
               'place_baseline':baseline[2], 'timeb': timeb}
    fname = subject + '_condition_lfp_visual.mat'
    fpath = result_path.joinpath(fname)
    savemat(fpath, ts_dict)
