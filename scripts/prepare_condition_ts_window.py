
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:43:20 2021
This script prepare condition specific time series for further analysis in 
mvgc toolbox
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


#%% Read time series

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
hfb = ecog.read_dataset()

# Read ROI info for mvgc
df_visual = ecog.read_channels_info(fname=args.channels)
df_electrodes = ecog.read_channels_info(fname='electrodes_info.csv')
functional_indices = hf.parcellation_to_indices(df_visual, 'group', matlab=True)
visual_chan = df_visual['chan_name'].to_list()

# %% Pick up channels of interest
# Retinotopic and Face responsive channels
picks = ["LTo1-LTo2", "LGRD60-LGRD61"]

#%% Read condition specific time serieschannels pre peak

window = [[-0.3, -0.05],[0, 0.25],[0.5, 0.75]]
window = np.asarray(window)
nwin = window.shape[0]
tsw = []
timew = []
for w in range(nwin):
    tmin_crop = window[w,0]
    tmax_crop = window[w,1]
    ts, time = hf.category_ts(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=tmin_crop, 
                              tmax_crop=tmax_crop)
    tsw.append(ts)
    timew.append(time)

tsw = np.stack(tsw, axis=-1)
timew = np.stack(timew, axis = -1)

#%% Save time series for GC analysis

ts_dict = {'data': tsw, 'sfreq': args.sfreq, 'time': timew, 'sub_id': args.subject, 
           'functional_indices': functional_indices}
fname = args.subject + '_condition_ts_visual_sliding.mat'
result_path = Path('~','projects','CIFAR','data', 'results').expanduser()
fpath = result_path.joinpath(fname)
savemat(fpath, ts_dict)
#%% To look at anatomical regions

# ROI_indices = hf.parcellation_to_indices(df_visual, 'DK', matlab=True)
# ROI_indices = {'LO': ROI_indices['ctx-lh-lateraloccipital'], 
#                'Fus': ROI_indices['ctx-lh-fusiform'] }
