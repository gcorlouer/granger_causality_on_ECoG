#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:57:30 2021
This script plot grand average event related HFB
@author: guime
"""

import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat

#%%
cohort_path = args.cohort_path
fpath = cohort_path.joinpath('all_visual_channels.csv')
df_all_visual_chans = pd.read_csv(fpath)
#%% Cross subject ts

cross_ts, time = hf.cross_subject_ts(args.cohort_path, args.cohort, proc=args.proc, 
                     channels = args.channels,
                     stage=args.stage, epoch=args.epoch,
                     sfreq=args.sfreq, tmin_crop=-0.5, 
                     tmax_crop=1.75)

#%%

cross_population_hfb, populations = hf.ts_to_population_hfb(cross_ts, 
                                                            df_all_visual_chans,
                                                            parcellation='group')
()
#%% Compute mean and SEM of cross populations
ntrials = cross_population_hfb.shape[2]
mean = np.mean(cross_population_hfb,2)
sem = np.std(cross_population_hfb,2)/np.sqrt(ntrials)
max_evok = np.amax(mean+1.96*sem)
#%%

plt.rcParams['font.size'] = '16'

step = 0.1
alpha = 0.5
sns.set(font_scale=1.6)
color = ['k', 'b', 'g']
cat = ['Rest', 'Face', 'Place']
ncat = len(cat)
npop = len(populations)
xticks = np.arange(-0.5, 1.5, 0.1)

f, ax = plt.subplots(3,1)
for i in range(ncat):
    for j in range(npop):
        ax[i].plot(time, mean[j,:,i], label = populations[j])
        ax[i].fill_between(time, mean[j,:,i]-1.96*sem[j,:,i], mean[j,:,i]+1.96*sem[j,:,i], 
          alpha=alpha)
        ax[i].set_ylim(bottom=-1, top=max_evok+1)
        ax[i].xaxis.set_ticks(xticks)
        ax[i].axvline(x=0, color='k')
        ax[i].axhline(y=0, color='k')
        ax[i].set_ylabel(f'HFB {cat[i]} (dB)')
        ax[i].legend()

plt.xlabel('Time (s)')