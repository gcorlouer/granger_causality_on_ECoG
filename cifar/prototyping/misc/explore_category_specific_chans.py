#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 12:02:35 2022
We explore face vs place response during rest, face and place presentation
WARNING: Make sure you are using 

@author: guime
"""


from src.preprocessing_lib import prepare_condition_scaled_ts
from src.input_config import args
from pathlib import Path
from scipy.stats import sem

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%% Make condition specific dictionary 
#%matplotlib qt
cohort = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
sns.set_theme(font_scale=2)
subject = 'SoGi'
path = args.data_path
ts = prepare_condition_scaled_ts(path, subject=subject, matlab=False)
indices = ts['indices']
baseline = ts['baseline']
baseline = np.average(baseline)
# Adapt to python indexing
conditions = ['Rest', 'Face', 'Place']
f, ax = plt.subplots(3,1, sharex=True, sharey = True)
for i, condition in enumerate(conditions):
    X = ts[condition]
    time = ts['time']
    for cat in indices.keys():
        idx = indices[cat]
        # Average HFA over channels in same category
        cat_chans = np.mean(X[idx,...],0)
        # Compute evoked response of category specific chans
        evok = np.mean(cat_chans, axis=1)
        # Compute standard error of mean averaged accross trials
        sm = sem(cat_chans, axis = 1)
        ax[i].plot(time, evok, label=cat)
        ax[i].fill_between(time, evok - 1.96*sm, evok + 1.96*sm, alpha=0.5)
    ax[i].axvline(x=0, color = 'k')
    ax[i].axhline(y=baseline, color = 'k')
    ax[i].set_ylabel(f"HFA {condition} (dB)")

ax[2].set_xlabel("Time (s)")
plt.legend()
plt.suptitle(f"{subject} condition and category specific HFA")