#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:03:59 2022
In this script we test condition specific time series
@author: guime
"""


from src.preprocessing_lib import EcogReader, Epocher, prepare_condition_ts
from src.input_config import args
from pathlib import Path
from scipy.io import savemat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%

ts = prepare_condition_ts(args.data_path, subject='DiAs', stage='preprocessed', matlab = True,
                     preprocessed_suffix='_hfb_continuous_raw.fif', decim=2,
                     epoch=False, t_prestim=-0.5, t_postim=1.75, tmin_baseline = -0.5,
                     tmax_baseline = 0, tmin_crop=-0.5, tmax_crop=1.5)

#%%


    
#%%

ic = 4

face = ts['Face']
place = ts['Place']
baseline = ts['Rest']

face = np.mean(face[ic,:,:], 1)
place = np.mean(place[ic,:,:], 1)
baseline = np.mean(baseline[ic,:,:], 1)

time = ts['time']

plt.plot(time, face, label ='face')
plt.plot(time, place, label = 'place')
plt.plot(time, baseline, label = 'rest')
plt.legend()
    
    
    
    
#%%
    
#%%





























