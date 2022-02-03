#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:22:32 2021

@author: guime
"""


import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing_lib import EcogReader
from input_config import args

#%% 
chan = ['LTo1-LTo2']
reader = EcogReader(args.data_path, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                 epoch=False)

hfb = reader.read_ecog()
hfb = hfb.copy().pick(chan)
hfb = hfb.copy().crop(tmin=500, tmax=506)
#%%
#%matplotlib qt

#hfb.plot()
#%%
sns.set(font_scale=5)
time = hfb.times
X = hfb.copy().get_data()
X = X[0,:]
plt.plot(time, X)
plt.xlabel('Time (s)')
plt.ylabel('HFA (V)')
