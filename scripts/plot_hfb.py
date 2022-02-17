#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this script we plot the high frequency narrow and broad envelope.
@author: guime
"""


import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing_lib import EcogReader
from src.input_config import args

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
