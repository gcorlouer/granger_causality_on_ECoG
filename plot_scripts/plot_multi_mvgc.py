#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:05:48 2022
In this script we plot MVGC on multitrial
@author: guime
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.preprocessing_lib import EcogReader, plot_multi_fc, parcellation_to_indices
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path

#%% Read data

# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs'];
# Load functional connectivity matrix
result_path = Path('../results')

fname = 'mvgc.mat'
fpath = result_path.joinpath(fname)

mvgc = loadmat(fpath)
mvgc = mvgc['dataset']

#%%
# Read visual channels 
(subject,s) = ('DiAs',2)
reader = EcogReader(args.data_path, subject=subject)
df_visual = reader.read_channels_info(fname='visual_channels.csv')
populations = parcellation_to_indices(df_visual, parcellation='group', matlab=False)

plot_multi_fc(mvgc, populations, s=s, sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8, 
                                 font_scale=1.6)


