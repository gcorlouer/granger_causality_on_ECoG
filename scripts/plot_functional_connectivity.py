#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:31:12 2021
This script plot functional connectivity i.e. Mutual information and pairwise
conditional Granger causality.
@author: guime
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing_lib import EcogReader
from scipy.io import loadmat
from input_config import args
from pathlib import Path, PurePath


#%% Read ROI and functional connectivity data

subject = "DiAs"

reader = EcogReader(args.data_path, subject=subject, stage='preprocessed')
# Read visual channels 
df_visual = reader.read_channels_info(fname='visual_channels.csv')

# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']

# Load functional connectivity matrix
result_path = Path('../results')

fname = subject + "_pairwise_dfc.mat"
fc_path = result_path.joinpath(fname)

fc = loadmat(fc_path)
fc = fc['dfc']
#%% Plot functional connectivity


#%%
