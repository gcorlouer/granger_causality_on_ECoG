#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:36:28 2022

@author: guime
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.preprocessing_lib import EcogReader, parcellation_to_indices, plot_multi_fc
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path


#%% Read ROI and functional connectivity data


# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs'];
# Load functional connectivity matrix
result_path = Path('../results')

fname = 'pairwise_fc.mat'
fc_path = result_path.joinpath(fname)

fc = loadmat(fc_path)
fc = fc['dataset']

#%% Plot mulittrial GC
(subject,s) = ('DiAs',2)
reader = EcogReader(args.data_path, subject=subject)

# Read visual channels 
df_visual = reader.read_channels_info(fname='visual_channels.csv')
populations = df_visual['group']
plot_multi_fc(fc, populations, s=s)      




























