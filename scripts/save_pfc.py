#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:28:26 2022
In this script we save pairwise conditional gc and mi as a csv dataset
@author: guime
"""


from src.preprocessing_lib import EcogReader, build_dfc
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path

import pandas as pd
#%% Read matlab format data


reader = EcogReader(args.data_path)
# Read visual channels 
df_visual = reader.read_channels_info(fname='visual_channels.csv')

# Load functional connectivity matrix
result_path = Path('../results')

fname = 'pairwise_fc.mat'
fc_path = result_path.joinpath(fname)

fc = loadmat(fc_path)
fc = fc['dataset']

# Build dataset fc dictionary

dfc = build_dfc(fc)

# Save dataset as csv file

fname = 'pairwise_fc.csv'
fpath = result_path.joinpath(fname)
dfc.to_csv(fpath, index=False)

df = pd.read_csv(fpath)