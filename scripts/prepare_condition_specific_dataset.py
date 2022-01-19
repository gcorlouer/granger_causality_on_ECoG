#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:57:31 2022

@author: guime
"""

from src.preprocessing_lib import EcogReader
from src.input_config import args


import pandas as pd
import seaborn as sns
import numpy as np

#%% Save condition specific dataset

subject = 'DiAs'

# Read baseline hfb
reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix='_hfb_Face_scaled-epo.fif', epoch=args.epoch)
df= reader.read_channels_info(fname='visual_channels.csv')
visual_chans = df['chan_name'].to_list()
category = df['group']
location = df['DK']
hfb = reader.read_ecog()
hfb = hfb.copy().pick_channels(visual_chans)
hfb = hfb.crop(tmin=-0.5, tmax = 0)
baseline = hfb.copy().get_data()

# Read face hfb
reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix='_hfb_Face_scaled-epo.fif', epoch=args.epoch)
hfb = reader.read_ecog()
hfb = hfb.copy().pick_channels(visual_chans)
hfb = hfb.crop(tmin=0, tmax = 0.5)
X = hfb.copy().get_data()

# Read place hfb
reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix='_hfb_Place_scaled-epo.fif', epoch=args.epoch)
hfb = reader.read_ecog()
hfb = hfb.copy().pick_channels(visual_chans)
hfb = hfb.crop(tmin=0, tmax = 0.5)
Y = hfb.copy().get_data()

#%% Make condition specific dataframe

condition_dict = {'chan_name': [], 'face': [], 'place':[], 'baseline':[], 
                  'category':[],'DK':[], 'Y':[]}
for idx, chan in enumerate(visual_chans):
    condition_dict['chan_name'].append(visual_chans[idx])
    condition_dict['face'].append(X[:,idx,:])
    condition_dict['place'].append(Y[:,idx,:])
    condition_dict['baseline'].append(baseline[:,idx,:])
    condition_dict['category'].append(df['group'][idx])
    condition_dict['DK'].append(location[idx])
    condition_dict['Y'].append(df['Y'][idx])

    
df_condition = pd.DataFrame(condition_dict)
conditions = ['baseline', 'face', 'place']
for condition in conditions:
    df_condition[condition] = df_condition[condition].apply(np.ndarray.flatten, 'columns')
    
df_condition = df_condition.explode(conditions)

for condition in conditions:
    df_condition = df_condition.astype({condition:'float'})
#%% Plot violins

ax = sns.violinplot(x='chan_name', y ='baseline', data = df_condition)