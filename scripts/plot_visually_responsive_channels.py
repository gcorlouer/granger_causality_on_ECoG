#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:18:05 2022

@author: guime
"""

from src.preprocessing_lib import EcogReader

import pandas as pd
from src.input_config import args

#%% Make condition specific dictionary

subject = 'DiAs'

# Read visual chans
reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
df= reader.read_channels_info(fname='visual_channels.csv')
visual_chans = df['chan_name'].to_list()
category = df['group']
location = df['DK']

# Read baseline hfb
hfb = reader.read_ecog()
hfb = hfb.copy().pick_channels(visual_chans)
hfb = hfb.crop(tmin=-0.45, tmax = -0.05)
baseline = hfb.copy().get_data()

# Read face hfb
reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix='_hfb_Face_scaled-epo.fif', epoch=args.epoch)
hfb = reader.read_ecog()
hfb = hfb.copy().pick_channels(visual_chans)
X = hfb.copy().get_data()

# Read place hfb
reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix='_hfb_Place_scaled-epo.fif', epoch=args.epoch)
hfb = reader.read_ecog()
hfb = hfb.copy().pick_channels(visual_chans)
Y = hfb.copy().get_data()

hfb_condition = {'baseline':baseline, 'Face': X, 'Place': Y}

#%% 

