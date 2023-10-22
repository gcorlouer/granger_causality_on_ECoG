#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:51:08 2022
In this script we plot a trace of visually responsive ECoG
@author: guime
"""
from src.input_config import args
from src.preprocessing_lib import EcogReader

#%%
# Get visually responsive ecog

reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                     preprocessed_suffix=args.preprocessed_suffix,
                     epoch=args.epoch)
raw = reader.read_ecog()
df_visual = reader.read_channels_info(fname='visual_channels.csv')
visual_chans = df_visual['chan_name'].to_list()
raw = raw.pick_channels(visual_chans)

raw.plot(duration=5, scalings=1e-3, butterfly=False)

#%%

rest = raw.copy().crop(tmin=100, tmax=150)
rest.plot(duration=2, scalings=1e-3, butterfly=False)

#%% Estimate Rest psd

rest.plot_psd(fmin=0, fmax=150, xscale='log')

#%% Plot visual psd

stim = raw.copy().crop(tmin=450, tmax=500)
stim.plot(duration=2, scalings=1e-3, butterfly=False)

#%% Estimate stim psd

stim.plot_psd(fmin=0, fmax=150, xscale='log')
