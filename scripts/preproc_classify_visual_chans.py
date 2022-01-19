#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:04:52 2021
Classify visual channels for all subjects
@author: guime
"""
import pandas as pd

from src.preprocessing_lib import EcogReader, VisualClassifier
from src.input_config import args
from pathlib import Path, PurePath
#%% Detect visual channels
derivatives_path = args.derivatives_path
for subject in args.cohort:
    # Read baseline rescaled hfb
    reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
    hfb = reader.read_ecog()
    dfelec = reader.read_channels_info(fname='electrodes_info.csv')
    # Classify visually responsive channels
    classifier = VisualClassifier(tmin_prestim=args.tmin_prestim, 
                              tmax_prestim=args.tmax_prestim, 
                              tmin_postim=args.tmin_postim,
               tmax_postim=args.tmax_postim, alpha=args.alpha, 
               zero_method=args.zero_method, alternative=args.alternative)
    visual_populations = classifier.classify_visual_chans(hfb, dfelec)
    
    # Save visually responsive populations into csv file
    df_visual = pd.DataFrame.from_dict(visual_populations)
    #df_visual = df_visual.sort_values(by='Y', ignore_index=True)
    # Remove channels with 0 latency response
    df_visual = df_visual[df_visual.latency != 0]
    fname = 'visual_channels.csv'
    subject_path = derivatives_path.joinpath(subject)
    brain_path = subject_path.joinpath('brain')
    fpath = brain_path.joinpath(fname)
    df_visual.to_csv(fpath, index=False)

#%% Make a table of all visual channels for all subjects
result_path = args.result_path
columns = list(df_visual.columns)
columns.append('subject_id')
df_all_visual_chans = pd.DataFrame(columns=columns)

for subject in args.cohort:
    fname = 'visual_channels.csv'
    subject_path = derivatives_path.joinpath(subject)
    brain_path = subject_path.joinpath('brain')
    fpath = brain_path.joinpath(fname)
    df_visual = pd.read_csv(fpath)
    subject_id = [subject]*len(df_visual)
    df_visual['subject_id'] = subject_id
    df_all_visual_chans = df_all_visual_chans.append(df_visual)
    

fname = 'all_visual_channels.csv'
fpath = result_path.joinpath(fname)
df_all_visual_chans.to_csv(fpath, index=False)

#%% Read all visual chans dataframe

df_all_visual_chans = pd.read_csv(fpath)
