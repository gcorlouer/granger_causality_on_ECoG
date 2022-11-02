#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:04:52 2021
Classify visual channels for all subjects
@author: guime
"""
import pandas as pd
import argparse 

from src.preprocessing_lib import EcogReader, VisualClassifier
from pathlib import Path

#%%

cohort = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

# Path to source data, derivatives and results. Enter your own path in local machine
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
derivatives_path = data_path.joinpath('derivatives')
result_path = cifar_path.joinpath('results')
fig_path = cifar_path.joinpath('results/figures')

parser = argparse.ArgumentParser()
# Paths
parser.add_argument("--data_path", type=list, default=data_path)
parser.add_argument("--derivatives_path", type=list, default=derivatives_path)
parser.add_argument("--result_path", type=list, default=result_path)
parser.add_argument("--fig_path", type=list, default=result_path)

# Dataset parameters 
parser.add_argument("--cohort", type=list, default=cohort)
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--sfeq", type=float, default=500.0)

parser.add_argument("--stage", type=str, default='preprocessed')
parser.add_argument("--preprocessed_suffix", type=str, default= '_hfb_Stim_scaled-epo.fif')
parser.add_argument("--epoch", type=bool, default=True)
parser.add_argument("--channels", type=str, default='visual_channels.csv')
parser.add_argument("--matlab", type=bool, default=False)

#% Visually responsive channels classification parmeters

parser.add_argument("--tmin_prestim", type=float, default=-0.4)
parser.add_argument("--tmax_prestim", type=float, default=-0.05)
parser.add_argument("--tmin_postim", type=float, default=0.05)
parser.add_argument("--tmax_postim", type=float, default=0.4)
parser.add_argument("--alpha", type=float, default=0.05)
#parser.add_argument("--zero_method", type=str, default='pratt')
parser.add_argument("--alternative", type=str, default='greater')

#% Create category specific time series

parser.add_argument("--decim", type=float, default=2)
parser.add_argument("--tmin_crop", type=float, default=-0.5)
parser.add_argument("--tmax_crop", type=float, default=1.5)

#% Functional connectivity parameters

parser.add_argument("--nfreq", type=float, default=1024)
parser.add_argument("--roi", type=str, default="functional")

args = parser.parse_args()
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
               tmax_postim=args.tmax_postim, alpha=args.alpha, alternative=args.alternative)
    visual_populations = classifier.classify_visual_chans(hfb, dfelec)
    
    # Save visually responsive populations into csv file
    df_visual = pd.DataFrame.from_dict(visual_populations)
    #df_visual = df_visual.sort_values(by='Y', ignore_index=True)
    # Remove channels with 0 latency response
    df_visual = df_visual[df_visual.latency != 0]
    fname = args.channels
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
    fname = args.channels # or visual channels
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
