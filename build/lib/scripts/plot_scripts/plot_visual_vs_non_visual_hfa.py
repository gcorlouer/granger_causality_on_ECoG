#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 21:31:36 2022

@author: guime
"""


from src.preprocessing_lib import EcogReader, Epocher
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import argparse

#%% Plot parameters

plt.style.use('ggplot')
fig_width = 16  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 10
params = {'backend': 'ps',
          'lines.linewidth': 1.5,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'font.size': label_size,
          'legend.fontsize': label_size,
          'xtick.labelsize': label_size,
          'ytick.labelsize': label_size,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

#%%

cohort = ['AnRa',  'ArLa', 'DiAs']
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

args = parser.parse_args()

#%%
# Read visual chans
for i, subject in enumerate(args.cohort):
    reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                     preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
    df_visual= reader.read_channels_info(fname=args.channels)
    visual_chans = df_visual['chan_name'].to_list()
    category = df_visual['group'].to_list()
    location = df_visual['DK'].to_list()
    
    # Read hfb
    
    hfb = reader.read_ecog()
    
    #epocher = Epocher()
    # hfb = epocher.epoch(hfb)
    hfb_visual = hfb.copy().pick_channels(visual_chans)
    hfb_nv = hfb.copy().drop_channels(visual_chans)
    baseline = hfb_visual.copy().crop(tmin=-0.5, tmax=0).get_data()
    baseline = np.average(baseline)
    
    #%% Plot event related potential of visual channels
    
    evok_visual = hfb_visual.average()
    
    #%% Plot event related potential of non visual channels
    
    evok_nv = hfb_nv.average()
    
    time = evok_visual.times
    #%% 
    
    X = evok_visual.get_data()
    mX = np.mean(X,0)
    Y = evok_nv.get_data()
    mY = np.mean(Y,0)
    plt.subplot(3,3, i+1)
    plt.plot(time, mX, label='visual')
    plt.plot(time, mY, label='non visual')
    plt.axhline(y=baseline)
    plt.axvline(x=0)
    plt.xlabel('Time (s)')
    plt.ylabel(f'HFA {subject}')
    #plt.legend()

plt.show()
