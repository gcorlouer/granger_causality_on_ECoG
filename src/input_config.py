#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:14:06 2021
Config file, contain all parameters for analysis
PROBLEM: How to pass argument into other script by calling them from terminal?
@author: guime
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
#%% Loading data parameters

cohort = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

# Path to source data, derivatives and results. Enter your own path in local machine
data_path = Path('~','projects', 'cifar', 'data').expanduser()
derivatives_path = Path('../data/derivatives/')
result_path = Path('../results/')

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=list, default=data_path)
parser.add_argument("--derivatives_path", type=list, default=derivatives_path)
parser.add_argument("--result_path", type=list, default=result_path)
parser.add_argument("--cohort", type=list, default=cohort)
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--stage", type=str, default='preprocessed')
parser.add_argument("--preprocessed_suffix", type=str, default= '_hfb_Stim_scaled-epo.fif')
parser.add_argument("--epoch", type=bool, default=True)
parser.add_argument("--channels", type=str, default='visual_channels.csv')

# Input parameters:
# Stages
# bipolar_montage
# preprocessed

# Preprocessing suffixes:
# Bad chans removed:
# _bad_chans_removed_raw.fif: 
# Continuous data:
# '_hfb_continuous_raw.fif' 
# Epoched data
# _hfb_Stim_scaled-epo.fif
# _hfb_Face_scaled-epo.fif
# _hfb_Condition_scaled-epo.fif

# Channels info:
# 'BP_channels.csv'
# 'all_bad_channels.csv'
# 'visual_channels.csv'
# 'all_visual_channels.csv'
# 'electrodes_info.csv'

#%% Filtering parameters

parser.add_argument("--l_freq", type=float, default=70.0)
parser.add_argument("--band_size", type=float, default=20.0)
parser.add_argument("--nband", type=float, default=5)
parser.add_argument("--l_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--h_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--filter_length", type=str, default='auto')
parser.add_argument("--phase", type=str, default='minimum')
parser.add_argument("--fir_window", type=str, default='blackman')

#%% Epoching parameters
parser.add_argument("--condition", type=str, default='Stim') 
parser.add_argument("--t_prestim", type=float, default=-0.5)
parser.add_argument("--t_postim", type=float, default=1.75)
parser.add_argument("--baseline", default=None) # No baseline from MNE
parser.add_argument("--preload", default=True)
parser.add_argument("--tmin_baseline", type=float, default=-0.4)
parser.add_argument("--tmax_baseline", type=float, default=-0.1)
parser.add_argument("--mode", type=str, default='logratio')

#%% Visually responsive channels classification parmeters

parser.add_argument("--tmin_prestim", type=float, default=-0.4)
parser.add_argument("--tmax_prestim", type=float, default=-0.05)
parser.add_argument("--tmin_postim", type=float, default=0.05)
parser.add_argument("--tmax_postim", type=float, default=0.4)
parser.add_argument("--alpha", type=float, default=0.05)
#parser.add_argument("--zero_method", type=str, default='pratt')
parser.add_argument("--alternative", type=str, default='greater')

#%% Create category specific time series

parser.add_argument("--decim", type=float, default=5)
parser.add_argument("--tmin_crop", type=float, default=0)
parser.add_argument("--tmax_crop", type=float, default=1)

#%% Functional connectivity parameters

parser.add_argument("--nfreq", type=float, default=1024)
parser.add_argument("--roi", type=str, default="functional")

# Types of roi:
# functional
# anatomical

#%%

args = parser.parse_args()
#print(f"Subject is {args.subject}")

#%% Style parameters

plt.style.use('ggplot')
fig_width = 17  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'lines.linewidth': 1.5,
          'axes.labelsize': 12,
          'axes.titlesize': 8,
          'font.size': 12,
          'legend.fontsize': 8,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
