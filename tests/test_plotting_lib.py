#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:06:00 2022
In this script we test function from plotting library
@author: guime
"""

from src.input_config import args
from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.plotting_lib import plot_rolling_specrad, plot_multi_fc, sort_populations
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

home = Path.home()
fpath = home.joinpath('thesis','overleaf_project','figures')
#%% Plot narrow band

fname = 'DiAs_narrow_broadband_stim.png'
home = Path.home()
fpath = home.joinpath('thesis','overleaf_project','figures')
plot_narrow_broadband(args, fpath, fname=fname, chan = ['LTo1-LTo2'], tmin=500, tmax=506)

#%% Plot log trial
fname = 'DiAs_log_trial.pdf'
home = Path.home()
fpath = home.joinpath('thesis','overleaf_project','figures')
plot_log_trial(args, fpath, fname = fname, 
                          chan = ['LTo1-LTo2'], itrial=2, nbins=50)
#%% Plot visual trial
fname = 'DiAs_visual_trial.pdf'
home = Path.home()
fpath = home.joinpath('thesis','overleaf_project','figures')
plot_visual_trial(args, fpath, fname = fname, 
                          chan = ['LTo1-LTo2'], itrial=2, nbins=50)
#%% Plot visual vs non visual
fname = 'visual_vs_non_visual.pdf'
plot_visual_vs_non_visual(args, fpath, fname=fname)
#%% Plot hierachical ordering of channels

reg = [('Y','latency'), ('Y','visual_responsivity'),('latency', 'visual_responsivity'),
 ('Y','category_selectivity')]
save_path = fpath

plot_linreg(reg, save_path, figname = 'visual_hierarchy.pdf')

#%%

plot_condition_ts(args, fpath, subject='DiAs')

#%% Plot var

result_path = Path('..','results')
fname = 'rolling_var_estimation.csv'
fpath = Path.joinpath(result_path, fname)
df = pd.read_csv(fpath)

fpath = home.joinpath('thesis','overleaf_project','figures')

plot_rolling_var(df, fpath, ncdt =3, momax=10, figname='rolling_var.pdf')




#%% Plot Spectral radius

# Read input
result_path = Path('..','results')
fname = 'rolling_var_estimation.csv'
fpath = Path.joinpath(result_path, fname)
df = pd.read_csv(fpath)
fpath = home.joinpath('thesis','overleaf_project','figures')

plot_rolling_specrad(df, fpath, ncdt =3, momax=10, figname='rolling_specrad.pdf')

#%% Plot multitrial pcGC

# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs'];
# Load functional connectivity matrix
result_path = Path('../results')
fname = 'multi_trial_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
# Read visual channels 
(subject,s) = ('DiAs',2)
reader = EcogReader(args.data_path, subject=subject)
df_visual = reader.read_channels_info(fname='visual_channels.csv')
# 
fpath = home.joinpath('thesis','overleaf_project','figures')
figname = subject + '_multi_pcgc.pdf'
fpath = fpath.joinpath(figname)
plot_multi_fc(fc, df_visual, fpath, mode='group', s=2)

#%% Plot multitrial pcGC
# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs'];
# Load functional connectivity matrix
result_path = Path('../results')

fname = 'pairwise_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
# Read visual channels 
(subject,s) = ('DiAs',2)
reader = EcogReader(args.data_path, subject=subject)
df_visual = reader.read_channels_info(fname='visual_channels.csv')
populations = df_visual['group'].tolist()
populations = parcellation_to_indices(df_visual, parcellation='group', matlab=False)
fpath = home.joinpath('thesis','overleaf_project','figures')
figname = subject + '_multi_pcgc.pdf'
fpath = fpath.joinpath(figname)
plot_multi_fc(fc, populations, fpath,  s=2)

#%% Plot multitrial groupwise GC

# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa',  'ArLa', 'DiAs'];
# Load functional connectivity matrix
result_path = Path('../results')

fname = 'mvgc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
# Read visual channels 
(subject,s) = ('DiAs',2)
reader = EcogReader(args.data_path, subject=subject)
df_visual = reader.read_channels_info(fname='visual_channels.csv')
populations = parcellation_to_indices(df_visual, parcellation='group', matlab=False)
populations = list(populations.keys())
#populations = parcellation_to_indices(df_visual, parcellation='group', matlab=False)
fpath = home.joinpath('thesis','overleaf_project','figures')
figname = subject + '_multi_mvgc.pdf'
fpath = fpath.joinpath(figname)
plot_multi_fc(fc, populations, fpath,  s=2)

#%% Plot single trial distribution


















