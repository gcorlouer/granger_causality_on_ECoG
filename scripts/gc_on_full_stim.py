#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:20:03 2022
In this script we plot GC analysis on full trial duration. This include
multitrial GC and single trial GC
@author: guime
"""


from src.input_config import args
from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.plotting_lib import full_stim_multi_pfc, full_stim_multi_gfc
from src.plotting_lib import plot_single_trial_pfc, plot_single_trial_gfc
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#%%

home = Path.home()
figpath = home.joinpath('thesis','overleaf_project','figures')
# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
nsub = len(args.cohort)
#%% Plot multitrial pair FC
# Load functional connectivity matrix
result_path = Path('../results')
fname = 'multi_trial_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
vmin = 0
#vmax = [11, 15, 12]
(ncdt, nsub) = fc.shape
cohort = ['AnRa', 'ArLa', 'DiAs']

full_stim_multi_pfc(fc, cohort, args, F='pGC',vmin=-3,vmax=3, sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8)


#%% Plot multitrial pair MI

full_stim_multi_pfc(fc, cohort, args, F='pMI', vmin=-6,vmax=6,  sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8)

#%% Plot multitrial groupwise GC

full_stim_multi_gfc(fc, cohort, args, F='gGC', vmin=-2,vmax=2,  sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8)

#%% Plot  multitrial groupwise MI

full_stim_multi_gfc(fc, cohort, args, F='gMI', vmin=-2,vmax=2,  sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8)


#%% Plot single trial pGC

# Take input data
fname = 'single_trial_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
cohort = ['AnRa', 'ArLa', 'DiAs']
# Plot single trial fc
plot_single_trial_pfc(fc, cohort, args, F='pGC', baseline= 'Rest', 
                    alternative='greater', vmin=-2, vmax=2, rotation=90, 
                    tau_x=0.5, tau_y=0.8)
#%% Plot single trial gGC

# Take input data
fname = 'single_trial_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
cohort = ['AnRa', 'ArLa', 'DiAs']
# Plot single trial fc
plot_single_trial_gfc(fc, cohort, args, F='gGC', baseline= 'Rest', 
                    alternative='greater', vmin=-2, vmax=2, rotation=90, 
                    tau_x=0.5, tau_y=0.8)



















