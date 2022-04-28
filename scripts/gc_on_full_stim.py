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
from src.plotting_lib import plot_single_trial_pfc, plot_single_trial_gfc, info_flow_stat
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#%%

home = Path.home()
figpath = home.joinpath('thesis','overleaf_project','figures')
result_path = Path('../results')
# List conditions
conditions = ['Rest', 'Face', 'Place', 'baseline']
cohort = ['AnRa', 'ArLa', 'DiAs']
nsub = len(args.cohort)
#%% Plot multitrial pair FC
# Load functional connectivity matrix
fname = 'multi_trial_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
vmax = 3
#vmax = [11, 15, 12]
(ncdt, nsub) = fc.shape

full_stim_multi_pfc(fc, cohort, args, F='pGC',vmin=-vmax,vmax=vmax, sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8)


#%% Plot multitrial pair MI

full_stim_multi_pfc(fc, cohort, args, F='pMI', vmin=-vmax,vmax=vmax,  sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8)

#%% Plot multitrial groupwise GC
vmin = 3
full_stim_multi_gfc(fc, cohort, args, F='gGC', vmin=vmin,vmax=-vmin,  sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8)

#%% Plot  multitrial groupwise MI
vmin = 4
full_stim_multi_gfc(fc, cohort, args, F='gMI', vmin=vmin,vmax=-vmin,  sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8)


#%% Plot single trial pFC

# Take input data
fname = 'single_trial_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
# Plot single trial fc
plot_single_trial_pfc(fc, cohort, args, F='pGC', baseline= 'Rest', 
                    alternative='greater', vmin=-3, vmax=3, rotation=90, 
                    tau_x=0.5, tau_y=0.8)
#%% Plot single trial gFC
cohort = ['AnRa', 'ArLa', 'DiAs']
# Take input data
fname = 'single_trial_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
vmax = 4
# Plot single trial fc
plot_single_trial_gfc(fc, cohort, args, F='gGC', baseline= 'Rest', 
                    alternative='greater', vmin=-vmax, vmax=vmax, rotation=90, 
                    tau_x=0.5, tau_y=0.8)

#%% Compare bottum up and top down

z, sig, pval = info_flow_stat(fc, cohort, args, subject ='DiAs',F='gGC', baseline= 'Rest', 
                    alternative='two-sided')



















