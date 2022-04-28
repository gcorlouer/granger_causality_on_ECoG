#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:11:14 2022
We plot single trial GC to compare GC across conditions.
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
#%%
# Take input data
fname = 'single_trial_fc.mat'
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc['dataset']
# Plot single trial fc
plot_single_trial_gfc(fc, cohort, args, F='gGC', baseline= 'Rest', 
                    alternative='greater', vmin=-3, vmax=3, rotation=90, 
                    tau_x=0.5, tau_y=0.8)


