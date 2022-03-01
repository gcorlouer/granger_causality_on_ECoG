#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:06:00 2022
In this script we test function from plotting library
@author: guime
"""

from src.input_config import args
from src.plotting_lib import plot_rolling_specrad, plot_rolling_var
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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







































