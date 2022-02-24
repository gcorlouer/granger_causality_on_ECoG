#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:06:00 2022
In this script we test function from plotting library
@author: guime
"""

from src.input_config import args
from src.plotting_lib import plot_narrow_broadband, plot_log_trial

from pathlib import Path
#%%

fname = 'DiAs_narrow_broadband_stim.png'
home = Path.home()
fpath = home.joinpath('thesis','overleaf_project','figures')
plot_narrow_broadband(args, fpath, fname=fname, chan = ['LTo1-LTo2'], tmin=500, tmax=506)

#%%
fname = 'DiAs_log_trial.pdf'
home = Path.home()
fpath = home.joinpath('thesis','overleaf_project','figures')
plot_log_trial(args, fpath, fname = fname, 
                          chan = ['LTo1-LTo2'], itrial=2, nbins=50)