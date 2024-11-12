#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:06:00 2022
In this script we test function from plotting library
@author: guime
"""

from src.input_config import args
from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.plotting_lib import plot_narrow_broadband, plot_log_trial, plot_visual_trial
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

home = Path.home()
fpath = home.joinpath("thesis", "overleaf_project", "figures")
# %% Plot narrow band

fname = "DiAs_narrow_broadband_stim.jpg"
home = Path.home()
fpath = home.joinpath("thesis", "overleaf_project", "figures", "method_figure")
plot_narrow_broadband(args, fpath, fname=fname, chan=["LTo1-LTo2"], tmin=500, tmax=506)

# %% Plot log trial
fname = "DiAs_log_trial.jpg"
home = Path.home()
fpath = home.joinpath("thesis", "overleaf_project", "figures", "method_figure")

plot_log_trial(args, fpath, fname=fname, chan=["LTo1-LTo2"], itrial=2, nbins=50)
# %% Plot visual trial
fname = "DiAs_visual_trial.jpg"
home = Path.home()
fpath = home.joinpath("thesis", "overleaf_project", "figures")
plot_visual_trial(args, fpath, fname=fname, chan=["LTo1-LTo2"], itrial=2, nbins=50)
# %% Plot visual vs non visual
# Need to read epoch data (use epo.fif
from src.plotting_lib import plot_visual_vs_non_visual

fname = "visual_vs_non_visual.jpg"
fpath = home.joinpath("thesis", "overleaf_project", "figures", "method_figure")

plot_visual_vs_non_visual(args, fpath, fname=fname)
# %% Plot hierachical ordering of channels
from src.plotting_lib import plot_linreg

reg = [
    ("Y", "latency (ms)"),
    ("Y", "Responsivity (Z)"),
    ("Latency (ms)", "Amplitude"),
    ("Y", "Selectivity (d)"),
]
fpath = home.joinpath("thesis", "overleaf_project", "figures", "method_figure")
save_path = fpath

plot_linreg(reg, save_path, figname="visual_hierarchy.jpg")

# %%
from src.plotting_lib import plot_condition_ts

fpath = home.joinpath("thesis", "overleaf_project", "figures", "method_figure")
plot_condition_ts(args, fpath, subject="DiAs", figname="_condition_ts.jpg")

# %% Plot var

from src.plotting_lib import plot_rolling_var

result_path = Path("..", "results")
fname = "rolling_var_estimation.csv"
fpath = Path.joinpath(result_path, fname)
df = pd.read_csv(fpath)

figname = "rolling_var.jpg"

fpath = home.joinpath("thesis", "overleaf_project", "figures", "method_figure")

plot_rolling_var(df, fpath, momax=10, figname=figname)


# %% Plot Spectral radius
from src.plotting_lib import plot_rolling_specrad

# Read input
result_path = Path("..", "results")
fname = "rolling_var_estimation.csv"
fpath = Path.joinpath(result_path, fname)
df = pd.read_csv(fpath)

fpath = home.joinpath("thesis", "overleaf_project", "figures", "method_figure")
plot_rolling_specrad(df, fpath, ncdt=3, momax=10, figname="rolling_specrad.jpg")


# %% Plot rolling window on multitrial

# List conditions
conditions = ["Rest", "Face", "Place"]
cohort = ["AnRa", "ArLa", "DiAs"]
# Load functional connectivity matrix
result_path = Path("../results")

fname = "rolling_multi_trial_fc.mat"
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc["dataset"]

figpath = home.joinpath("thesis", "overleaf_project", "figures")
figname = "cross_rolling_multi_mvgc.pdf"
figpath = fpath.joinpath(figname)

plot_multitrial_rolling_fc(fc, figpath, interaction="gGC", fc_type="gc")
