#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 01:16:58 2022

@author: guime
"""

from pathlib import Path
from scipy.io import loadmat
from src.input_config import args

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# %%

home = Path.home()
figpath = home.joinpath("thesis", "overleaf_project", "figures")
result_path = home.joinpath("projects/cifar/results")
# List conditions
conditions = ["Rest", "Face", "Place", "baseline"]
cohort = ["AnRa", "ArLa", "DiAs"]
nsub = len(args.cohort)
vmax = 3
vmin = -vmax
sfreq = 500
decim = args.decim
sfreq = sfreq / decim
min_postim = args.tmin_crop
max_postim = args.tmax_crop
print(f"\n Sampling frequency is {sfreq}Hz\n")
print(f"\n Stimulus is during {min_postim} and {max_postim}s\n")

# %%

# %matplotlib qt
fname = "null_gc_all_chan.mat"
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc["PwGC"]

F = fc["Face"][0][0]["F"][0][0]
sig = fc["Face"][0][0]["sig"][0][0]
Frest = fc["Rest"][0][0]["F"][0][0]

Fscale = np.log(F / Frest)

sns.heatmap(Fscale, cmap="bwr", vmax=vmax, vmin=vmin)

# %%

sns.heatmap(sig, cmap="bwr", vmax=vmax, vmin=vmin)
