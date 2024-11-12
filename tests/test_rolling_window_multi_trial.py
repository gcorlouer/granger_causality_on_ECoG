#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:21:59 2022

@author: guime
"""

from libs.input_config import args
from libs.preprocessing_lib import EcogReader
from libs.plotting_lib import plot_rolling_specrad, plot_multi_fc, sort_populations
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# %%

# List conditions
conditions = ["Rest", "Face", "Place", "baseline"]
cohort = ["AnRa", "ArLa", "DiAs"]
# Load functional connectivity matrix
result_path = Path("../results")

fname = "rolling_multi_trial_fc.mat"
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc["dataset"]
# Read channels
(subject, s) = ("DiAs", 2)
reader = EcogReader(args.data_path, subject=subject)
df_visual = reader.read_channels_info(fname="visual_channels.csv")

# %% Plot pgc

cdt = ["Rest", "Face"]
fchan = "LGRD60-LGRD61"
rchan = "LTo1-LTo2"

iF = df_visual.index[df_visual["chan_name"] == fchan].to_list()[0]
iR = df_visual.index[df_visual["chan_name"] == rchan].to_list()[0]

for c in range(4):
    pgc = fc[c, s]["pGC"]["gc"][0][0]
    time = fc[c, s]["time"]
    plt.subplot(4, 1, c + 1)
    plt.plot(time, pgc[iF, iR], label="R to F")
    plt.plot(time, pgc[iR, iF], label="F to R")
    plt.ylim(0, 0.005)

plt.legend()

# %% Plot significance


for c in range(4):
    pgc = fc[c, s]["pGC"]["sig"][0][0]
    time = fc[c, s]["time"]
    plt.subplot(4, 1, c + 1)
    plt.plot(time, pgc[iF, iR], label="R to F")
    plt.plot(time, pgc[iR, iF], label="F to R")

plt.legend()

# %% plot group GC

group = ["O", "F", "R"]
for c in range(4):
    iF = group.index("F")
    iR = group.index("R")
    indices = fc[c, s]["indices"]
    gc = fc[c, s]["gGC"]["gc"][0][0]
    time = fc[c, s]["time"]
    plt.subplot(4, 1, c + 1)
    plt.plot(time, gc[iF, iR], label="R to F")
    plt.plot(time, gc[iR, iF], label="F to R")
    plt.ylim(0, 0.03)

plt.legend()


# %% Significance of group GC

for c in range(4):
    indices = fc[c, s]["indices"]
    group = list(indices.dtype.fields.keys())
    gc = fc[c, s]["gGC"]["sig"][0][0]
    iF = group.index("F")
    iR = group.index("R")
    time = fc[c, s]["time"]
    plt.subplot(4, 1, c + 1)
    plt.plot(time, gc[iF, iR], label="R to F")
    plt.plot(time, gc[iR, iF], label="F to R")

plt.legend()
