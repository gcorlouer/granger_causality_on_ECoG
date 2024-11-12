#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:12:18 2022

@author: guime
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.io import loadmat
from pathlib import Path

# %%

plt.style.use("ggplot")
fig_width = 24  # figure width in cm
inches_per_cm = 0.393701  # Convert cm to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width * inches_per_cm  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
label_size = 10
params = {
    "backend": "ps",
    "lines.linewidth": 1,
    "axes.labelsize": label_size,
    "axes.titlesize": label_size,
    "font.size": label_size,
    "legend.fontsize": label_size,
    "xtick.labelsize": label_size,
    "ytick.labelsize": label_size,
    "text.usetex": False,
    "figure.figsize": fig_size,
}
plt.rcParams.update(params)

# %%
cifar_path = Path("~", "projects", "cifar").expanduser()
data_path = cifar_path.joinpath("data")
result_path = cifar_path.joinpath("results")
eeg_bands = {
    "[4 7]": "θ",
    "[8 12]": "α",
    "[13 30]": "β",
    "[32 60]": "γ",
    "[60 120]": "hγ",
}
# Band names as recorded in matlab structure
band_names = ["theta", "alpha", "beta", "gamma", "hgamma"]
bands = list(eeg_bands.keys())
conditions = ["Rest", "Face", "Place"]
directions = ["BU", "TD"]
fname = "rolling_GC_two_chan.mat"
fpath = result_path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
GC = dataset["GC"]
nbands = len(bands)
ncdt = len(conditions)
ymax = 0.01

# %%

fig, ax = plt.subplots(nbands, ncdt)
for i, band in enumerate(band_names):
    for j, condition in enumerate(conditions):
        F = GC[band][0][0][condition][0][0]["F"][0][0]
        win_time = GC[band][0][0][condition][0][0]["time"][0][0]
        bu = F[0, 1, :]  # bottom up
        td = F[1, 0, :]  # top down
        time = win_time[:, -1]
        ax[i, j].plot(time, bu, label="bu", color="r")
        ax[i, j].plot(time, td, label="td", color="b")
        ax[i, j].set_ylim(0, ymax)
        ax[0, j].set_title(f"{condition}")
        ax[i, 0].set_ylabel(f"{band}")
