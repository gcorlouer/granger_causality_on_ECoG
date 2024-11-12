#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:09:59 2022

@author: guime
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.io import loadmat
from pathlib import Path

# %%

plt.style.use("ggplot")
fig_width = 28  # figure width in cm
inches_per_cm = 0.393701  # Convert cm to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width * inches_per_cm  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
label_size = 14
tick_size = 12
params = {
    "backend": "ps",
    "lines.linewidth": 0.5,
    "axes.labelsize": label_size,
    "axes.titlesize": label_size,
    "font.size": label_size,
    "legend.fontsize": tick_size,
    "xtick.labelsize": tick_size,
    "ytick.labelsize": tick_size,
    "text.usetex": False,
    "figure.figsize": fig_size,
    "font.weight": "bold",
    "axes.labelweight": "bold",
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
    "[0 62]": "hfa",
}
bands = list(eeg_bands.keys())
conditions = ["Rest", "Face", "Place"]
ncdt = len(conditions)
nbands = len(bands)
ymax = 6
ymin = -6
xticks = []
gc = []
colors = []
fig, ax = plt.subplots(ncdt, 1)
for c, condition in enumerate(conditions):
    xticks = []
    gc = []
    colors = []
    for i, band in enumerate(bands):
        fname = "two_chans_TD_BU_GC_" + band + "Hz.mat"
        path = result_path
        fpath = path.joinpath(fname)
        # Read dataset
        dataset = loadmat(fpath)
        F = dataset["GC"]
        band_name = eeg_bands[band]
        z = F[condition][0][0]["z"][0][0][0][0]
        sig = F[condition][0][0]["sig"][0][0][0][0]
        z_crit_plus = 1.96
        z_crit_minus = -z_crit_plus
        xticks.append(f"{band_name}")
        gc.append(z)  # condition, band specific gc z score
        if z >= 0:
            color = "orange"
        elif z <= 0:
            color = "purple"
        pcrit = F[condition][0][0]["pval"][0][0][0][0]
        zcrit = F[condition][0][0]["zcrit"][0][0][0][0]
        print(f"pval={pcrit}\n")
        print(f"z={z}\n")
        print(f"zcrit={zcrit}\n")
        colors.append(color)
    ax[c].bar(xticks, gc, width=0.1, color=colors)
    ax[c].set_ylim(ymin, ymax)
    ax[c].set_ylabel(f"Z-score, {condition}")
    rects = ax[c].patches
    ax[c].axhline(y=z_crit_plus, color="r")
    ax[c].axhline(y=z_crit_minus, color="r")
    # label = '*'
    # if sig==1:
    #     for rect in rects:
    #         height = rect.get_height()
    #     ax[c].text(
    #         rect.get_x() + rect.get_width() / 2, height + 0.1, label, ha="center", va="bottom"
    #     )
    # else:
    #      continue
