#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:33:29 2022

@author: guime
"""


from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.plotting_lib import info_flow_stat
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# %%

plt.style.use("ggplot")
fig_width = 26  # figure width in cm
inches_per_cm = 0.393701  # Convert cm to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width * inches_per_cm  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
label_size = 9
tick_size = 8
params = {
    "backend": "ps",
    "lines.linewidth": 1.5,
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

cohort = ["AnRa", "ArLa", "DiAs"]
signal = "lfp"
# Useful paths
cifar_path = Path("~", "projects", "cifar").expanduser()
data_path = cifar_path.joinpath("data")
result_path = cifar_path.joinpath("results")
fname = "_".join(["spectral", "group", "GC", f"{signal}.mat"])
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
F = dataset["sGC"]


# %%


def plot_condition_sgc(F):
    """
    Plot spectral GC across subjects and directions to compare between conditions
    """
    conditions = ["Rest", "Face", "Place"]
    populations = ["R", "O", "F"]
    pop_dic = {"R": "Retinotopic", "O": "Other", "F": "Face"}
    pair_dic = {"R->F": [0, 2], "F->R": [2, 0], "R->R": [0, 0], "F->F": [2, 2]}
    npair = len(list(pair_dic.keys()))
    ncdt = len(conditions)
    nsub = len(cohort)
    npop = len(populations)
    color = ["r", "b", "g"]
    fig, ax = plt.subplots(npair, nsub, sharex=False, sharey=False)
    for s, subject in enumerate(cohort):
        indices = F[subject][0][0]["indices"][0][0]
        for p, pair in enumerate(list(pair_dic.keys())):
            for c, condition in enumerate(conditions):
                f = F[subject][0][0][condition][0][0]
                freqs = F["freqs"][0][0]
                i = pair_dic[pair][1]
                j = pair_dic[pair][0]
                ax[p, s].plot(freqs, f[i, j, :], label=f"{condition}", color=color[c])
                ax[p, s].set_xscale("linear")
                ax[p, s].set_yscale("log")
                ax[c, s].set_ylim(0.00001, 1)
                ax[p, 0].set_ylabel(f"Spectral GC \n  {pair}")
                ax[-1, s].set_xlabel("Frequency (Hz)")
                ax[0, s].set_title(f"Subject {s}")
    handles, labels = ax[p, s].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")


def plot_direction_sgc(F):
    """
    Plot spectral GC across subjects and directions to compare between conditions
    """
    conditions = ["Rest", "Face", "Place"]
    populations = ["R", "O", "F"]
    pop_dic = {"R": "Retinotopic", "O": "Other", "F": "Face"}
    pair_dic = {"R->F": [0, 2], "F->R": [2, 0]}
    npair = len(list(pair_dic.keys()))
    ncdt = len(conditions)
    nsub = len(cohort)
    npop = len(populations)
    color = ["r", "b"]
    fig, ax = plt.subplots(ncdt, nsub, sharex=False, sharey=False)
    for s, subject in enumerate(cohort):
        indices = F[subject][0][0]["indices"][0][0]
        for c, condition in enumerate(conditions):
            for p, pair in enumerate(list(pair_dic.keys())):
                f = F[subject][0][0][condition][0][0]
                freqs = F["freqs"][0][0]
                i = pair_dic[pair][1]
                j = pair_dic[pair][0]
                ax[c, s].plot(freqs, f[i, j, :], label=f"{pair}", color=color[p])
                ax[c, s].set_xscale("linear")
                ax[c, s].set_yscale("log")
                ax[c, s].set_ylim(0.00001, 1)
                ax[-1, s].set_xlabel("Frequency (Hz)")
                ax[c, 0].set_ylabel(f"Spectral GC \n {condition}")
                ax[0, s].set_title(f"Subject {s}")
    handles, labels = ax[p, s].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")


# %%

plot_condition_sgc(F)
figpath = Path(
    "~", "thesis", "overleaf_project", "figures", "results_figures"
).expanduser()
fname = "_".join(["spectral_gc", signal, "condition.pdf"])
figpath = figpath.joinpath(fname)
plt.savefig(figpath)
# %%

plot_direction_sgc(F)
figpath = Path(
    "~", "thesis", "overleaf_project", "figures", "results_figures"
).expanduser()
fname = "_".join(["spectral_gc", signal, "direction.pdf"])
figpath = figpath.joinpath(fname)
plt.savefig(figpath)

# %%
