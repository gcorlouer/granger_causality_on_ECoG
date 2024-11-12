#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:25:44 2022
In this script we plot pairwise unconditional GC between all channels
@author: guime
"""
from src.preprocessing_lib import EcogReader, parcellation_to_indices
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# %%

plt.style.use("ggplot")
fig_width = 25  # figure width in cm
inches_per_cm = 0.393701  # Convert cm to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width * inches_per_cm  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
params = {
    "backend": "ps",
    "lines.linewidth": 1.5,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "font.size": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 15,
    "text.usetex": False,
    "figure.figsize": fig_size,
}
plt.rcParams.update(params)

# %% Read channels

cifar_path = Path("~", "projects", "cifar").expanduser()
data_path = cifar_path.joinpath("data")
subject = "DiAs"
brain_path = data_path.joinpath("derivatives", subject, "brain")
fname = "unsorted_bp_channels.csv"
fpath = brain_path.joinpath(fname)

reader = EcogReader(data_path, subject=subject)
df = reader.read_channels_info(fname="BP_channels.csv")
df = df.sort_values(by=["Unnamed: 0"])
df = df.rename(columns={"Unnamed: 0": "Index"})
df = df.drop(columns="Index")
df = df.reset_index(drop=True)
df.to_csv(fpath)

chan_path = fpath


# Read dataset
result_path = cifar_path.joinpath("results")
fname = "unconditional_GC.mat"
path = result_path
fpath = path.joinpath(fname)
dataset = loadmat(fpath)
GC = dataset["GC"]

## Read bad channels
fname = "all_bad_channels.csv"
bads_path = result_path.joinpath(fname)
df_bads = pd.read_csv(bads_path)
df_bads = df_bads.loc[df_bads["subject"] == subject]
bad_channels = df_bads["bad_channel"].tolist()
# %% Plot pairwise unconditional GC


def plot_unconditional_GC(GC, condition="Face", vmin=-5, vmax=5):
    # Get visual channels
    # reader = EcogReader(data_path, subject=subject)
    # df = reader.read_channels_info(fname='unsorted_bp_channels.csv')
    # df_sorted = df = df.sort_values(by = "Y")
    # ls = df_sorted.index.tolist()
    # sorted_chan = df_sorted['electrode_name'].tolist()
    # nchan = len(sorted_chan)
    # ordered_F = np.zeros((nchan,nchan))
    # Get GC from matlab analysis
    F = GC[subject][0][0][condition][0][0]["F"][0][0]
    # Plot Z score as a heatmap
    # Hierarchical ordering
    # for ix, i in enumerate(ls):
    #    for jx, j in enumerate(ls):
    #        ordered_F[ix,jx] = F[i,j]
    # F = ordered_F
    # Get statistics from matlab analysis
    plt.figure()
    sns.heatmap(F, vmin=vmin, vmax=vmax, cmap="YlOrBr")


# %%

plot_unconditional_GC(GC, condition="Rest", vmin=0, vmax=0.01)
