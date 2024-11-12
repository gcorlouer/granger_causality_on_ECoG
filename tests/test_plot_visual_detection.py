#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:11:36 2022
Test plotting visual detection function
@author: guime
"""


import mne
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from libs.preprocessing_lib import EcogReader, Epocher
from pathlib import Path
from scipy.stats import linregress

# %% Hierarchical ordering of visual channels

result_path = Path("../results")
fname = "all_visual_channels.csv"
fpath = result_path.joinpath(fname)
df = pd.read_csv(fpath)
# Remove outlier
outlier = "LTm5-LTm6"
df = df[df["chan_name"] != outlier]
# %% Linear regresssion

Y = df["Y"].to_numpy()
latency = df["latency"].to_numpy()
visual_response = df["visual_responsivity"].to_numpy()
category_response = df["category_selectivity"].to_numpy()
Z = df["Z"].to_numpy()
# Compute linear regression between latency,Y, Z visual responsivity and category
# Selectivity
stats_Y = linregress(latency, Y)
stats_Z = linregress(Y, visual_response)
stats_visual = linregress(latency, visual_response)
stats_category = linregress(Y, category_response)
# Make linear slope
max_latency = np.amax(latency)
alatency = np.arange(0, max_latency)
max_Y = np.amax(Y)
min_Y = np.amin(Y)
aY = np.arange(min_Y, max_Y)
# linear slope for latency/Y
l_Y = stats_Y.slope * alatency + stats_Y.intercept
# Plot figure of linear regression
plt.plot(alatency, l_Y)
plt.scatter(latency, Y)
plt.annotate(
    f"r2={round(stats_Y.rvalue,2)}\n p={round(stats_Y.pvalue,3)}",
    xy=(300, -60),
    xycoords="data",
)

# %% Build function to plot linregd

reg = [
    ("Y", "latency"),
    ("Y", "visual_responsivity"),
    ("latency", "visual_responsivity"),
    ("Y", "category_selectivity"),
]


def plot_linreg(reg, fname="all_visual_channels.csv"):
    result_path = Path("../results")
    fpath = result_path.joinpath(fname)
    df = pd.read_csv(fpath)
    # Remove outlier
    outlier = "LTm5-LTm6"
    df = df[df["chan_name"] != outlier]
    for i, pair in enumerate(reg):
        x = df[pair[0]].to_numpy()
        y = df[pair[1]].to_numpy()
        xlabel = pair[0]
        ylabel = pair[1]
        plt.subplot(2, 2, i + 1)
        stats = linregress(x, y)
        xmin = np.amin(x)
        xmax = np.amax(x)
        ax = np.arange(xmin, xmax)
        ay = stats.slope * ax + stats.intercept
        plt.plot(ax, ay, color="r")
        plt.scatter(x, y, color="b")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.annotate(
            f"r2={round(stats.rvalue,2)}\n p={round(stats.pvalue,3)}",
            xy=(0.75, 0.75),
            xycoords="axes fraction",
            fontsize=8,
        )
        plt.tight_layout()


plot_linreg(reg, fname="all_visual_channels.csv")
