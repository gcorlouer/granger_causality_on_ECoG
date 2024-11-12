#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:37:21 2022
In this script we test stochastic dominance of top down vs bottom up GC
@author: guime
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path

# %% Read pair Zscore

fname = "top_down_vs_bottom_up.mat"
path = args.result_path
fpath = path.joinpath(fname)
dataset = loadmat(fpath)
pZ = dataset["pZscore"]

# %% Write plotting function


def plot_top_down_pZscore(pZ, args, vmin=-3, vmax=3, tau_x=0.5, tau_y=0.8):
    conditions = ["Rest", "Face", "Place"]
    cohort = args.cohort
    nsub = len(cohort)
    ncdt = len(conditions)
    vmin = -3
    vmax = 3
    tau_x = 0.5
    tau_y = 0.8
    fig, ax = plt.subplots(nsub, ncdt)
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            # Get statistics from matlab analysis
            z = pZ[subject][0][0][condition][0][0]["zval"][0][0]
            (nR, nF) = z.shape
            zcrit = pZ[subject][0][0][condition][0][0]["zcrit"][0][0]
            sig = pZ[subject][0][0][condition][0][0]["sig"][0][0]
            Rticks = ["R"] * (nR)
            Fticks = ["F"] * (nF)
            # Plot Z score as a heatmap
            g = sns.heatmap(
                z,
                vmin=vmin,
                vmax=vmax,
                cmap="bwr",
                ax=ax[c, s],
                xticklabels=Fticks,
                yticklabels=Rticks,
            )
            g.set_yticklabels(g.get_yticklabels(), rotation=90)
            ax[c, s].xaxis.tick_top()
            ax[c, 0].set_ylabel(f"Z score {condition}")
            # Plot statistical significant entries
            for y in range(z.shape[0]):
                for x in range(z.shape[1]):
                    if sig[y, x] == 1:
                        ax[c, s].text(
                            x + tau_x,
                            y + tau_y,
                            "*",
                            horizontalalignment="center",
                            verticalalignment="center",
                            color="k",
                        )
                    else:
                        continue
        ax[0, s].set_title(f"Subject {s}")
    plt.tight_layout()
    print(f"Critical Z score is {zcrit}")


# %%

sfreq = 500
decim = args.decim
sfreq = sfreq / decim
min_postim = args.tmin_crop
max_postim = args.tmax_crop
print(f"\n Sampling frequency is {sfreq}Hz\n")
print(f"\n Stimulus is during {min_postim} and {max_postim}s\n")

plot_top_down_pZscore(pZ, args, vmin=-3, vmax=3, tau_x=0.5, tau_y=0.8)
