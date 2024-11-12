#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:59:39 2021
This script plot var estimation for all subjects. 
@author: guime
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata as ud

from pathlib import Path
from src.input_config import args


# %%

result_path = Path("..", "results")
fname = "rolling_var_ecog_estimation.csv"
fpath = Path.joinpath(result_path, fname)
df = pd.read_csv(fpath)

# %% Plot varmodel order

# %matplotlib qt

cohort = args.cohort
nsub = len(cohort)
ncdt = 3
ic = ["aic", "bic", "hqc", "lrt"]
cdt = list(df["condition"].unique())
ymax = 20

f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)

for c in range(ncdt):
    for s in range(nsub):
        for i in ic:
            time = (
                df["time"]
                .loc[(df["subject"] == cohort[s]) & (df["condition"] == cdt[c])]
                .to_numpy()
            )
            morder = (
                df[i]
                .loc[(df["subject"] == cohort[s]) & (df["condition"] == cdt[c])]
                .to_numpy()
            )
            ax[c, s].plot(time, morder, label=i)
            ax[c, s].set_ylim(0, ymax)
            ax[0, s].set_title(f"Subject {s}", fontsize=20)
            ax[c, 0].set_ylabel(cdt[c], fontsize=22)
            ax[2, s].set_xlabel("Time (s)", fontsize=22)
            ax[c, s].tick_params(labelsize=20)

handles, labels = ax[c, s].get_legend_handles_labels()
f.legend(handles, labels, loc="upper right")
plt.tight_layout()
fpath = Path("~", "thesis", "overleaf_project", "figures", "method_figure").expanduser()
fname = "rolling_var.png"
figpath = fpath.joinpath(fname)
plt.savefig(figpath)
# %% Plot Spectral radius

f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)

# ypos = np.arange(0.95, 1, 0.01)
for c in range(ncdt):
    for s in range(nsub):
        time = (
            df["time"]
            .loc[(df["subject"] == cohort[s]) & (df["condition"] == cdt[c])]
            .to_numpy()
        )
        rho = (
            df["rho"]
            .loc[(df["subject"] == cohort[s]) & (df["condition"] == cdt[c])]
            .to_numpy()
        )
        ax[c, s].plot(time, rho)
        ax[c, s].set_ylim(0.8, 1)
        ax[0, s].set_title(f"Subject {s}", fontsize=20)
        ax[c, 0].set_ylabel(f"{cdt[c]}", fontsize=22)
        # ax[c,0].set_yticks(ypos)
        ax[2, s].set_xlabel("Time (s)", fontsize=22)
        ax[c, s].tick_params(labelsize=20)

handles, labels = ax[c, s].get_legend_handles_labels()
f.legend(handles, labels, loc="upper right")
f.suptitle("Rolling spectral radius", fontsize=22)
plt.tight_layout()
fname = "rolling_specrad.png"
figpath = fpath.joinpath(fname)
plt.savefig(figpath)
