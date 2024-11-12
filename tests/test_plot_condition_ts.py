#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:25:40 2022
Plotting condition specific time series accross all subjects
@author: guime
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src.preprocessing_lib import EcogReader, Epocher, prepare_condition_scaled_ts
from pathlib import Path
from scipy.stats import sem
from src.input_config import args


# %%

subject = "DiAs"
conditions = ["Rest", "Face", "Place"]
ts = prepare_condition_scaled_ts(
    args.data_path,
    subject=subject,
    stage="preprocessed",
    matlab=False,
    preprocessed_suffix="_hfb_continuous_raw.fif",
    decim=2,
    epoch=False,
    t_prestim=-0.5,
    t_postim=1.75,
    tmin_baseline=-0.5,
    tmax_baseline=0,
    tmin_crop=-0.5,
    tmax_crop=1.75,
)
populations = ts["indices"].keys()
time = ts["time"]
baseline = ts["baseline"]
baseline = np.average(baseline)


def plot_condition_ts(args, fpath, subject="DiAs"):
    # Prepare condition ts
    ts = prepare_condition_scaled_ts(
        args.data_path,
        subject=subject,
        stage="preprocessed",
        matlab=False,
        preprocessed_suffix="_hfb_continuous_raw.fif",
        decim=2,
        epoch=False,
        t_prestim=-0.5,
        t_postim=1.75,
        tmin_baseline=-0.5,
        tmax_baseline=0,
        tmin_crop=-0.5,
        tmax_crop=1.75,
    )
    populations = ts["indices"].keys()
    time = ts["time"]
    baseline = ts["baseline"]
    baseline = np.average(baseline)
    # Plot condition ts
    f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
    for i, cdt in enumerate(conditions):
        for pop in populations:
            # Condition specific neural population
            X = ts[cdt]
            pop_idx = ts["indices"][pop]
            X = X[pop_idx, :, :]
            X = np.average(X, axis=0)
            # Compute evoked response
            evok = np.average(X, axis=1)
            # Compute confidence interval
            smX = sem(X, axis=1)
            up_ci = evok + 1.96 * smX
            down_ci = evok - 1.96 * smX
            # Plot condition-specific evoked HFA
            ax[i].plot(time, evok, label=pop)
            ax[i].fill_between(time, down_ci, up_ci, alpha=0.6)
            ax[i].axvline(x=0, color="k")
            ax[i].axhline(y=baseline, color="k")
            ax[i].set_ylabel(f"{cdt} (dB)")
            ax[0].legend()
    ax[2].set_xlabel("time (s)")
    plt.tight_layout()
    fname = subject + "_condition.ts.pdf"
    fpath = fpath.joinpath(fname)
    plt.savefig(fpath)


# %%

home = Path.home()
fpath = home.joinpath("thesis", "overleaf_project", "figures")
plot_condition_ts(args, fpath, subject="DiAs")
# %%

f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
for i, cdt in enumerate(conditions):
    for pop in populations:
        # Condition specific neural population
        X = ts[cdt]
        pop_idx = ts["indices"][pop]
        X = X[pop_idx, :, :]
        X = np.average(X, axis=0)
        # Compute evoked response
        evok = np.average(X, axis=1)
        # Compute confidence interval
        smX = sem(X, axis=1)
        up_ci = evok + 1.96 * smX
        down_ci = evok - 1.96 * smX
        # Plot condition specific population time series
        ax[i].plot(time, evok, label=pop)
        ax[i].fill_between(time, down_ci, up_ci, alpha=0.6)
        ax[i].axvline(x=0, color="k")
        ax[i].axhline(y=baseline, color="k")
        ax[i].set_ylabel(f"{cdt} (dB)")
        ax[0].legend()
ax[2].set_xlabel("time (s)")
plt.tight_layout()
