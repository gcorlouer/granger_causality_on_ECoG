#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:31:56 2022

@author: guime
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.io import loadmat
from pathlib import Path

# %%
plt.style.use("ggplot")
fig_width = 20  # figure width in cm
inches_per_cm = 0.393701  # Convert cm to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width * inches_per_cm  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
label_size = 12
tick_size = 8
params = {
    "backend": "ps",
    "lines.linewidth": 1.2,
    "axes.labelsize": label_size,
    "axes.titlesize": label_size,
    "font.size": label_size,
    "legend.fontsize": tick_size,
    "xtick.labelsize": tick_size,
    "ytick.labelsize": tick_size,
    "text.usetex": False,
    "figure.figsize": fig_size,
}
plt.rcParams.update(params)


# %%

cohort = ["AnRa", "ArLa", "DiAs"]
# Useful paths
cifar_path = Path("~", "projects", "cifar").expanduser()
data_path = cifar_path.joinpath("data")
result_path = cifar_path.joinpath("results")
signal = "lfp"
if signal == "lfp":
    infocrit = "bic"
elif signal == "hfa":
    infocrit = "hqc"
fname = signal + "_model_order_estimation.m"
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
model_order = dataset["ModelOrder"]

# %%


def plot_var_model_order(model_order, cohort, infocrit="aic"):
    conditions = ["Rest", "Face", "Place"]
    nsub = len(cohort)
    ncdt = len(conditions)
    fig, ax = plt.subplots(ncdt, nsub)
    # Loop over subject and comparison to plot varmodel order
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            varmo = model_order[subject][0][0][condition][0][0]["varmo"][0][0]
            rho = model_order[subject][0][0][condition][0][0]["rho"][0][0][0][0]
            rho = round(rho, 2)
            morder = varmo[infocrit][0][0]
            saic = varmo["saic"][0][0]
            sbic = varmo["sbic"][0][0]
            shqc = varmo["shqc"][0][0]
            lags = varmo["lags"][0][0]
            ax[c, s].plot(lags, saic, color="r", label="aic")
            ax[c, s].plot(lags, shqc, color="b", label="hqc")
            ax[c, s].plot(lags, sbic, color="g", label="bic")
            ax[c, s].axvline(x=morder, color="k")
            ax[c, s].annotate(
                r"$\rho$" + f"={rho}",
                xy=(0.50, 0.50),
                xycoords="axes fraction",
                fontsize=12,
            )
            ticks_labels = [0, 5, 10, 15, 20, 25, 30]
            ax[0, s].set_title(f"Subject {s}, {signal}")
            ax[-1, s].set_xlabel("Lags (observations)")
            # if c<=1:
            #             ax[c,s].set_xticks([]) # (turn off xticks)
            # if s>=1:
            #             ax[c,s].set_yticks([]) # (turn off xticks)
            ax[c, s].set_xticks([0, 5, 10, 15, 20, 25, 30])
            ax[c, s].set_xticklabels(ticks_labels)
            ax[c, 0].set_ylabel(f"Information \n criterion \n {condition}")
    # ax[-1,-1].legend()
    handles, labels = ax[c, s].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")


def plot_ss_model_order(model_order, cohort, xmax=50):
    conditions = ["Rest", "Face", "Place"]
    nsub = len(cohort)
    ncdt = len(conditions)
    fig, ax = plt.subplots(ncdt, nsub)
    # Loop over subject and comparison to plot varmodel order
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            ssmo = model_order[subject][0][0][condition][0][0]["ssmo"][0][0]
            morder = ssmo["mosvc"][0][0][0][0]
            lags = ssmo["lags"][0][0]
            ssvc = ssmo["ssvc"][0][0]
            ax[c, s].plot(lags, ssvc, color="b")
            ax[c, s].axvline(x=morder, color="k")
            ticks_labels = [0, 10, 20, 30, 40, 50]
            ax[0, s].set_title(f"Subject {s}, {signal}")
            ax[-1, s].set_xlabel("Lags (observations)")
            # if c<=1:
            #             ax[c,s].set_xticks([]) # (turn off xticks)
            # if s>=1:
            #             ax[c,s].set_yticks([]) # (turn off xticks)
            ax[c, s].set_xlim(0, xmax)
            ax[c, s].set_xticks(ticks_labels)
            ax[c, s].set_xticklabels(ticks_labels)
            ax[c, 0].set_ylabel(f"Singular \n value criterion \n {condition}")
    # ax[-1,-1].legend()
    handles, labels = ax[c, s].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")


# %% Plot var model order estimation

plot_var_model_order(model_order, cohort, infocrit=infocrit)
fpath = Path("~", "thesis", "overleaf_project", "figures", "method_figure").expanduser()
fname = signal + "_varmorder_multi_trial_corrected.pdf"
figpath = fpath.joinpath(fname)
plt.savefig(figpath)


# %% Plot ss model order estimation

plot_ss_model_order(model_order, cohort)
fpath = Path("~", "thesis", "overleaf_project", "figures", "method_figure").expanduser()
fname = signal + "_ssmorder_multi_trial_corrected.pdf"
figpath = fpath.joinpath(fname)
plt.savefig(figpath)
