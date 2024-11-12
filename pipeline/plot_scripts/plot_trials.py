#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 21:03:22 2022
We plot trials distribution
@author: guime
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

from math import ceil
from src.preprocessing_lib import EcogReader, Epocher, prepare_condition_ts
from pathlib import Path
from scipy.stats import sem, skew, kurtosis

# %% Plot parameters

plt.style.use("ggplot")
fig_width = 32  # figure width in cm
inches_per_cm = 0.393701  # Convert cm to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width * inches_per_cm  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
label_size = 14
tick_size = 12
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

signal = "LFP"
cohort = ["AnRa", "ArLa", "DiAs"]
cohort_dic = {"AnRa": "Subject 0", "ArLa": "Subject 1", "DiAs": "Subject 2"}
# Signal dic
hfa_dic = {
    "suffix": "_hfb_continuous_raw.fif",
    "signal": "HFA",
    "log_transf": True,
    "decim": 4,
}
lfp_dic = {
    "suffix": "_bad_chans_removed_raw.fif",
    "signal": "LFP",
    "log_transf": False,
    "decim": 2,
}
if signal == "HFA":
    signal_dic = hfa_dic
elif signal == "LFP":
    signal_dic = lfp_dic
# Path to source data, derivatives and results. Enter your own path in local machine
cifar_path = Path("~", "projects", "cifar").expanduser()
data_path = cifar_path.joinpath("data")
derivatives_path = data_path.joinpath("derivatives")
result_path = cifar_path.joinpath("results")
fig_path = Path("~", "thesis", "new_figures").expanduser()

for subject in cohort:
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument("--subject", type=str, default=subject)
    parser.add_argument("--sfeq", type=float, default=500.0)
    parser.add_argument("--stage", type=str, default="preprocessed")
    parser.add_argument("--preprocessed_suffix", type=str, default=signal_dic["suffix"])
    parser.add_argument(
        "--signal", type=str, default=signal_dic["signal"]
    )  # correspond to preprocessed_suffix
    parser.add_argument("--epoch", type=bool, default=False)
    parser.add_argument("--channels", type=str, default="visual_channels.csv")

    # Epoching parameters
    parser.add_argument("--condition", type=str, default="Stim")
    parser.add_argument("--t_prestim", type=float, default=-0.5)
    parser.add_argument("--t_postim", type=float, default=1.5)
    parser.add_argument("--baseline", default=None)  # No baseline from MNE
    parser.add_argument("--preload", default=True)
    parser.add_argument("--tmin_baseline", type=float, default=-0.5)
    parser.add_argument("--tmax_baseline", type=float, default=0)

    # Wether to log transform the data
    parser.add_argument("--log_transf", type=bool, default=signal_dic["log_transf"])
    # Mode to rescale data (mean, logratio, zratio)
    parser.add_argument("--mode", type=str, default="logratio")
    # Pick visual chan
    parser.add_argument("--pick_visual", type=bool, default=True)
    # Create category specific time series
    parser.add_argument("--l_freq", type=float, default=0.1)
    parser.add_argument("--decim", type=float, default=signal_dic["decim"])
    parser.add_argument("--tmin_crop", type=float, default=0)
    parser.add_argument("--tmax_crop", type=float, default=1.5)
    parser.add_argument("--matlab", type=bool, default=False)

    args = parser.parse_args()

    ts = prepare_condition_ts(
        data_path,
        subject=args.subject,
        stage=args.stage,
        matlab=args.matlab,
        preprocessed_suffix=args.preprocessed_suffix,
        decim=args.decim,
        epoch=args.epoch,
        t_prestim=args.t_prestim,
        t_postim=args.t_postim,
        tmin_baseline=args.tmin_baseline,
        tmax_baseline=args.tmax_baseline,
        tmin_crop=args.tmin_crop,
        tmax_crop=args.tmax_crop,
        mode=args.mode,
        log_transf=args.log_transf,
        pick_visual=args.pick_visual,
        channels=args.channels,
    )

    # %% Plot distribution of channels
    conditions = ["Rest", "Face", "Place"]
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    nbins = 100
    # Plot histogram
    for condition in conditions:
        X = ts[condition]
        (n, m, N) = X.shape
        nax = ceil(np.sqrt(n))
        minX = np.min(X)
        maxX = np.max(X)
        xticks = [minX, maxX]
        plt.figure()
        for i in range(n):
            X = ts[condition]
            x = X[i, :, :]
            x = np.ndarray.flatten(x)
            sx = round(skew(x), 2)
            kx = round(kurtosis(x), 2)
            textstr = "\n".join((f"skew={sx}", f"kurt={kx}"))
            ax = plt.subplot(nax, nax, i + 1)
            ax.hist(x, bins=nbins, density=True, color="b", alpha=0.5)
            ax.text(0.03, 0.70, textstr, transform=ax.transAxes, fontsize=tick_size)
            ax.set_xlim((minX, maxX))
            # ax.set_ylim((0,1))
            handles, labels = ax.get_legend_handles_labels()
            ax.set_xlabel(f"{args.signal}, channel {i}")
            ax.set_ylabel("Density")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.suptitle(
            cohort_dic[args.subject]
            + " "
            + f"distributions, visual {signal}, {condition} condition"
        )
        figname = (
            args.subject
            + "_"
            + args.signal.lower()
            + "_"
            + condition
            + "_"
            + "visual_chan_distribution.pdf"
        )
        save_path = fig_path.joinpath(figname)
        plt.savefig(save_path)
# %% Look indivisual trials

# conditions = ['Rest','Face', 'Place']
# time = ts['time']
# X = ts['Face'] # Take condition ts to return N trials to specify nax
# (n,m,N) = X.shape
# # Plot histogram
# nax = ceil(np.sqrt(N))
# for i in range(n):
#     plt.figure()
#     for j in range(N):
#         ax = plt.subplot(nax,nax,j+1)
#         for condition in conditions:
#             X = ts[condition]
#             x = X[i,:,j]
#             ax.plot(time, x)
#         if j==nax**2 - 3:
#             ax.set_xlabel(args.signal + " " + condition + " amplitude (dB)")
#             ax.set_ylabel(args.signal)
#         ax.set_xticks([])
#         ax.set_yticks([])
# figname = args.subject + '_' + args.signal + '_' + condition + '_' + 'visual_chan_distribution.png'
# figpath = result_path.joinpath('figures',figname)
# Extract epochs
# def epoch_ts(data_path, args, condition='Face'):
#        reader = EcogReader(data_path, subject=args.subject, stage=args.stage,
#                                 preprocessed_suffix=args.preprocessed_suffix, preload=True,
#                                 epoch=False)
#        raw = reader.read_ecog()
#        # Read visually responsive channels
#        df_visual = reader.read_channels_info(fname=args.channels)
#        visual_chans = df_visual['chan_name'].to_list()
#        # Pick visually responsive HFA/LFP
#        raw = raw.pick_channels(visual_chans)
#        # Return condition specific epochs
#        epocher = Epocher(condition=condition, t_prestim=args.t_prestim, t_postim =args.t_postim,
#                                baseline=None, preload=True, tmin_baseline=args.tmin_baseline,
#                                tmax_baseline=args.tmax_baseline, mode=args.mode)
#        #Epoch condition specific hfb
#        if args.log_transf == True:
#            epoch = epocher.log_epoch(raw) #log transform to approach Gaussian
#        else:
#            epoch = epocher.epoch(raw)
#        epoch = epoch.copy().crop(tmin =args.tmin_crop, tmax=args.tmax_crop)
#        # Low pass filter
#        epoch = epoch.copy().filter(l_freq=args.l_freq, h_freq=None)
#            # Downsample by factor of 2 and check decimation
#        epoch = epoch.copy().decimate(args.decim)
#        time = epoch.times
#        return epoch
