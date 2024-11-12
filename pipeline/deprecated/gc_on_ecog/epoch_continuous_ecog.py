#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:31:38 2022
In this script we epoch continuous ecog and visualise epochs properties

@author: guime
"""

import mne
import matplotlib.pyplot as plt

from src.input_config import args
from src.preprocessing_lib import EcogReader, Epocher

# %% Parameters
channel = ["LTo1-LTo2"]
condition = "Face"
band = "alpha"
freq_bands = {
    "delta": [1, 3],
    "theta": [4, 7],
    "alpha": [8, 12],
    "beta": [13, 30],
    "gamma": [30, 70],
    "high_gamma": [70, 124],
    "spectrum": [0.1, 124],
}

# %% Epoch continuous ecog
reader = EcogReader(
    args.data_path,
    subject=args.subject,
    stage=args.stage,
    preprocessed_suffix=args.preprocessed_suffix,
    epoch=args.epoch,
)
raw = reader.read_ecog()
df_visual = reader.read_channels_info(fname="visual_channels.csv")
visual_chans = df_visual["chan_name"].to_list()
raw = raw.pick_channels(visual_chans)

epocher = Epocher(
    condition=condition,
    t_prestim=args.t_prestim,
    t_postim=args.t_postim,
    baseline=None,
    preload=True,
    tmin_baseline=args.tmin_baseline,
    tmax_baseline=args.tmax_baseline,
    mode=args.mode,
)

epochs = epocher.epoch(raw)
epochs = epochs.filter(l_freq=0.1, h_freq=None)
epochs = epochs.copy().decimate(args.decim)
epochs = epochs.pick(channel)


# %% Filter according to frequency bands

epochs = epochs.copy().filter(l_freq=freq_bands[band][0], h_freq=freq_bands[band][1])

# %% Visualise epochs

epochs.plot_image()

# %% Plot psd

epochs.plot_psd(xscale="log")
# %% Look at individual trial

i = 10
time = epochs.times
X = epochs.get_data()
trial = X[i, :, :]

plt.plot(time, trial[0, :])


# %%
