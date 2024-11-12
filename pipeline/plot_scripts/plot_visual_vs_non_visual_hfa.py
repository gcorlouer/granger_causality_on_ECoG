#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 21:31:36 2022

@author: guime
"""


from src.preprocessing_lib import EcogReader, Epocher
from src.input_config import args

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# %%
# Read visual chans
for i, subject in enumerate(args.cohort):
    reader = EcogReader(
        args.data_path,
        subject=subject,
        stage=args.stage,
        preprocessed_suffix=args.preprocessed_suffix,
        epoch=args.epoch,
    )
    df_visual = reader.read_channels_info(fname="visual_channels.csv")
    visual_chans = df_visual["chan_name"].to_list()
    category = df_visual["group"].to_list()
    location = df_visual["DK"].to_list()

    # Read hfb

    hfb = reader.read_ecog()

    # epocher = Epocher()
    # hfb = epocher.epoch(hfb)
    hfb_visual = hfb.copy().pick_channels(visual_chans)
    hfb_nv = hfb.copy().drop_channels(visual_chans)
    baseline = hfb_visual.copy().crop(tmin=-0.5, tmax=0).get_data()
    baseline = np.average(baseline)

    # %% Plot event related potential of visual channels

    evok_visual = hfb_visual.average()

    # %% Plot event related potential of non visual channels

    evok_nv = hfb_nv.average()

    time = evok_visual.times
    # %%

    X = evok_visual.get_data()
    mX = np.mean(X, 0)
    Y = evok_nv.get_data()
    mY = np.mean(Y, 0)
    plt.subplot(3, 3, i + 1)
    plt.plot(time, mX, label="visual")
    plt.plot(time, mY, label="non visual")
    plt.axhline(y=baseline)
    plt.axvline(x=0)
    plt.xlabel("Time (s)")
    plt.ylabel(f"HFA {subject}")
    plt.legend()
