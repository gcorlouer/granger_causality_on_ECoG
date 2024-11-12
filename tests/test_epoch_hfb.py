#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 00:58:02 2022

@author: guime
"""

from libs.preprocessing_lib import EcogReader, Epocher
from libs.input_config import args

import mne
import numpy as np
import matplotlib.pyplot as plt

# %% Test epoching
# %matplotlib qt
chans = ["LTo1-LTo2", "LTo5-LTo6", "LGRD58-LGRD59", "LGRD60-LGRD61"]


def test_epoch(chans):
    reader = EcogReader(
        args.data_path,
        subject=args.subject,
        stage=args.stage,
        preprocessed_suffix=args.preprocessed_suffix,
    )
    epocher = Epocher(
        condition="Stim",
        t_prestim=args.t_prestim,
        t_postim=args.t_postim,
        baseline=args.baseline,
        preload=args.preload,
        tmin_baseline=args.tmin_baseline,
        tmax_baseline=args.tmax_baseline,
        mode=args.mode,
    )
    hfb = reader.read_ecog()
    hfb = epocher.epoch(hfb)
    hfb.plot_image()
    X = hfb.copy().get_data()
    print(f"HFA has shape {X.shape}")
    print(f"HFb info for {args.condition}: {hfb}")
    hfb = hfb.pick_channels(chans)
    hfb.plot()
    time = hfb.times
    plt.figure()
    X = hfb.copy().get_data()
    X = np.mean(X, axis=0)
    for i in range(len(chans)):
        plt.plot(time, X[i, :])


test_epoch(chans)


# %% Test condition id extraction


def test_extract_condition_id():
    reader = EcogReader(
        args.data_path,
        subject=args.subject,
        stage=args.stage,
        preprocessed_suffix=args.preprocessed_suffix,
    )
    epocher = Epocher(
        condition=args.condition,
        t_prestim=args.t_prestim,
        t_postim=args.t_postim,
        baseline=args.baseline,
        preload=args.preload,
        tmin_baseline=args.tmin_baseline,
        tmax_baseline=args.tmax_baseline,
        mode=args.mode,
    )
    hfb = reader.read_ecog()
    events, events_id = mne.events_from_annotations(hfb)
    condition_id = epocher.extract_condition_id(events_id)
    print(condition_id)


test_extract_condition_id()


# %% Test Rescaling
def test_scale_epoch():
    reader = EcogReader(
        args.data_path,
        subject=args.subject,
        stage=args.stage,
        preprocessed_suffix=args.preprocessed_suffix,
    )
    epocher = Epocher(
        condition="Face",
        t_prestim=args.t_prestim,
        t_postim=args.t_postim,
        baseline=args.baseline,
        preload=args.preload,
        tmin_baseline=args.tmin_baseline,
        tmax_baseline=args.tmax_baseline,
        mode=args.mode,
    )
    hfb = reader.read_ecog()
    hfb = epocher.scale_epoch(hfb)
    hfb.plot_image(scalings=1e6)
    hfb.plot_image()
    hfb = hfb.pick_channels(chans)
    hfb.plot(scalings=1e1)
    time = hfb.times
    plt.figure()
    X = hfb.copy().get_data()
    X = np.mean(X, axis=0)
    for i in range(len(chans)):
        plt.plot(time, X[i, :])


test_scale_epoch()

# %% Test log transform


def test_log_epoch():
    reader = EcogReader(
        args.data_path,
        subject=args.subject,
        stage=args.stage,
        preprocessed_suffix=args.preprocessed_suffix,
    )
    epocher = Epocher(
        condition="Face",
        t_prestim=args.t_prestim,
        t_postim=args.t_postim,
        baseline=args.baseline,
        preload=args.preload,
        tmin_baseline=args.tmin_baseline,
        tmax_baseline=args.tmax_baseline,
        mode=args.mode,
    )
    hfb = reader.read_ecog()
    hfb = epocher.log_epoch(hfb)
    hfb.plot_image()
    hfb = hfb.pick_channels(chans)
    hfb.plot(scalings=1e1)
    time = hfb.times
    plt.figure()
    X = hfb.copy().get_data()
    X = np.mean(X, axis=0)
    for i in range(len(chans)):
        plt.plot(time, X[i, :])


test_log_epoch()
