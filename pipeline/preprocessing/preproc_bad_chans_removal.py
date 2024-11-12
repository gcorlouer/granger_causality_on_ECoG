#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:33:20 2021
This script concatenate ieeg dataset and remove bad channels for all subjects.
It also saves all bad channels into a csv file 
@author: guime
"""
from src.preprocessing_lib import EcogReader, drop_bad_chans
from src.input_config import args
from pathlib import Path

import pandas as pd

# %%

# Make sure you are in the "/scripts" directory
derivatives_path = args.derivatives_path
result_path = args.result_path
cohort = args.cohort
data_path = args.path
stage = args.stage

# %% Save bad channels name in a dataframe

columns = ["bad_channel", "subject"]
df = pd.DataFrame(columns=columns)
for subject in cohort:
    # Read bipolar montage data
    ecog = EcogReader(
        data_path, stage=stage, subject=subject, preload=True, epoch=False
    )
    raw = ecog.concatenate_condition()
    # Drop bad channels
    raw, bad_chans = drop_bad_chans(raw, q=99, voltage_threshold=500e-6, n_std=5)
    # Append bad channels to dataframe
    subject_list = [subject] * len(bad_chans)
    dfi = pd.DataFrame({"bad_channel": bad_chans, "subject": subject_list})
    df = df.append(dfi, ignore_index=True)
    # Save dataset with bad channels removed
    fname = subject + "_bad_chans_removed_raw.fif"
    fpath = derivatives_path.joinpath(subject).joinpath("ieeg").joinpath(fname)
    print(fpath)
    raw.save(fpath, overwrite=True)

# %%

# Save all bad channels into csv file

fname = "all_bad_channels.csv"
fpath = result_path.joinpath(fname)
df.to_csv(fpath)
