#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:29:20 2022
In this script we run cross subjects time frequency analysis
@author: guime
"""

#%%
import argparse
import numpy as np
import pandas as pd

from src.input_config import args
from src.time_frequency import compute_group_power

#%% Parameters
cohort = args.cohort
conditions = ["Rest", "Face", "Place"]
groups = ["R", "O", "F"]
ngroup = len(groups)
ncdt = len(conditions)
power_dict = {"subject": [], "condition": [], "group": [], "power": [],
              "time": [], "freqs": []}
# Command arguments
parser = argparse.ArgumentParser()
# Frequency space
parser.add_argument("--sfreq", type=float, default=500 / args.decim)
parser.add_argument("--nfreqs", type=float, default=2**9)
parser.add_argument("--fmin", type=float, default=0.5)
parser.add_argument("--l_freq", type=float, default=0.01)
# Baseline correction
# tf_mode and tf_baseline to not confuse with mode and baseline of epoching
parser.add_argument("--mode", type=str, default="zscore")
parser.add_argument("--tmin_baseline", type=float, default=-0.5)
parser.add_argument("--tmax_baseline", type=float, default=0)
tf_args = parser.parse_args()

#%% For a given subject, run time frequency analysis across groups and
# conditions


def cross_tf_analysis(args, tf_args):
    for subject in args.cohort:
        for group in groups:
            for condition in conditions:
                power, time, freqs = compute_group_power(
                    args, tf_args, subject=subject, group=group, condition=condition
                )
                power_dict["subject"].append(subject)
                power_dict["condition"].append(condition)
                power_dict["group"].append(group)
                power_dict["power"].append(power)
                power_dict["freqs"].append(freqs)
                power_dict["time"].append(time)
    df_power = pd.DataFrame(power_dict)
    return df_power


def main():
    fname = "tf_power_dataframe.pkl"
    df_power = cross_tf_analysis(args, tf_args)
    fpath = args.transfer_path
    fpath = fpath.joinpath(fname)
    df_power.to_pickle(fpath)


if __name__ == "__main__":
    main()
