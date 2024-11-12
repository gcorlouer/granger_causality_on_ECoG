#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:48:16 2022
In this script we test function to build pairwise gc dataset.
@author: guime
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from libs.preprocessing_lib import EcogReader, build_dfc, parcellation_to_indices
from libs.input_config import args
from scipy.io import loadmat
from pathlib import Path


# %% Read ROI and functional connectivity data

reader = EcogReader(args.data_path)
# Read visual channels
df_visual = reader.read_channels_info(fname="visual_channels.csv")

# List conditions
conditions = ["Rest", "Face", "Place", "baseline"]

# Load functional connectivity matrix
result_path = Path("../results")

fname = "pairwise_fc.mat"
fc_path = result_path.joinpath(fname)

fc = loadmat(fc_path)
fc = fc["dataset"]

# %% Build dataset fc dictionary

# Shape of functional connectivity dataset.
(ncdt, nsub) = fc.shape
# Flatten array to build dictionarry
fc_flat = np.ndarray.flatten(fc.T)
# Initialise dictionary
fc_dict = {
    "subject": [],
    "condition": [],
    "visual_idx": [],
    "mi": [],
    "sig_mi": [],
    "gc": [],
    "sig_gc": [],
    "smi": [],
    "sgc": [],
    "bias": [],
}
subject = [0] * (ncdt * nsub)
condition = [0] * (ncdt * nsub)
visual_idx = [0] * (ncdt * nsub)
mi = [0] * (ncdt * nsub)
sig_mi = [0] * (ncdt * nsub)
gc = [0] * (ncdt * nsub)
sig_gc = [0] * (ncdt * nsub)
sgc = [0] * (ncdt * nsub)
smi = [0] * (ncdt * nsub)
bias = [0] * (ncdt * nsub)

# Build dictionary
for i in range(ncdt * nsub):
    subject[i] = fc_flat[i]["subject"][0]
    condition[i] = fc_flat[i]["condition"][0]
    # Read visual channels to track visual channels indices
    data_path = Path("../data")
    reader = EcogReader(data_path, subject=subject[i])
    df_visual = reader.read_channels_info(fname="visual_channels.csv")
    visual_idx[i] = parcellation_to_indices(
        df_visual, parcellation="group", matlab=False
    )
    # Multitrial MI
    mi[i] = fc_flat[i]["MI"]
    # MI significance against null
    sig_mi = fc_flat[i]["sigMI"]
    # Multitrial gc
    gc[i] = fc_flat[i]["F"]
    # GC significance against null
    sig_gc[i] = fc_flat[i]["sigF"]
    # Sample MI
    smi[i] = fc_flat[i]["single_MI"]
    # Sample GC
    sgc[i] = fc_flat[i]["single_F"]
    # Bias
    bias[i] = fc_flat[i]["bias"]


fc_dict["subject"] = subject
fc_dict["condition"] = condition
fc_dict["visual_idx"] = visual_idx
fc_dict["mi"] = mi
fc_dict["sig_mi"] = sig_mi
fc_dict["gc"] = gc
fc_dict["sig_gc"] = sig_gc
fc_dict["smi"] = smi
fc_dict["sgc"] = sgc
fc_dict["bias"] = bias

# Build dataframe
dfc = pd.DataFrame.from_dict(fc_dict)
