#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:30:45 2022

@author: guime
"""

from src.input_config import args
from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.plotting_lib import plot_multitrial_rolling_fc
from pathlib import Path
from scipy.io import loadmat

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# %%

home = Path.home()
result_path = Path("../results")
# List conditions
conditions = ["Rest", "Face", "Place", "baseline"]
cohort = ["AnRa", "ArLa", "DiAs"]
nsub = len(args.cohort)

# %%
# Set ymax
top = [6, 10, 10]
# Load functional connectivity matrix
fname = "rolling_multi_trial_fc.mat"
fc_path = result_path.joinpath(fname)
fc = loadmat(fc_path)
fc = fc["dataset"]
figname = "rolling_multi_trial_fc.png"
figpath = home.joinpath("PhD", "notes", "figures")
figpath = figpath.joinpath(figname)
plot_multitrial_rolling_fc(fc, figpath, top, F="gGC")
