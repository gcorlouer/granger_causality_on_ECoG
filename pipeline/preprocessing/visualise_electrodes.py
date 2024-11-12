#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:36:28 2022

In this script we visualise the subjects' ECoG electrodes. We first read ECoG raw channels
and visuasile all electrodes positions on a 3D brain obtained from averaging
patients scans with FreeSurfer's fsaverage function. 

Note: I cannot manage to plot with MNE, I think because it requires a trans.fif
file that I do not have. It seems that I do not have the right format to plot
the electrodes on a 3d surface. Maybe it is not too difficult to take the .mat
SUMA freesurfer files that I have but I have no idea how to do it. Oh, well.

I recommand using matlab to visualise electrodes on 3d brain.

@author: guime
"""

import mne

from src.preprocessing_lib import EcogReader
from src.input_config import args
from mne.viz import plot_alignment, snapshot_brain_montage

# %% Read raw ECoG data

path = args.data_path
ecog = EcogReader(path, subject="DiAs", stage="raw_signal")
raw = ecog.read_ecog()

# %% Get montage

montage = raw.get_montage()
print(montage.get_positions()["coord_frame"])

# %%

fig = plot_alignment(raw.info, coord_frame="head")
