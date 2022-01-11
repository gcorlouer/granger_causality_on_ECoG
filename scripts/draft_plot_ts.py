#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 22:59:34 2021

@author: guime
"""


import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pathlib import Path, PurePath
from config import args

#%%
#%matplotlib qt
subject = 'DiAs'
fname = subject + '_condition_ts_visual.mat'


ecog = hf.Ecog(args.cohort_path, subject=subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
raw = ecog.read_dataset()

#%%

raw.plot(scalings = 5e-4)

#%% Visual channels to pick

ret = ['LTo1-LTo2']
face = ['LGRD58-LGRD59']

vchan = ['LTo1-LTo2', 'LGRD58-LGRD59']

vraw = raw.copy().pick(vchan)
