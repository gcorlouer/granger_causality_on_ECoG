#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:22:32 2021

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
from scipy.io import savemat

#%% 

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                           stage = args.stage, epoch=args.epoch)

hfb = ecog.read_dataset()

#%%
#%matplotlib qt

hfb.plot()