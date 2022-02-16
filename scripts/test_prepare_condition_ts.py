#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:03:59 2022

@author: guime
"""


from src.preprocessing_lib import EcogReader, Epocher, parcellation_to_indices
from src.input_config import args
from pathlib import Path
from scipy.io import savemat

import numpy as np
import pandas as pd

#%%

subject = 'DiAs'
path = Path('../results')
fname = subject + '_condition_visual_ts.mat'
fpath = path.joinpath(fname)

