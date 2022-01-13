#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 00:58:02 2022

@author: guime
"""

from src.preprocessing_lib import EcogReader, HfbEpocher
from src.input_config import args

#import matplotlib.pyplot as plt

#%% Test epoching

chans = ['LTo1-LTo2', 'LTo5-LTo6', 'LGRD58-LGRD59', 'LGRD60-LGRD61']
def test_epoch_hfb(chans):
    reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix)
    epocher = HfbEpocher()
    hfb = reader.read_ecog()
    hfb = epocher.epoch_hfb(hfb)
    hfb = hfb.pick_channels(chans)
    hfb.plot()

test_epoch_hfb(chans)
#%% Test baseline rescaled

#%% Test log transformation

