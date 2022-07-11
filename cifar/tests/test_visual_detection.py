#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 22:44:37 2022

@author: guime
"""


from src.preprocessing_lib import EcogReader, VisualDetector
from src.input_config import args

import matplotlib.pyplot as plt
import numpy as np
#%%  Test visual detection
cohort = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

#%matplotlib qt
def test_detect():
    reader = EcogReader(args.data_path, subject= 'DiAs', stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
    hfb = reader.read_ecog()
    detector = VisualDetector(tmin_prestim=args.tmin_prestim, 
                              tmax_prestim=args.tmax_prestim, 
                              tmin_postim=args.tmin_postim,
               tmax_postim=args.tmax_postim, alpha=args.alpha, 
               zero_method=args.zero_method, alternative=args.alternative)
    visual_chan, effect_size = detector.detect_visual_chans(hfb)
    print(f"Visual responsive channels: {visual_chan}")
    hfb_visual = hfb.copy().pick_channels(visual_chan)
    time = hfb_visual.times
    X = hfb_visual.get_data()
    X = np.mean(X,axis=0)
    for i in range(len(visual_chan)):
        plt.plot(time, X[i,:])
    plt.figure()
    hfb = hfb.drop_channels(visual_chan)
    X = hfb.copy().get_data()
    X = np.mean(X, axis=0)
    (nchan, nobs) = X.shape
    for i in range(nchan):
        plt.plot(time, X[i,:])
    
    # Plot response for channel who are not visually responsive
    
    
    
test_detect()

#%%

