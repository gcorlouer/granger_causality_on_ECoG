#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:44:20 2022

@author: guime
"""

from src.preprocessing_lib import EcogReader, VisualClassifier, VisualDetector
from src.input_config import args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%%
#%matplotlib qt
cohort = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
def test_classify_visual_chans():
    reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
    hfb = reader.read_ecog()
    dfelec = reader.read_channels_info(fname='electrodes_info.csv')
    classifier = VisualClassifier(tmin_prestim=args.tmin_prestim, 
                              tmax_prestim=args.tmax_prestim, 
                              tmin_postim=args.tmin_postim,
               tmax_postim=args.tmax_postim, alpha=args.alpha, 
               zero_method=args.zero_method, alternative=args.alternative)
    visual_populations = classifier.classify_visual_chans(hfb, dfelec)
    return visual_populations

visual_populations = test_classify_visual_chans()

#%% 

conditions = ['Face', 'Place']
visual_chan = visual_populations['chan_name']
group = visual_populations['group']
for condition in conditions:
    suffix = '_hfb_'+ condition + '_scaled-epo.fif'
    reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                     preprocessed_suffix=suffix, epoch=args.epoch)
    hfb = reader.read_ecog()
    dfelec = reader.read_channels_info(fname='electrodes_info.csv')
    
    hfb_visual= hfb.copy().pick_channels(visual_chan)
    X = hfb_visual.copy().get_data()
    X = np.mean(X,axis=0)
    (nchan, nobs)=X.shape
    time =  hfb_visual.times
    for j in range(nchan):
        plt.subplot(4,4,j+1)
        plt.plot(time, X[j,:], label=condition)
        plt.title(f'{visual_chan[j]}, {group[j]}')
        plt.axis('off')
plt.suptitle(f'{args.subject} visual HFA')
plt.legend()


#%% test classify face and place

def test_classify_face_place():
    reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
    hfb = reader.read_ecog()
    visual_channels = visual_populations['chan_name']
    classifier = VisualClassifier(tmin_prestim=args.tmin_prestim, 
                              tmax_prestim=args.tmax_prestim, 
                              tmin_postim=args.tmin_postim,
               tmax_postim=args.tmax_postim, alpha=args.alpha, 
               zero_method=args.zero_method, alternative=args.alternative)
    event_id = hfb.event_id
    face_id = classifier.extract_stim_id(event_id, cat = 'Face')
    print(face_id)
    place_id = classifier.extract_stim_id(event_id, cat='Place')
    print(place_id)
    hfb = hfb.pick_channels(visual_channels)
    group, category_selectivity = classifier.classify_face_place(hfb, 
                                                                 face_id, place_id, 
                                                                 visual_channels)
    return group, category_selectivity

group, category_selectivity = test_classify_face_place()

#%% Test Wilcoxon statistical test

detector = VisualDetector(tmin_prestim=args.tmin_prestim, 
                              tmax_prestim=args.tmax_prestim, 
                              tmin_postim=args.tmin_postim,
               tmax_postim=args.tmax_postim, alpha=args.alpha, 
               zero_method=args.zero_method, alternative=args.alternative)
reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
classifier = VisualClassifier(tmin_prestim=args.tmin_prestim, 
                              tmax_prestim=args.tmax_prestim, 
                              tmin_postim=args.tmin_postim,
               tmax_postim=args.tmax_postim, alpha=args.alpha, 
               zero_method=args.zero_method, alternative=args.alternative)
visual_channels = visual_populations['chan_name']
hfb = reader.read_ecog()
hfb = hfb.copy().pick_channels(visual_channels)
event_id = hfb.event_id
face_id = classifier.extract_stim_id(event_id, cat = 'Face')
print(face_id)
place_id = classifier.extract_stim_id(event_id, cat='Place')
print(place_id)
visual_channels = visual_populations['chan_name']
# Where the problem lies is A_place !
A_face = detector.crop_stim_hfb(hfb, face_id, tmin=args.tmin_postim, tmax=args.tmax_postim)
A_place = detector.crop_stim_hfb(hfb, place_id, tmin=args.tmin_postim, tmax=args.tmax_postim)
w_test_face = detector.multiple_wilcoxon_test(A_face, A_place)
reject_face = w_test_face[0]    

w_test_place = detector.multiple_wilcoxon_test(A_place, A_face)
reject_place = w_test_place[0]    
