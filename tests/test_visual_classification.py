#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:44:20 2022

@author: guime
"""

from src.preprocessing_lib import EcogReader, VisualClassifier, VisualDetector
from pathlib import Path 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

#%% Parameters
cohort = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

# Path to source data, derivatives and results. Enter your own path in local machine
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
derivatives_path = data_path.joinpath('derivatives')
result_path = cifar_path.joinpath('results')
fig_path = cifar_path.joinpath('results/figures')

parser = argparse.ArgumentParser()
# Paths
parser.add_argument("--data_path", type=list, default=data_path)
parser.add_argument("--derivatives_path", type=list, default=derivatives_path)
parser.add_argument("--result_path", type=list, default=result_path)
parser.add_argument("--fig_path", type=list, default=result_path)

# Dataset parameters 
parser.add_argument("--cohort", type=list, default=cohort)
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--sfeq", type=float, default=500.0)

parser.add_argument("--stage", type=str, default='preprocessed')
parser.add_argument("--preprocessed_suffix", type=str, default= '_hfb_Stim_scaled-epo.fif')
parser.add_argument("--epoch", type=bool, default=True)
parser.add_argument("--channels", type=str, default='visual_channels.csv')
parser.add_argument("--matlab", type=bool, default=False)

#% Visually responsive channels classification parmeters

parser.add_argument("--tmin_prestim", type=float, default=-0.4)
parser.add_argument("--tmax_prestim", type=float, default=-0.05)
parser.add_argument("--tmin_postim", type=float, default=0.05)
parser.add_argument("--tmax_postim", type=float, default=0.4)
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--zero_method", type=str, default='wilcox')
parser.add_argument("--alternative", type=str, default='greater')

#% Create category specific time series

parser.add_argument("--decim", type=float, default=2)
parser.add_argument("--tmin_crop", type=float, default=-0.5)
parser.add_argument("--tmax_crop", type=float, default=1.5)

#% Functional connectivity parameters

parser.add_argument("--nfreq", type=float, default=1024)
parser.add_argument("--roi", type=str, default="functional")

args = parser.parse_args()
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
               alternative=args.alternative)
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

#%% Test detection and classification
subject = 'DiAs'
detector = VisualDetector(tmin_prestim=args.tmin_prestim, 
                              tmax_prestim=args.tmax_prestim, 
                              tmin_postim=args.tmin_postim,
               tmax_postim=args.tmax_postim, alpha=args.alpha, 
               zero_method=args.zero_method, alternative=args.alternative)
reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
classifier = VisualClassifier(tmin_prestim=args.tmin_prestim, 
                              tmax_prestim=args.tmax_prestim, 
                              tmin_postim=args.tmin_postim,
               tmax_postim=args.tmax_postim, alpha=args.alpha, 
               zero_method=args.zero_method, alternative=args.alternative)

hfb = reader.read_ecog()
visual_chan, effect_size = detector.detect_visual_chans(hfb)

#%% Test multiple wilcoxon
import scipy.stats as spstats
from statsmodels.stats.multitest import fdrcorrection

def multiple_wilcoxon_test(A_postim, A_prestim, alternative='greater', alpha=0.05):
        """
        Wilcoxon test hypothesis of no difference between prestimulus and postimulus amplitude
        Correct for multilple hypothesis test.
        ----------
        Parameters
        ----------
        A_postim: (...,times) array
                Postimulus amplitude
        A_prestim: (...,times) array
                    Presimulus amplitude
        alpha: float
            significance threshold to reject the null
        From scipy.stats.wilcoxon:
        alternative: {“two-sided”, “greater”, “less”}, optional
        zero_method: {“pratt”, “wilcox”, “zsplit”}, optional
        """
        # Average over time window 
        A_postim = np.mean(A_postim, axis=-1)
        A_prestim = np.mean(A_prestim, axis=-1)
        # Iniitialise inflated p values
        nchans = A_postim.shape[-1]
        w = [0]*nchans
        pval = [0]*nchans
        z = [0]*nchans
        # Compute test stats given non normal distribution
        for i in range(0,nchans):
            # Paired test
            n = A_postim.shape[0]
            mn = n * (n + 1)/4 # mean of null
            se = np.sqrt(n * (n + 1) * (2*n+1)/24) # se of null (no correction for tie)
            w[i], pval[i] = spstats.wilcoxon(A_postim[:,i], A_prestim[:,i], alternative='greater')
            z[i] = (w[i] - mn)/se
            z[i] = round(z[i],2)
            # Unpaired test
            #z[i], pval[i], z[i] = spstats.ranksums(A_postim[:,i], A_prestim[:,i], 
            #                                     alternative=self.alternative) 
        # Correct for multiple testing    
        reject, pval_correct = fdrcorrection(pval, alpha=alpha)
        return reject, pval_correct, z
    
