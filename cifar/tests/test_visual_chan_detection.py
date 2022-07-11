#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:20:23 2021

@author: guime
"""


import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from config import args

#%% Detect visual channels

subject = 'AnRa'
ecog = hf.Ecog(args.cohort_path, subject=subject, proc='preproc', 
                   stage = '_hfb_db_epo.fif', epoch=True)
hfb = ecog.read_dataset()
visual_detection = hf.VisualDetector(tmin_prestim=args.tmin_prestim, 
                                         tmax_prestim=args.tmax_prestim, 
                                         tmin_postim=args.tmin_postim,
                                         tmax_postim=args.tmax_postim, 
                                         alpha=args.alpha, 
                                         zero_method=args.zero_method, 
                                         alternative=args.alternative)


#%%

visual_chan, effect_size = visual_detection.detect(hfb)
#%% Classify visual channels

dfelec = ecog.read_channels_info()
visual_classifier = hf.VisualClassifier(tmin_prestim=args.tmin_prestim, 
                                         tmax_prestim=args.tmax_prestim, 
                                         tmin_postim=args.tmin_postim,
                                         tmax_postim=args.tmax_postim, 
                                         alpha=args.alpha, 
                                         zero_method=args.zero_method, 
                                         alternative=args.alternative)


# Extract events
event_id = hfb.event_id
face_id = visual_classifier.extract_stim_id(event_id, cat = 'Face')
place_id = visual_classifier.extract_stim_id(event_id, cat='Place')
image_id = face_id+place_id

# Detect visual channels
visual_chan, visual_responsivity = visual_classifier.detect(hfb)
visual_hfb = hfb.copy().pick_channels(visual_chan)

#%% Compute latency response
latency_response = visual_classifier.compute_latency(visual_hfb, image_id, visual_chan)

#%% Test latency response function

import scipy.stats as spstats
from statsmodels.stats.multitest import fdrcorrection, multipletests


sfreq = visual_hfb.info['sfreq']
A_postim = visual_classifier.crop_stim_hfb(visual_hfb, image_id, tmin=0, tmax=1.5)
A_prestim = visual_classifier.crop_stim_hfb(visual_hfb, image_id, tmin=-0.4, tmax=0)
A_baseline = np.mean(A_prestim, axis=-1) #No

pval = [0]*A_postim.shape[2]
tstat = [0]*A_postim.shape[2]
latency_response = [None]*len(visual_chan)

for i in range(len(visual_chan)):
    for t in range(np.size(A_postim,2)):
        tstat[t] = spstats.wilcoxon(A_postim[:,i,t], A_baseline[:,i],
                                    zero_method=visual_classifier.zero_method, 
                                    alternative='greater')
        pval[t] = tstat[t][1]
        
    reject, pval_correct = fdrcorrection(pval, alpha=visual_classifier.alpha) # correct for multiple hypotheses
    
    for t in range(0,np.size(A_postim,2)):
        if np.all(reject[t:t+50])==True :
            latency_response[i]=t/sfreq*1e3
            print("Found latency")
            break 
        else:
            continue
    


#%% Classify Face and Place populations
group, category_selectivity = visual_classifier.face_place(visual_hfb, face_id, place_id, visual_chan)

# Classify retionotopic

group = visual_classifier.retinotopic(visual_chan, group, dfelec)

#%% 

def retinotopic(visual_chan, group, dfelec):
        """
        Return retinotopic site from V1 and V2 given Brodman atlas.
        """
        nchan = len(group)
        bipolar_visual = [visual_chan[i].split('-') for i in range(nchan)]
        for i in range(nchan):
            brodman = (dfelec['Brodman'].loc[dfelec['electrode_name']==bipolar_visual[i][0]].to_string(index=False), 
                       dfelec['Brodman'].loc[dfelec['electrode_name']==bipolar_visual[i][1]].to_string(index=False))
            print(brodman)
            if 'V1' in brodman[0] or 'V2' in brodman[0] and ('V1' in brodman[1] or 'V2' in brodman[1]):
                group[i]='R'
                print(group[i])
        return group

#%% Drop visual channel with 0 latency:
        
nchan = len(visual_chan)
for i in range(nchan):
    if latency_response[i]==None :
        del visual_chan[i]
    else:
        continue
#%%
visual_populations = visual_classifier.hfb_to_visual_populations(hfb, dfelec)


df_visual = pd.DataFrame.from_dict(visual_populations)
df_visual = df_visual.sort_values(by='Y', ignore_index=True)
df_visual = df_visual[df_visual.latency != 0]

#%% Save into csv file
fname = 'visual_channels.csv'
subject_path = args.cohort_path.joinpath(subject)
brain_path = subject_path.joinpath('brain')
fpath = brain_path.joinpath(fname)
df_visual.to_csv(fpath, index=False)
