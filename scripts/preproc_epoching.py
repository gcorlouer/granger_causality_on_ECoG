#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 19:16:11 2022
This script epoch hfb or ecog into condition specific multitrial time series
and save it as a derivatives
@author: guime
"""


from src.preprocessing_lib import EcogReader, Epocher
from src.input_config import args 

#%% Epoch baseline rescale hfb 

conditions = ['Rest', 'Stim','Face', 'Place']
for subject in args.cohort:
    reader = EcogReader(args.data_path, subject=subject, stage=args.stage, 
                   preprocessed_suffix=args.preprocessed_suffix)
    raw = reader.read_ecog()
    for condition in conditions:
        epocher = Epocher(condition=condition, t_prestim=args.t_prestim, 
                          t_postim = args.t_postim, baseline=args.baseline, 
                          preload=args.preload, tmin_baseline=args.tmin_baseline, 
                          tmax_baseline=args.tmax_baseline, mode=args.mode)
        hfb = reader.read_ecog()
        hfb = epocher.scale_epoch(hfb)
        subject_path = args.derivatives_path.joinpath(subject, 'ieeg')
        fname = subject + '_hfb_' + condition + '_scaled_raw.fif'
        fpath = subject_path.joinpath(fname)
        hfb.save(fpath, overwrite=True)

#%% Epoch ECoG

