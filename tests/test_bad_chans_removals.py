"""
This script contains testing functions for each steps of bad channels removal
"""
from libs.preprocessing_lib import Ecog, drop_bad_chans, pick_99_percentile_chans
from libs.preprocessing_lib import mark_high_std_outliers, mark_physio_chan
from libs.input_config import args

import numpy as np

#%% Parameters

path = args.path
stage = args.stage
cohort = args.cohort


#%% Test percentile removal

def test_percentile_removal():
    # Print number of removed channels for all subjects
    for subject in cohort:
        ecog = Ecog(path, stage = stage, subject=subject, preload=True, 
                         epoch=False)
        raw = ecog.concatenate_condition()
        outliers_chans = pick_99_percentile_chans(raw, q=99,voltage_threshold=500e-6)
        print(f"Subject {subject} has {len(outliers_chans)} above threshold")

test_percentile_removal()

#%% Test std removal

def test_std_removal():
     for subject in cohort:
        ecog = Ecog(path, stage = stage, subject=subject, preload=True, 
                         epoch=False)
        raw = ecog.concatenate_condition()
        raw = mark_high_std_outliers(raw, n_std=5)
        bads_std = raw.info['bads']
        print(f"Subject {subject} has chans {bads_std} as bads")
        
test_std_removal()

#%% Test marking physiological channels:

def test_mark_physio():
    for subject in cohort:
        ecog = Ecog(path, stage = stage, subject=subject, preload=True, 
                         epoch=False)
        raw = ecog.concatenate_condition()
        raw = mark_physio_chan(raw)
        physio = raw.info['bads']
        print(f"Subject {subject} has chans {physio} as physio")
        
test_mark_physio()

#%% Test drop all bad chans:

def test_drop_bad_chans():
    for subject in cohort:
        ecog = Ecog(path, stage = stage, subject=subject, preload=True, 
                         epoch=False)
        raw = ecog.concatenate_condition()
        raw, bads = drop_bad_chans(raw)
        print(f"All bad chans are {bads}")

test_drop_bad_chans()


#%% Plot psd to check no bad channels is left

for subject in cohort:
    ecog = Ecog(path, stage = stage, subject=subject, preload=True, 
                     epoch=False)
    raw = ecog.concatenate_condition()
    raw = drop_bad_chans(raw, q=99, voltage_threshold=500e-6, n_std=5)
    raw.plot_psd(xscale='log')
