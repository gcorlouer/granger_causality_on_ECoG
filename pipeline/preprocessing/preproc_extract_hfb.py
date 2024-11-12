#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:36:49 2021
This script extract HFB, epoch it, and rescale it into db for all subjects
@author: guime
"""

from src.preprocessing_lib import EcogReader, HfbExtractor
from src.input_config import args

# %% Extract hfb

for subject in args.cohort:
    reader = EcogReader(
        args.data_path,
        subject=subject,
        stage=args.stage,
        preprocessed_suffix=args.preprocessed_suffix,
    )
    raw = reader.read_ecog()
    extractor = HfbExtractor(
        l_freq=args.l_freq,
        nband=args.nband,
        band_size=args.band_size,
        l_trans_bandwidth=args.l_trans_bandwidth,
        h_trans_bandwidth=args.h_trans_bandwidth,
        filter_length=args.filter_length,
        phase=args.phase,
        fir_window=args.fir_window,
    )
    # Extract hfb
    hfb = extractor.extract_hfb(raw)
    subject_path = args.derivatives_path.joinpath(subject, "ieeg")
    fname = subject + "_hfb_continuous_raw.fif"
    fpath = subject_path.joinpath(fname)
    hfb.save(fpath, overwrite=True)
