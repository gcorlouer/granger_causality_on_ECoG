#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 14:41:19 2022

@author: guime
"""

from src.preprocessing_lib import prepare_condition_scaled_ts
from src.input_config import args

subject = 'DiAs'
matlab = False

ts = prepare_condition_scaled_ts(args.data_path, subject=subject, stage=args.stage, matlab = matlab,
                     preprocessed_suffix=args.preprocessed_suffix, decim=args.decim,
                     epoch=args.epoch, t_prestim=args.t_prestim, t_postim=args.t_postim, 
                     tmin_baseline = args.tmin_baseline, tmax_baseline = args.tmax_baseline,
                     tmin_crop=args.tmin_crop, tmax_crop=args.tmax_crop, mode=args.mode)