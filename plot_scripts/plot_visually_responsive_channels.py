#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:18:05 2022

@author: guime
"""

from src.preprocessing_lib import EcogReader, Epocher
from src.input_config import args

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%% Make condition specific dictionary
#%matplotlib qt
cohort = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
subject =  'DiAs'
preprocessed_suffix = '_hfb_Stim_scaled-epo.fif'
# Read visual chans
reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
df_visual= reader.read_channels_info(fname='visual_channels.csv')
visual_chans = df_visual['chan_name'].to_list()
category = df_visual['group'].to_list()
location = df_visual['DK'].to_list()

# Read hfb

hfb = reader.read_ecog()
hfb = hfb.copy().pick_channels(visual_chans)

hfb_postim = hfb.copy().crop(tmin=0.05, tmax=0.5)
baseline = hfb.copy().crop(tmin=-0.5, tmax=0)

#%%  Make visual hfb dataframe

def build_hfb_dataframe(hfb, long_format=True, bad_condition = ['-1','-2']):
    # Remove irrelevant conditions
    df = hfb.to_data_frame(long_format=long_format)
    df['condition'] = df['condition'].astype(str)
    for i in bad_condition:
        df = df[df.condition != i]
        condition = df['condition'].to_list()
    for i in range(len(condition)):
        # P or F condition
        condition[i]=condition[i][0]
    df['condition'] = condition
    df = df.reset_index(drop=True)
    return df

#%% 
    
df_postim = build_hfb_dataframe(hfb_postim, long_format=True, bad_condition = ['-1','-2'])
df_baseline = build_hfb_dataframe(baseline, long_format=True, bad_condition = ['-1','-2'])

condition = df_baseline['condition'].to_list()
for i in range(len(condition)):
    # P or F condition
    condition[i]='B' 

df_baseline['condition'] = condition
df = df_postim.copy().append(df_baseline, ignore_index=True)


#%% Violin plot

#ax = sns.violinplot(x='channel', y ='value', hue='condition', data = df)
#ax.set_xticklabels(category)
#ax.axhline(y=0, color='k')

#%%

visual_chan_cat = [i + '-'+ j for i,j in zip(visual_chans,category)]
visual_dict = dict(zip(visual_chans, visual_chan_cat))
df = df.replace(to_replace=visual_dict)
#%% Histogram
sns.set(font_scale=1.2)

pal = dict(F='blue',P='orange', B='green')
g = sns.FacetGrid(df, col='channel', hue='condition',col_wrap=4, legend_out='True',
                  palette=pal)
g.map_dataframe(sns.histplot, x='value',hue='condition',stat='probability', palette=pal)
g.add_legend()
g.set_axis_labels("HFA value")
#g.fig.suptitle(f"{subject} histogram of log-baseline rescaled HFA")