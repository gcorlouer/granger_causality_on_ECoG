#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:18:05 2022

@author: guime
"""

from src.preprocessing_lib import EcogReader, Epocher
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import argparse

#%% Plot parameters

plt.style.use('ggplot')
fig_width = 16  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 10
params = {'backend': 'ps',
          'lines.linewidth': 1.5,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'font.size': label_size,
          'legend.fontsize': label_size,
          'xtick.labelsize': label_size,
          'ytick.labelsize': label_size,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

#%%
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

args = parser.parse_args()
#%% Make condition specific dictionary
#%matplotlib qt
# Read visual chans
reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
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
g.map_dataframe(sns.histplot, x='observation',hue='condition',stat='probability', palette=pal)
g.add_legend()
g.set_axis_labels("HFA value")
#g.fig.suptitle(f"{subject} histogram of log-baseline rescaled HFA")