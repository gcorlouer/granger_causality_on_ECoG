#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:50:45 2022
This script explore visual responsive hfb, compare scaled and unscaled
as well as log and non log hfb.

@author: guime
"""

from src.preprocessing_lib import EcogReader, Epocher
from src.input_config import args

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
# Read visual chans
subject = 'DiAs'
reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
df_visual= reader.read_channels_info(fname='visual_channels.csv')
visual_chans = df_visual['chan_name'].to_list()
category = df_visual['group'].to_list()
location = df_visual['DK'].to_list()

# Read hfb

hfb = reader.read_ecog()
hfb = hfb.copy().pick_channels(visual_chans)
epocher = Epocher(condition=args.condition)
epoch = epocher.epoch(hfb)
epoch_log= epocher.log_epoch(hfb)
time = epoch.times
#%% 

epoch.plot(n_epochs=5, n_channels=4)

#%%

epoch_log.plot(scalings=1e1, n_epochs=5, n_channels=4)

#%% 

evok = epoch.average()
evok.plot()

#%% 

evok_log = epoch_log.average()
evok_log.plot()

#%%

X = epoch.copy().get_data()
lX = np.log(X)

#%%

ichan = 4
itrial = 10
plt.subplot(1,2,1)
plt.plot(time, X[itrial, ichan,:])
plt.ylabel('Non log HFA')
plt.xlabel('Time (s)')
plt.subplot(1,2,2)
plt.plot(time, lX[itrial, ichan,:])
plt.xlabel('Time (s)')
plt.ylabel('Log HFA')

plt.suptitle('DiAs retinotopic channel Log vs Non log trial')

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
    
df = build_hfb_dataframe(epoch, long_format=True, bad_condition = ['-1','-2'])
df_log = build_hfb_dataframe(epoch_log, long_format=True, bad_condition = ['-1','-2'])


#%% Histogram
# sns.set(font_scale=1.2)

g = sns.FacetGrid(df, col='channel', col_wrap=4)
g.map_dataframe(sns.histplot, x='value', stat='probability')
g.set_axis_labels("HFA ")

#%% Histogram
# sns.set(font_scale=1.2)

g = sns.FacetGrid(df_log, col='channel', col_wrap=4)
g.map_dataframe(sns.histplot, x='value', stat='probability')
g.set_axis_labels("HFA ")

#%% Plot event related potential of visual channels

evok_visual = hfb_visual.average()

#%% Plot event related potential of non visual channels

evok_nv = hfb_nv.average()

time = evok_visual.times

#%% 

#%% 

X = evok_visual.get_data()
mX = np.mean(X,0)
Y = evok_nv.get_data()
mY = np.mean(Y,0)

plt.plot(time, mX, label='visual')
plt.plot(time, mY, label='non visual')
plt.axhline(y=0)
plt.axvline(x=0)
plt.xlabel('Time (s)')
plt.ylabel('HFA')
plt.legend()
plt.title(f'Visual vs Non visual baseline rescaled HFA {subject}')

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
