#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:23:24 2022

@author: guime
"""

from pathlib import Path

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
result_path = Path('../results/')
fname='all_visual_channels.csv'
fpath = result_path.joinpath(fname)
df= pd.read_csv(fpath)

#%% Build dataframe with number of visually responsive per DK regions (face and
# retinotopic channels only)

category = ['R', 'F']
rois = list(df['DK'].unique())
visual_distrib ={'ROI':[], 'R':[], 'F':[]}

for roi in rois:
    visual_distrib['ROI'].append(roi)
    for cat in category:
        if roi in list(df['DK'].loc[df['group']==cat]):
            ncat = df['DK'].loc[df['group']==cat].value_counts()[roi]
        else:
            ncat = 0
        visual_distrib[cat].append(ncat)

df_dist = pd.DataFrame.from_dict(visual_distrib)

#%% Pick anatomical regions


R_ROI = df_dist['ROI'].loc[(df_dist['R']==1)].to_list()
F_ROI = df_dist['ROI'].loc[(df_dist['F']==1)].to_list()
RF_ROI = R_ROI + F_ROI

#%%

distrib_dict = {'R':[],'F':[]}
for cat in category:
    df_cat = pd.DataFrame(columns=['ROI', cat])
    df_cat['ROI'] = df_dist['ROI'].loc[df_dist[cat] > 0].to_list()
    df_cat[cat] = df_dist[cat].loc[df_dist[cat] > 0].to_list()
    distrib_dict[cat] = df_cat

nR = df_dist['R'].sum()
nF = df_dist['F'].sum()


#%% Plot bar plot of visually reponsive channels per regions
#%matplotlib qt
fig, axes = plt.subplots(1,2, sharey= True)
for i, cat in enumerate(category):
    sns.barplot(x='ROI', y=cat, data=distrib_dict[cat], ax=axes[i], color='k')
    labels = axes[i].get_xticklabels()
    axes[i].set_xticklabels(labels, rotation=90)
    
#%% Make visual channel distribution table

    
visual_channels = {'subject':[], 'nR':[], 'nF':[], 'nO':[], 'nP':[]}
cohort = df['subject_id'].unique().tolist()
nsub = len(cohort)
nR = [0]*nsub
nF = [0]*nsub
nO = [0]*nsub
nP = [0]*nsub

for i, subject in enumerate(cohort):
    n = len(df.loc[(df['group']=='R') & (df['subject_id']==subject)])
    nR[i] = n
    n = len(df.loc[(df['group']=='F') & (df['subject_id']==subject)])
    nF[i] = n
    n = len(df.loc[(df['group']=='O') & (df['subject_id']==subject)])
    nO[i] = n
    n = len(df.loc[(df['group']=='P') & (df['subject_id']==subject)])
    nP[i] = n

visual_channels['subject'] = [i for i in range(len(cohort))]
visual_channels['nR'] = nR
visual_channels['nF'] = nF
visual_channels['nO'] = nO
visual_channels['nP'] = nP

df_table = pd.DataFrame.from_dict(visual_channels)
df_table = df_table.append(df_table.sum(numeric_only=True), ignore_index=True)

    
