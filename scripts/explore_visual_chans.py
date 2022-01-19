#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:23:24 2022

@author: guime
"""


from src.preprocessing_lib import EcogReader, Epocher
from src.input_config import args
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

#%% Build dataframe with number of visually responsive per DK regions

category = list(df['group'].unique())
rois = list(df['DK'].unique())
visual_distrib ={'ROI':[], 'O':[], 'R':[], 'F':[], 'P':[]}

for roi in rois:
    visual_distrib['ROI'].append(roi)
    for cat in category:
        if roi in list(df['DK'].loc[df['group']==cat]):
            ncat = df['DK'].loc[df['group']==cat].value_counts()[roi]
        else:
            ncat = 0
        visual_distrib[cat].append(ncat)

df_dist = pd.DataFrame.from_dict(visual_distrib)


#%%

distrib_dict = {'O':[], 'R':[],'F':[], 'P':[]}
for cat in category:
    df_cat = pd.DataFrame(columns=['ROI', cat])
    df_cat['ROI'] = df_dist['ROI'].loc[df_dist[cat] > 0].to_list()
    df_cat[cat] = df_dist[cat].loc[df_dist[cat] > 0].to_list()
    distrib_dict[cat] = df_cat
    
#%% Plot bar plot of visually reponsive channels per regions
#%matplotlib qt
fig, axes = plt.subplots(1,4, sharey= True)
for i, cat in enumerate(category):
    sns.barplot(x='ROI', y=cat, data=distrib_dict[cat], ax=axes[i], color='k')
    labels = axes[i].get_xticklabels()
    axes[i].set_xticklabels(labels, rotation=90)
    
#%% Add total row to count toal of face, ret etc
    
