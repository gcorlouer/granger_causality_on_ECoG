#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 22:46:54 2022

@author: guime
"""


from pathlib import Path

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#%%

result_path = Path('../results/')
fname='all_visual_channels.csv'
fpath = result_path.joinpath(fname)
df= pd.read_csv(fpath)

#%%

columns = list(df['group'].unique())
index = list(df['subject_id'].unique())

df_dem = pd.DataFrame(columns=columns, index=index)

for subject in index:
    for group in columns:
        dgroup = df['group'].loc[ (df['group']== group) & (df['subject_id']==subject)]
        df_dem.at[subject, group] = len(dgroup)

#%% 
        
df_dem['total'] = df_dem.sum(axis=1)

#%%
tot = df_dem.sum(axis=0)
tot.name = 'total'
df_dem = df_dem.append(tot)

#%% Save dataframe

fname = 'visual_channels_per_subjects.csv'
fpath = result_path.joinpath(fname)
df_dem.to_csv(fpath)