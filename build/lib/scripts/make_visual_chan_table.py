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

#%% Read visual chan table across all subjects

home = Path.home()
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
derivatives_path = data_path.joinpath('derivatives')
result_path = cifar_path.joinpath('results')
fname='all_visual_channels.csv'
fpath = result_path.joinpath(fname)
df= pd.read_csv(fpath)
# Since we do not study place responsive replace by others
df = df.replace('P','O',regex=True)
col_names = {'latency':'latency (ms)', 'DK': 'ROI (DK)', 'peak_time':'peak (ms)',
             'visual_responsivity':'response (z)','category_selectivity': 'selectivity (z)',
             'brodman': 'ROI (BM)'}
df = df.rename(columns=col_names)
#%% Make dataframe of visual channels distribution

columns = list(df['group'].unique())
index = list(df['subject_id'].unique())

df_visual_disritb = pd.DataFrame(columns=columns, index=index)

for subject in index:
    for group in columns:
        dgroup = df['group'].loc[ (df['group']== group) & (df['subject_id']==subject)]
        df_visual_disritb.at[subject, group] = len(dgroup)

#%% 
        
df_visual_disritb['total'] = df_visual_disritb.sum(axis=1)

#%%
tot = df_visual_disritb.sum(axis=0)
tot.name = 'total'
df_visual_disritb = df_visual_disritb.append(tot)

#%% Save dataframe

fname = 'visual_channels_distribution.csv'
fpath = result_path.joinpath(fname)
df_visual_disritb.to_csv(fpath)

#%% Visual channels info

subjects_id = {'AnRa':0, 'ArLa':1, 'DiAs':2}
# Replace subjects with subject id
df_info = df.replace(subjects_id)
# Keep subjects of interest
df_info = df_info.loc[df_info['subject_id'].isin([0,1,2])]
# Remove X and Z MNI coordinates
df_info = df_info.drop(labels=['X','Z'], axis=1)
# Since we do not study place responsie replace by others
df_info = df_info.replace('P','O',regex=True)
# Replace DK ROI labels
df_info = df_info.replace('ctx-rh-','',regex=True)
df_info = df_info.replace('ctx-lh-','',regex=True)
# Drop chan names
df_info = df_info.drop(labels='chan_name', axis=1)
# Since we do not study place responsie replace by others
df_info = df_info.replace('P','O',regex=True)
df_info = df_info.sort_values(by='Y')
df_info = df_info.round(2)
# Re-index
df_info = df_info.reset_index(drop=True)
df_info.reset_index(drop=True)
#%% Save dataframe

fname = 'visual_channels_location_and_response.csv'
fpath = result_path.joinpath(fname)
df_info.to_csv(fpath, index='False')


# #%% Save data info per subjects

# cohort = ['AnRa', 'ArLa','DiAs']
# for subject in cohort:
#     df_info = df.loc[df['subject_id']==subject]
#     df_info = df_info.drop(labels='subject_id', axis=1)
#     df_info = df_info.drop(labels=['X','Z'], axis=1)
#     # Replace DK ROI labels
#     df_info = df_info.replace('ctx-rh-','',regex=True)
#     df_info = df_info.replace('ctx-lh-','',regex=True)
#     # Drop Brodman
#     #df_info = df_info.drop(labels='brodman', axis=1)
#     # Drop chan names
#     df_info = df_info.drop(labels='chan_name', axis=1)
#     # Since we do not study place responsie replace by others
#     df_info = df_info.replace('P','O',regex=True)
#     df_info = df_info.sort_values(by='Y')
#     df_info = df_info.round(2)
#     # Re-index
#     df_info = df_info.reset_index(drop=True)
#     fname = subject + '_visual_channels_location_and_response.csv'
#     fpath = result_path.joinpath(fname)
#     df_info.to_csv(fpath, index='False')