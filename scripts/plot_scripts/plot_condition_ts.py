#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:08:32 2022

@author: guime
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

from src.preprocessing_lib import prepare_condition_scaled_ts
from pathlib import Path
from scipy.stats import sem

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

cohort = ['AnRa',  'ArLa', 'DiAs']
cohort_dic = {'AnRa': 'S0', 'ArLa': 'S1', 'DiAs': 'S2'}

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
parser.add_argument("--preprocessed_suffix", type=str, default= '_hfb_continuous_raw.fif')
parser.add_argument("--epoch", type=bool, default=False)
parser.add_argument("--channels", type=str, default='visual_channels.csv')
parser.add_argument("--matlab", type=bool, default=False)

#Filtering parameters

parser.add_argument("--l_freq", type=float, default=70.0)
parser.add_argument("--band_size", type=float, default=20.0)
parser.add_argument("--nband", type=float, default=5)
parser.add_argument("--l_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--h_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--filter_length", type=str, default='auto')
parser.add_argument("--phase", type=str, default='minimum')
parser.add_argument("--fir_window", type=str, default='blackman')

#% Epoching parameters
parser.add_argument("--condition", type=str, default='Stim') 
parser.add_argument("--t_prestim", type=float, default=-0.5)
parser.add_argument("--t_postim", type=float, default=1.75)
parser.add_argument("--baseline", default=None) # No baseline from MNE
parser.add_argument("--preload", default=True)
parser.add_argument("--tmin_baseline", type=float, default=-0.4)
parser.add_argument("--tmax_baseline", type=float, default=0)
# Wether to log transform the data
parser.add_argument("--log_transf", type=bool, default=False)
# Mode to rescale data (mean, logratio, zratio)
parser.add_argument("--mode", type=str, default='logratio')

#% Visually responsive channels classification parmeters

parser.add_argument("--tmin_prestim", type=float, default=-0.4)
parser.add_argument("--tmax_prestim", type=float, default=-0.05)
parser.add_argument("--tmin_postim", type=float, default=0.05)
parser.add_argument("--tmax_postim", type=float, default=0.4)
parser.add_argument("--alpha", type=float, default=0.05)
#parser.add_argument("--zero_method", type=str, default='pratt')
parser.add_argument("--alternative", type=str, default='greater')

#% Create category specific time series

parser.add_argument("--decim", type=float, default=2)
parser.add_argument("--tmin_crop", type=float, default=-0.5)
parser.add_argument("--tmax_crop", type=float, default=1.5)

#% Functional connectivity parameters

parser.add_argument("--nfreq", type=float, default=1024)
parser.add_argument("--roi", type=str, default="functional")

args = parser.parse_args()

#%%

def plot_condition_ts(args, cohort):
    # Prepare inputs for plotting
    conditions = ['Rest','Face','Place']
    ncdt = len(conditions)
    nsub = len(cohort)
    fig, ax = plt.subplots(ncdt,nsub, sharex=False, sharey=False)
    # Prepare condition ts
    for s, subject in enumerate(cohort):
        ts = prepare_condition_scaled_ts(args.data_path, subject=subject, 
                                         stage=args.stage,
                            preprocessed_suffix=args.preprocessed_suffix, decim=args.decim,
                            epoch=args.epoch, t_prestim=args.t_prestim, t_postim=args.t_postim, 
                            tmin_baseline = args.tmin_baseline, tmax_baseline = args.tmax_baseline,
                            tmin_crop=args.tmin_crop, tmax_crop=args.tmax_crop, 
                            mode = args.mode, channels = args.channels)
        populations = ts['indices'].keys()
        time = ts['time']
        baseline = ts['baseline']
        baseline = np.average(baseline)
        # Plot condition ts
        for c, cdt in enumerate(conditions):
            for pop in populations:
                # Condition specific neural population
                X = ts[cdt]
                pop_idx = ts['indices'][pop]
                X = X[pop_idx,:,:]
                X = np.average(X, axis = 0)
                # Compute evoked response
                evok = np.average(X, axis=1)
                # Compute confidence interval
                smX = sem(X,axis=1)
                up_ci = evok + 1.96*smX
                down_ci = evok - 1.96*smX
                # Plot condition-specific evoked HFA
                ax[c,s].plot(time, evok, label=pop)
                ax[c,s].fill_between(time, down_ci, up_ci, alpha=0.4)
                ax[c,s].set_ylim([-2.5, 6])
                ax[c,s].axvline(x=0, color ='k')
                ax[c,s].axhline(y=baseline, color='k')
                ax[c,0].set_ylabel(f'{cdt} (dB)')
                ax[0,s].set_title(f'subject S{s}')
                if c<=2:
                        ax[c,s].set_xticks([]) # (turn off xticks)
                if s>=1:
                        ax[c,s].set_yticks([]) # (turn off xticks)
                handles, labels = ax[c,s].get_legend_handles_labels()
                #ax[2,s].set_xticks([-400, 0, 1000])
                #ax[2,s].set_xticklabels([-0.4, 0, 1]) 
                #ax[2,s].set_xlabel("Time (s)")
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle('Average HFA', )
    #fig.supxlabel("Time (s)")
    plt.show()
    #fname = subject + figname
    #fpath = fpath.joinpath(fname)
    #plt.savefig(fpath)
    
#%%

plot_condition_ts(args, cohort)