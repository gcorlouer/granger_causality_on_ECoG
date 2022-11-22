#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:32:18 2022

@author: guime
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:59:39 2021
This script plot var estimation for all subjects. 
@author: guime
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from pathlib import Path

#%% Style parameters

plt.style.use('ggplot')
fig_width = 16  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 10
tick_size = 8
params = {'backend': 'ps',
          'lines.linewidth': 1.2,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'font.size': label_size,
          'legend.fontsize': tick_size,
          'xtick.labelsize': tick_size,
          'ytick.labelsize': tick_size,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

#%%

cohort = ['AnRa',  'ArLa', 'DiAs']
# Useful paths
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')
signal = 'lfp'
fname = signal + '_rolling_model_order_estimation.m'
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
model_order = dataset['ModelOrder']
#min-max
varmo_min = 1
varmo_max = 10
ssmo_min = 5
ssmo_max = 30
rho_min = 0.93 #0.5 for lfp
rho_max = 1

#%% Function model order

def plot_rolling_var(model_order, cohort, varmo_min = 3, varmo_max=10):
    conditions = ['Rest', 'Face', 'Place']
    nsub = len(cohort)
    ncdt = len(conditions)
    fig, ax = plt.subplots(ncdt, nsub)
    # Loop over subject and comparison to plot varmodel order
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            aic = model_order[subject][0][0][condition][0][0]['aic'][0][0]
            hqc = model_order[subject][0][0][condition][0][0]['hqc'][0][0]
            bic = model_order[subject][0][0][condition][0][0]['bic'][0][0]
            rho = model_order[subject][0][0][condition][0][0]['rho'][0][0][0][0]
            win_time = model_order[subject][0][0][condition][0][0]['time'][0][0]
            time = win_time[:,-1]
            rho = round(rho,2)
            ax[c,s].plot(time, aic, color = 'r', label = 'aic')
            ax[c,s].plot(time, hqc, color = 'b', label = 'hqc')
            ax[c,s].plot(time, bic, color = 'g', label = 'bic')
            ax[c,s].set_ylim(varmo_min,varmo_max)
            ax[c,s].axvline(x=0, color='k')
            ax[0,s].set_title(f'Subject {s}')
            ax[-1,s].set_xlabel('Time (s)')
            if c<=1:
                        ax[c,s].set_xticks([]) # (turn off xticks)
            if s>=1:
                        ax[c,s].set_yticks([]) # (turn off xticks)
            ticks_labels = [-0.5,0,0.5,1,1.5]
            ax[-1,s].set_xticks(ticks_labels)
            ax[-1,s].set_xticklabels(ticks_labels)
            ax[c,0].set_ylabel(f'varmo {condition}')
        #ax[-1,-1].legend()
        handles, labels = ax[c,s].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        
def plot_rolling_svc(model_order, cohort, ssmo_min=10, ssmo_max=30):
    conditions = ['Rest', 'Face', 'Place']
    nsub = len(cohort)
    ncdt = len(conditions)
    fig, ax = plt.subplots(ncdt, nsub)
    # Loop over subject and comparison to plot varmodel order
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            svc = model_order[subject][0][0][condition][0][0]['ssmo'][0][0]
            win_time = model_order[subject][0][0][condition][0][0]['time'][0][0]
            time = win_time[:,-1]
            ax[c,s].plot(time, svc, color = 'b')
            ax[c,s].axvline(x=0, color='k')
            ax[c,s].set_ylim(ssmo_min,ssmo_max)
            ax[0,s].set_title(f'Subject {s}')
            ax[-1,s].set_xlabel('Time (s)')
            if c<=1:
                        ax[c,s].set_xticks([]) # (turn off xticks)
            if s>=1:
                        ax[c,s].set_yticks([]) # (turn off xticks)
            ticks_labels = [-0.5,0,0.5,1,1.5]
            ax[-1,s].set_xticks(ticks_labels)
            ax[-1,s].set_xticklabels(ticks_labels)
            ax[c,0].set_ylabel(f'ssmo {condition}')
        
def plot_rolling_specrad(model_order, cohort, rho_min=0.9, rho_max=1):
    conditions = ['Rest', 'Face', 'Place']
    nsub = len(cohort)
    ncdt = len(conditions)
    fig, ax = plt.subplots(ncdt, nsub)
    # Loop over subject and comparison to plot varmodel order
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            rho = model_order[subject][0][0][condition][0][0]['rho'][0][0]
            win_time = model_order[subject][0][0][condition][0][0]['time'][0][0]
            time = win_time[:,-1]
            ax[c,s].plot(time, rho, color = 'b')
            ax[c,s].axvline(x=0, color='k')
            ax[c,s].set_ylim(rho_min,rho_max)
            ax[0,s].set_title(f'Subject {s}')
            ax[-1,s].set_xlabel('Time (s)')
            if c<=1:
                        ax[c,s].set_xticks([]) # (turn off xticks)
            if s>=1:
                        ax[c,s].set_yticks([]) # (turn off xticks)
            ticks_labels = [-0.5,0,0.5,1,1.5]
            ax[-1,s].set_xticks(ticks_labels)
            ax[-1,s].set_xticklabels(ticks_labels)
            ax[c,0].set_ylabel(r'$\rho$' + f' {condition}')

#%% Plot rolling var

fpath = Path('~','thesis','overleaf_project', 'figures','method_figure').expanduser()
fname = signal + '_rolling_var.png'
figpath = fpath.joinpath(fname)

plot_rolling_var(model_order, cohort, varmo_min=varmo_min, varmo_max=varmo_max)
plt.savefig(figpath)

#%% Plot rolling var

fpath = Path('~','thesis','overleaf_project', 'figures','method_figure').expanduser()
fname = signal + '_rolling_ss.png'
figpath = fpath.joinpath(fname)

plot_rolling_svc(model_order, cohort, ssmo_min=ssmo_min, ssmo_max=ssmo_max)
plt.savefig(figpath)

#%% Plot rolling rho

fpath = Path('~','thesis','overleaf_project', 'figures','method_figure').expanduser()
fname = signal + '_rolling_specrad.png'
figpath = fpath.joinpath(fname)

plot_rolling_specrad(model_order, cohort, rho_min=rho_min, rho_max=rho_max)
plt.savefig(figpath)


