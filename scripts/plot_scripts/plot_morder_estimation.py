#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:31:56 2022

@author: guime
"""
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

from scipy.io import loadmat
from pathlib import Path

#%%
plt.style.use('ggplot')
fig_width = 16  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 10
params = {'backend': 'ps',
          'lines.linewidth': 1,
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
# Useful paths
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
result_path = cifar_path.joinpath('results')
signal = 'lfp'
infocrit = 'bic'
fname = signal + '_model_order_estimation.m'
path = result_path
fpath = path.joinpath(fname)
# Read dataset
dataset = loadmat(fpath)
model_order = dataset['ModelOrder']

#%%

def plot_var_model_order(model_order, cohort, infocrit='aic'):
    conditions = ['Rest', 'Face', 'Place']
    nsub = len(cohort)
    ncomp = len(conditions)
    fig, ax = plt.subplots(ncomp, nsub)
    # Loop over subject and comparison to plot varmodel order
    for s, subject in enumerate(cohort):
        for c, condition in enumerate(conditions):
            varmo = model_order[subject][0][0][condition][0][0]['varmo'][0][0]
            rho = model_order[subject][0][0][condition][0][0]['rho'][0][0][0][0]
            rho = round(rho,2)
            morder = varmo[infocrit][0][0]
            saic = varmo['saic'][0][0]
            sbic = varmo['sbic'][0][0]
            shqc = varmo['shqc'][0][0]
            lags = varmo['lags'][0][0]
            ax[c,s].plot(lags, saic, color = 'r', label = 'aic')
            ax[c,s].plot(lags, shqc, color = 'b', label = 'hqc')
            ax[c,s].plot(lags, sbic, color = 'g', label = 'bic')
            ax[c,s].axvline(x=morder, color='k')
            ax[c,s].annotate(f'rho={rho}', 
                       xy = (0.75, 0.75), xycoords='axes fraction', fontsize = 8)
            ax[c,0].set_ylabel(f'Morder {condition}')
            ax[0,s].set_title(f'Subject {s}')
            ax[-1,s].set_xlabel('Lags (obs)')
    ax[-1,-1].legend()
    
#%%

plot_var_model_order(model_order, cohort, infocrit=infocrit)
