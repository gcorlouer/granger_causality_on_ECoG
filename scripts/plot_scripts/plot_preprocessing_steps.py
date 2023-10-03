#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:06:00 2022
In this script we test function from plotting library
@author: guime
"""

from src.preprocessing_lib import EcogReader, parcellation_to_indices, Epocher
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import sem
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import argparse

#%% Plotting parameters

plt.style.use('ggplot')
fig_width = 20  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
label_size = 12
tick_label_size = 8
params = {'backend': 'ps',
          'lines.linewidth': 1.2,
          'axes.labelsize': label_size,
          'axes.titlesize': label_size,
          'font.size': label_size,
          'legend.fontsize': tick_label_size,
          'xtick.labelsize': tick_label_size,
          'ytick.labelsize': tick_label_size,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

#%% Paths
home = Path.home()
cifar_path = Path('~','projects','cifar').expanduser()
data_path = cifar_path.joinpath('data')
derivatives_path = data_path.joinpath('derivatives')
result_path = cifar_path.joinpath('results')

#%% Input parameters
cohort = ['AnRa',  'ArLa', 'DiAs']
# Path to source data, derivatives and results. Enter your own path in local machine
parser = argparse.ArgumentParser()
# Dataset parameters 
parser.add_argument("--cohort", type=list, default=cohort)
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--sfeq", type=float, default=500.0)

parser.add_argument("--stage", type=str, default='preprocessed')
parser.add_argument("--preprocessed_suffix", type=str, default= '_hfb_Stim_scaled-epo.fif')
parser.add_argument("--epoch", type=bool, default=True)
parser.add_argument("--channels", type=str, default='visual_channels.csv')

# Input parameters:
# Stages
# bipolar_montage
# preprocessed

# Preprocessing suffixes:
# Bad chans removed, concatenated raw ECoG:
# _bad_chans_removed_raw.fif: 
# Continuous data:
# '_hfb_continuous_raw.fif' 
# Epoched data
# _hfb_Stim_scaled-epo.fif
# _hfb_Face_scaled-epo.fif
# _hfb_Condition_scaled-epo.fif

# Channels info:
# 'BP_channels.csv'
# 'all_bad_channels.csv'
# 'visual_channels.csv'
# 'all_visual_channels.csv'
# 'electrodes_info.csv'

#% Filtering parameters

parser.add_argument("--l_freq", type=float, default=70.0)
parser.add_argument("--band_size", type=float, default=20.0)
parser.add_argument("--nband", type=float, default=5)
parser.add_argument("--l_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--h_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--filter_length", type=str, default='auto')
parser.add_argument("--phase", type=str, default='minimum')
parser.add_argument("--fir_window", type=str, default='blackman')

#%  Epoching parameters
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
parser.add_argument("--mode", type=str, default='mean')

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
parser.add_argument("--tmax_crop", type=float, default=1.75)

#% Functional connectivity parameters

parser.add_argument("--nfreq", type=float, default=1024)
parser.add_argument("--roi", type=str, default="functional")

# Types of roi:
# functional
# anatomical

# 
args = parser.parse_args()
#%% Plot narrow band

def plot_narrow_broadband(args, data_path, 
                          chan = ['LTo1-LTo2'], tmin=500, tmax=506):
    """
    Plot the narrowband envelope and broadband hfa between (tmin, tmax) for 
    one subject and one reperesentative channel
    """
    # Read EcoG data
    reader = EcogReader(data_path, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                 epoch=False)
    ecog = reader.read_ecog()

    #Extract narrowband envelope
    raw_band = ecog.copy().filter(l_freq=args.l_freq, h_freq=args.l_freq+args.band_size,
                                         phase=args.phase, filter_length=args.filter_length,
                                         l_trans_bandwidth= args.l_trans_bandwidth, 
                                         h_trans_bandwidth= args.h_trans_bandwidth,
                                             fir_window=args.fir_window) 
    envelope = raw_band.copy().apply_hilbert(envelope=True)
    # Filtered signal
    X_filt = raw_band.copy().pick_channels(chan).crop(tmin=tmin, tmax=tmax).get_data()
    X_filt = X_filt[0,:]
    # muV
    X_filt =X_filt*1e6
    print(f"Filtered signal shape is {X_filt.shape}\n")
    # Signal narrow band envelope
    narrow_envelope = envelope.copy().pick_channels(chan).crop(tmin=tmin, tmax=tmax).get_data()
    narrow_envelope = narrow_envelope[0,:]
    narrow_envelope = narrow_envelope*1e6
    print(f"narrow band envelope shape is {narrow_envelope.shape}\n")
    # Time axis
    time = raw_band.copy().pick_channels(chan).crop(tmin=tmin, tmax=tmax).times
    
    #Rread hfb
    reader = EcogReader(data_path, stage=args.stage,
                     preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                     epoch=False)
    
    ecog = reader.read_ecog()
    # Extract single channel broadband envelope
    hfb = reader.read_ecog()
    hfb = hfb.copy().pick(chan)
    hfb = hfb.copy().crop(tmin=tmin, tmax=tmax)
    broadband_envelope = hfb.copy().get_data()
    broadband_envelope = broadband_envelope[0,:]
    broadband_envelope = broadband_envelope*1e6
    print(f"broadband_envelope shape is {narrow_envelope.shape}\n")
    # Plot narrowband and broadband envelope
    f, ax = plt.subplots(2,1,sharex=True)
    # Plot narrow band
    ax[0].plot(time, X_filt, label='ECoG')
    ax[0].plot(time, narrow_envelope, label='Narrow band envelope')
    ax[0].set_ylabel('Signal (V)')
    ax[0].legend(bbox_to_anchor=(0.3,1.02), loc='lower left')
    ax[0].set_title('a)', loc='left')
    # Plot broadband
    ax[1].plot(time, broadband_envelope)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Amplitude (muV)')
    ax[1].set_title('b)', loc='left')
    
fname = 'DiAs_narrow_broadband_stim.pdf'
home = Path.home()
fpath = home.joinpath('thesis','overleaf_project','figures','method_figure')
plot_narrow_broadband(args, data_path, chan = ['LTo1-LTo2'], tmin=500, tmax=506)

# Save figure
fpath = fpath.joinpath(fname)
plt.savefig(fpath)

#%% Plot log trial
fname = 'DiAs_log_trial.pdf'
home = Path.home()
fpath = home.joinpath('thesis','overleaf_project','figures','method_figure')

def plot_log_trial(args, data_path,
                          chan = ['LTo1-LTo2'], itrial=2, nbins=50):
    """
    This function plot log trial vs non log trial and their respective 
    distribution
    """
    reader = EcogReader(data_path, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                 epoch=False)
    # Read visually responsive channels
    hfa = reader.read_ecog()
    epocher = Epocher(condition='Face', t_prestim=args.t_prestim, t_postim = args.t_postim, 
                                 baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                                 tmax_baseline=args.tmax_baseline, mode=args.mode)
    # Epoch HFA
    epoch = epocher.epoch(hfa)
    # Downsample by factor of 2 and check decimation
    epoch = epoch.copy().decimate(args.decim)
    time = epoch.times
    # Get epoch data
    epoch = epoch.copy().pick(chan)
    # Get baseline
    prestim = epoch.copy().crop(tmin=args.tmin_prestim, tmax=args.tmax_prestim)
    prestim = prestim.get_data()
    prestim = np.ndarray.flatten(prestim)
    # Results in muV
    baseline = np.mean(prestim)
    # Get epoch data
    X = epoch.copy().get_data()
    # Results in muV
    X = X*1e6
    # Trial
    trial = X[itrial,0, :]
    # Log epoch HFA
    l_epoch = epocher.log_epoch(hfa)
    l_epoch = l_epoch.copy().decimate(args.decim)
    l_epoch = l_epoch.copy().pick(chan)
    l_baseline = np.log(baseline)
    l_X = l_epoch.copy().get_data()
    # Log trial
    l_trial = l_X[itrial,0, :]
    # Get array for histogram
    x =  np.ndarray.flatten(X)
    l_x = np.ndarray.flatten(l_X)
    amax = np.amax(x)
    # Plot log vs non log trial and histogram
    f, ax = plt.subplots(2,2)
    # Plot trial
    ax[0,0].plot(time, trial)
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel('HFA (muV)')
    ax[0,0].axvline(x=0, color='k')
    ax[0,0].axhline(y=baseline*1e6, color='k')
    ax[0,0].set_title('a)',loc='left')
    # Plot log trial
    ax[0,1].plot(time, l_trial)
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel('Log HFA ')
    ax[0,1].axvline(x=0, color='k')
    ax[0,1].set_title('b)',loc='left')
    ax[0,1].axhline(y=l_baseline, color='k')
    # Plot hfa histogram
    sns.histplot(x, stat = 'probability', bins=nbins, kde=True, ax=ax[1,0])
    ax[1,0].set_xlabel('HFA (muV)')
    ax[1,0].set_ylabel('Probability')
    ax[1,0].set_xlim(left=0, right=amax)
    ax[1,0].set_title('c)',loc='left')

    # Plot log hfa histogram
    sns.histplot(l_x, stat = 'probability', bins=nbins, kde=True, ax=ax[1,1])
    ax[1,1].set_xlabel('Log HFA')
    ax[1,1].set_ylabel('Probability')
    ax[1,1].set_title('d)',loc='left')
    #ax[1,1].set_xlim(left=0, right=amax)
    plt.tight_layout()

plot_log_trial(args, data_path, chan = ['LTo1-LTo2'], itrial=2, nbins=50)

# Save figure
fpath = fpath.joinpath(fname)
plt.savefig(fpath)
#%% Plot visual trial

def plot_visual_trial(args, data_path,
                          chan = ['LTo1-LTo2'], itrial=2, nbins=50):
    """
    Plot visually responsive trial and pre/postim distribution
    """
    reader = EcogReader(data_path, stage=args.stage,
                 preprocessed_suffix=args.preprocessed_suffix, preload=True, 
                 epoch=False)

    # Read visually responsive channels
    df= reader.read_channels_info(fname='visual_channels.csv')
    latency = df['latency'].loc[df['chan_name']==chan[0]].tolist()
    latency = latency[0]*1e-3
    
    hfa = reader.read_ecog()
    # Epoch HFA
    epocher = Epocher(condition='Face', t_prestim=args.t_prestim, t_postim = args.t_postim, 
                                 baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                                 tmax_baseline=args.tmax_baseline, mode=args.mode)
    epoch = epocher.log_epoch(hfa)
    # Downsample by factor of 2 
    epoch = epoch.copy().decimate(args.decim)
    time = epoch.times
    # Get trial data
    epoch = epoch.copy().pick(chan)
    X = epoch.copy().get_data()
    trial = X[itrial,0, :]
    # Get evoked data
    evok = np.mean(X, axis=0)
    evok = evok[0,:]
    sm = sem(X, axis=0)
    sm = sm[0,:]
    up_ci = evok + 1.96*sm
    down_ci = evok - 1.96*sm
    # Get histogram of prestimulus
    prestim = epoch.copy().crop(tmin=args.tmin_prestim, tmax=args.tmax_prestim)
    prestim = prestim.get_data()
    prestim = np.ndarray.flatten(prestim)
    amin = np.amin(prestim)
    baseline = np.mean(prestim)
    # Get histogram of postimulus
    postim = epoch.copy().crop(tmin=args.tmin_postim, tmax=args.tmax_postim)
    postim = postim.get_data()
    postim = np.ndarray.flatten(postim)
    amax = np.amax(postim)
    # Plot trial and pre.postim distribution
    f, ax = plt.subplots(2,2,)
    # Plot representative trial
    ax[0,0].plot(time, trial, color='b')
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel('Trial log HFA')
    ax[0,0].axvline(x=0, color='k')
    ax[0,0].axvline(x=latency, color='r', label='latency response')
    ax[0,0].axhline(y=baseline, color='k')
    ax[0,0].set_title('a)',loc='left')
    ax[0,0].legend()
    # Plot evoked response
    ax[1,0].plot(time, evok, color='b')
    ax[1,0].fill_between(time, down_ci, up_ci, alpha=0.4, color='b')
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].set_ylabel('Average log HFA ')
    ax[1,0].axvline(x=0, color='k')
    ax[1,0].axvline(x=latency, color='r', label='latency response')
    ax[1,0].axhline(y=baseline, color='k')
    ax[1,0].set_title('b)',loc='left')
    ax[1,0].legend()
    # Plot postimulus histogram
    sns.histplot(postim, stat = 'probability', bins=nbins, kde=True, ax=ax[0,1])
    ax[0,1].set_xlabel('Postimulus Amplitude')
    ax[0,1].set_ylabel('Probability')
    ax[0,1].set_xlim(left=amin, right=amax)
    ax[0,1].set_title('c)',loc='left')
    # Plot prestimulus histogram
    sns.histplot(prestim, stat = 'probability', bins=nbins, kde=True, ax=ax[1,1])
    ax[1,1].set_xlabel('Prestimulus Amplitude')
    ax[1,1].set_ylabel('Probability')
    ax[1,1].set_xlim(left=amin, right=amax)
    ax[1,1].set_title('d)',loc='left')
    plt.tight_layout()

fname = 'DiAs_visual_trial.pdf'
home = Path.home()
fpath = home.joinpath('thesis','overleaf_project','figures','method_figure')
plot_visual_trial(args, data_path,
                          chan = ['LTo1-LTo2'], itrial=2, nbins=50)
# Save figure
fpath = fpath.joinpath(fname)
plt.savefig(fpath)
#%% Plot visual vs non visual
# Need to read epoch data (use epo.fif)
def plot_visual_vs_non_visual(args, data_path):
    # Read visual chans
    for i, subject in enumerate(cohort):
        reader = EcogReader(data_path, subject=subject, stage=args.stage,
                         preprocessed_suffix=args.preprocessed_suffix, epoch=args.epoch)
        df_visual= reader.read_channels_info(fname='visual_channels.csv')
        visual_chans = df_visual['chan_name'].to_list()
        # Read hfb
        hfb = reader.read_ecog()
        hfb_visual = hfb.copy().pick_channels(visual_chans)
        hfb_nv = hfb.copy().drop_channels(visual_chans)
        baseline = hfb_visual.copy().crop(tmin=-0.5, tmax=0).get_data()
        baseline = np.average(baseline)
        # Plot event related potential of visual channels
        evok_visual = np.average(hfb_visual.get_data(), axis=0)
        # Plot event related potential of non visual channels
        evok_nv = np.average(hfb_nv.get_data(), axis=0)
        time = hfb_visual.times
        X = evok_visual
        mX = np.mean(X,0)
        semX = sem(X,0)
        up_ciX = mX + 1.96*semX
        down_ciX = mX - 1.96*semX
        Y = evok_nv
        semY = sem(Y,0)
        mY = np.mean(Y,0)
        up_ciY = mY + 1.96*semY
        down_ciY = mY - 1.96*semY
        # Plot visual vs non visual
        plt.subplot(2,2, i+1)
        plt.plot(time, mX, label='visual', color='b')
        plt.fill_between(time, down_ciX, up_ciX, alpha=0.3, color='b')
        plt.plot(time, mY, label='non visual', color = 'r')
        plt.fill_between(time, down_ciY, up_ciY, alpha=0.3, color='r')
        plt.axhline(y=baseline, color='k')
        plt.axvline(x=0, color='k')
        plt.xlabel('Time (s)')
        plt.ylabel(f'HFA subject {i} (dB)') # Data has been baseline rescaled
        plt.tight_layout()
        #plt.legend(loc='lower left', bbox_to_anchor=(1.02, 1.02))
fname = 'visual_vs_non_visual.pdf'
fpath = home.joinpath('thesis','overleaf_project','figures','method_figure')

plot_visual_vs_non_visual(args, data_path)
# Save figure
save_path = fpath.joinpath(fname)
plt.savefig(save_path)
#%% Plot hierachical ordering of channels
from scipy.stats import linregress
def plot_linreg(reg, xlabels, ylabels):
    """
    Plot linear regression between latency, Y and visual responsivity for
    visual channel hierarchy
    
    reg = [('Y','latency'), ('Y','visual_responsivity'),('latency', 'visual_responsivity'),
           ('Y','category_selectivity')]
    
    Regressors/regressands
    """
    fname = 'all_visual_channels.csv'
    fpath = result_path.joinpath(fname)
    df = pd.read_csv(fpath)
    # Remove outlier
    outlier = 'LTm5-LTm6'
    df = df[df['chan_name']!=outlier]
    for i, pair in enumerate(reg):
        x = df[pair[0]].to_numpy()
        y = df[pair[1]].to_numpy()
        xlabel = xlabels[i]
        ylabel = ylabels[i]
        plt.subplot(2,2,i+1)
        stats = linregress(x, y)
        xmin = np.amin(x)
        xmax = np.amax(x)
        ax = np.arange(xmin, xmax)
        ay = stats.slope*ax + stats.intercept
        plt.plot(ax, ay, color='r')
        plt.scatter(x,y, color='b')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        pval = '{:.1e}'.format(stats.pvalue)
        plt.annotate(f'r={round(stats.rvalue,2)}\n'f'p={pval}', 
                       xy = (0.6, 0.75), xycoords='axes fraction', fontsize = 12)
        plt.tight_layout()

reg = [('Y','latency'), ('Y','visual_responsivity'),('latency', 'visual_responsivity'),
           ('Y','category_selectivity')]
xlabels = ['Y axis (MNI)', 'Y axis (MNI)', 'latency (ms)', 'Y (MNI)']
ylabels = ['latency (ms)', 'responsivity (z)', 'responsivity (z)', 'selectivity (z)']
fpath = home.joinpath('thesis','overleaf_project','figures','method_figure')
figname = 'visual_hierarchy_corrected.pdf'
plot_linreg(reg, xlabels, ylabels)
fpath = fpath.joinpath(figname)
plt.savefig(fpath)
#%%
# from src.plotting_lib import plot_condition_ts
# fpath = home.joinpath('thesis','overleaf_project','figures','method_figure')
# plot_condition_ts(args, fpath, subject='DiAs', figname='_condition_ts.jpg')

#%% Plot rolling var

# def plot_rolling_var(df, fpath, momax=10, figname='rolling_var.pdf'):
#     """
#     This function plots results of rolling VAR estimation
#     """
#     cohort = ['AnRa', 'ArLa', 'DiAs']
#     nsub = len(cohort)
#     ic = ["aic", "bic", "hqc", "lrt"]
#     cdt = list(df["condition"].unique())
#     ncdt = len(cdt)
#     # Plot rolling var
#     f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
#     for c in range(ncdt):
#         for s in range(nsub):
#             for i in ic:
#                 time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
#                 morder = df[i].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
#                 ax[c,s].plot(time, morder, label=i)
#                 ax[c,s].set_ylim(0,momax)
#                 ax[0,s].set_title(f"Subject {s}")
#                 ax[c,0].set_ylabel(cdt[c])
#                 ax[2,s].set_xlabel("Time (s)")
#                 #ax[2,s].set_xticks(ticks)
                        
#     # legend situated upper right                
#     handles, labels = ax[c,s].get_legend_handles_labels()
#     f.legend(handles, labels, loc='upper right')
#     plt.tight_layout()
#     # Save figure
#     fpath = fpath.joinpath(figname)
#     #plt.savefig(fpath)


# result_path = Path('..','results')
# fname = 'rolling_var_estimation.csv'
# fpath = Path.joinpath(result_path, fname)
# df = pd.read_csv(fpath)

# figname = 'rolling_var.pdf'

# fpath = home.joinpath('thesis','overleaf_project','figures','method_figure')

# plot_rolling_var(df, fpath, momax=10, figname=figname)




# #%% Plot Spectral radius
# from src.plotting_lib import plot_rolling_specrad

# # Read input
# result_path = Path('..','results')
# fname = 'rolling_var_estimation.csv'
# fpath = Path.joinpath(result_path, fname)
# df = pd.read_csv(fpath)

# fpath = home.joinpath('thesis','overleaf_project','figures','method_figure')
# plot_rolling_specrad(df, fpath, ncdt =3, momax=10, figname='rolling_specrad.jpg')


# #%% Plot rolling window on multitrial

# # List conditions
# conditions = ['Rest', 'Face', 'Place']
# cohort = ['AnRa',  'ArLa', 'DiAs'];
# # Load functional connectivity matrix
# result_path = Path('../results')

# fname = 'rolling_multi_trial_fc.mat'
# fc_path = result_path.joinpath(fname)
# fc = loadmat(fc_path)
# fc = fc['dataset']
    
# figpath = home.joinpath('thesis','overleaf_project','figures')
# figname = 'cross_rolling_multi_mvgc.pdf'
# figpath = fpath.joinpath(figname)

# plot_multitrial_rolling_fc(fc, figpath, interaction='gGC' ,fc_type='gc')





