#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:55:11 2022
This script contain plotting functions for the project
@author: guime
"""

import mne 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src.preprocessing_lib import EcogReader, Epocher, prepare_condition_scaled_ts
from pathlib import Path
from scipy.stats import sem, linregress, ranksums
from mne.stats import fdr_correction



#%% Style parameters

plt.style.use('ggplot')
fig_width = 17  # figure width in cm
inches_per_cm = 0.393701               # Convert cm to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width*inches_per_cm  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'lines.linewidth': 1.5,
          'axes.labelsize': 12,
          'axes.titlesize': 8,
          'font.size': 12,
          'legend.fontsize': 8,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

#%% Preprocessing plots

#def plot_stimuli_presentation():

def plot_narrow_broadband(args, fpath, fname = 'DiAs_narrow_broadband_stim.pdf', 
                          chan = ['LTo1-LTo2'], tmin=500, tmax=506):
    """
    Plot the narrowband envelope and broadband hfa between (tmin, tmax) for 
    one subject and one reperesentative channel
    """
    # Read EcoG data
    reader = EcogReader(args.data_path, stage=args.stage,
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
    reader = EcogReader(args.data_path, stage=args.stage,
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
    f, ax = plt.subplots(2,1)
    # Plot narrow band
    ax[0].plot(time, X_filt, label='ECoG')
    ax[0].plot(time, narrow_envelope, label='Narrow band envelope')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Signal (V)')
    ax[0].legend(bbox_to_anchor=(0.3,1.02), loc='lower left')
    ax[0].set_title('a)', loc='left')
    # Plot broadband
    ax[1].plot(time, broadband_envelope)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Amplitude (muV)')
    ax[1].set_title('b)', loc='left')
    plt.tight_layout()
    # Save figure
    fpath = fpath.joinpath(fname)
    plt.savefig(fpath)

# Plot log vs non log trials
    
def plot_log_trial(args, fpath, fname = 'DiAs_log_trial.pdf', 
                          chan = ['LTo1-LTo2'], itrial=2, nbins=50):
    """
    This function plot log trial vs non log trial and their respective 
    distribution
    """
    reader = EcogReader(args.data_path, stage=args.stage,
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
    X = epoch.copy().get_data()
    # Results in muV
    X = X*1e6
    # Trial
    trial = X[itrial,0, :]
    # Log epoch HFA
    l_epoch = epocher.log_epoch(hfa)
    l_epoch = l_epoch.copy().decimate(args.decim)
    l_epoch = l_epoch.copy().pick(chan)
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
    # Plot log trial
    ax[0,1].plot(time, l_trial)
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel('Log HFA (muV)')
    # Plot hfa histogram
    sns.histplot(x, stat = 'probability', bins=nbins, kde=True, ax=ax[1,0])
    ax[1,0].set_xlabel('HFA (muV)')
    ax[1,0].set_ylabel('Probability')
    ax[1,0].set_xlim(left=0, right=amax)

    # Plot log hfa histogram
    sns.histplot(l_x, stat = 'probability', bins=nbins, kde=True, ax=ax[1,1])
    ax[1,1].set_xlabel('Log HFA')
    ax[1,1].set_ylabel('Probability')
    #ax[1,1].set_xlim(left=0, right=amax)
    plt.tight_layout()
    # Save figure
    fpath = fpath.joinpath(fname)
    plt.savefig(fpath)
#%% Visual channel detection plots

def plot_visual_trial(args, fpath, fname = 'DiAs_visual_trial.pdf', 
                          chan = ['LTo1-LTo2'], itrial=2, nbins=50):
    """
    Plot visually responsive trial and pre/postim distribution
    """
    reader = EcogReader(args.data_path, stage=args.stage,
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
    ax[0,0].set_ylabel('Log HFA')
    ax[0,0].axvline(x=0, color='k')
    ax[0,0].axvline(x=latency, color='r', label='latency response')
    ax[0,0].axhline(y=baseline, color='k')
    ax[0,0].set_title('a)',loc='left')
    ax[0,0].legend()
    # Plot evoked response
    ax[1,0].plot(time, evok, color='b')
    ax[1,0].fill_between(time, down_ci, up_ci, alpha=0.6)
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].set_ylabel('Log HFA')
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
    # Save figure
    fpath = fpath.joinpath(fname)
    plt.savefig(fpath)
    

#%% Visual channels detection plots

def plot_visual_vs_non_visual(args, fpath, fname='visual_vs_non_visual.pdf'):
    # Read visual chans
    for i, subject in enumerate(args.cohort):
        reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
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
        evok_visual = hfb_visual.average()
        # Plot event related potential of non visual channels
        evok_nv = hfb_nv.average()
        time = evok_visual.times
        X = evok_visual.get_data()
        mX = np.mean(X,0)
        Y = evok_nv.get_data()
        mY = np.mean(Y,0)
        # Plot visual vs non visual
        plt.subplot(3,3, i+1)
        plt.plot(time, mX, label='visual', color='b')
        plt.plot(time, mY, label='non visual', color = 'r')
        plt.axhline(y=baseline, color='k')
        plt.axvline(x=0, color='k')
        plt.xlabel('Time (s)')
        plt.ylabel(f'HFA {subject}')
        plt.tight_layout()
        #plt.legend(loc='lower left', bbox_to_anchor=(1.02, 1.02))
        # Save figure
        save_path = fpath.joinpath(fname)
        plt.savefig(save_path)
        
def plot_linreg(reg, save_path, figname = 'visual_hierarchy.pdf'):
    """
    Plot linear regression between latency, Y and visual responsivity for
    visual channel hierarchy
    
    reg = [('Y','latency'), ('Y','visual_responsivity'),('latency', 'visual_responsivity'),
           ('Y','category_selectivity')]
    
    Regressors/regressands
    """
    result_path = Path('../results')
    fname = 'all_visual_channels.csv'
    fpath = result_path.joinpath(fname)
    df = pd.read_csv(fpath)
    # Remove outlier
    outlier = 'LTm5-LTm6'
    df = df[df['chan_name']!=outlier]
    for i, pair in enumerate(reg):
        x = df[pair[0]].to_numpy()
        y = df[pair[1]].to_numpy()
        xlabel = pair[0]
        ylabel = pair[1]
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
        plt.annotate(f'r2={round(stats.rvalue,2)}\n p={round(stats.pvalue,3)}', 
                       xy = (0.75, 0.75), xycoords='axes fraction', fontsize = 8)
        plt.tight_layout()
        fpath = save_path.joinpath(figname)
        plt.savefig(fpath)

#%% Visual channels classification plots

def plot_condition_ts(args, fpath, subject='DiAs', figname='_condition_ts.pdf'):
    # Prepare condition ts
    ts = prepare_condition_scaled_ts(args.data_path, subject=subject, 
                                     stage='preprocessed', matlab = False,
                     preprocessed_suffix='_hfb_continuous_raw.fif', decim=2,
                     epoch=False, t_prestim=-0.5, t_postim=1.75, tmin_baseline = -0.5,
                     tmax_baseline = 0, tmin_crop=-0.5, tmax_crop=1.75)
    # Prepare inputs for plotting
    conditions = ['Rest','Face','Place']
    populations = ts['indices'].keys()
    time = ts['time']
    baseline = ts['baseline']
    baseline = np.average(baseline)
    # Plot condition ts
    f, ax = plt.subplots(3,1, sharex=True, sharey=True)
    for i, cdt in enumerate(conditions):
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
            ax[i].plot(time, evok, label=pop)
            ax[i].fill_between(time, down_ci, up_ci, alpha=0.6)
            ax[i].axvline(x=0, color ='k')
            ax[i].axhline(y=baseline, color='k')
            ax[i].set_ylabel(f'{cdt} (dB)')
            ax[0].legend()
    ax[2].set_xlabel('time (s)')
    plt.tight_layout()
    fname = subject + figname
    fpath = fpath.joinpath(fname)
    plt.savefig(fpath)



#%% VAR model estimation
    
def plot_rolling_var(df, fpath, ncdt =3, momax=10, figname='rolling_var.pdf'):
    """
    This function plots results of rolling VAR estimation
    """
    cohort = list(df["subject"].unique())
    nsub = len(cohort)
    ic = ["aic", "bic", "hqc", "lrt"]
    cdt = list(df["condition"].unique())
    ticks = [0, 0.8]
    # Plot rolling var
    f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
    for c in range(ncdt):
        for s in range(nsub):
            for i in ic:
                time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
                morder = df[i].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
                ax[c,s].plot(time, morder, label=i)
                ax[c,s].set_ylim(0,momax)
                ax[0,s].set_title(cohort[s])
                ax[c,0].set_ylabel(cdt[c])
                ax[2,s].set_xlabel("Time (s)")
                ax[2,s].set_xticks(ticks)
                        
    # legend situated upper right                
    handles, labels = ax[c,s].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    # Save figure
    fpath = fpath.joinpath(figname)
    plt.savefig(fpath)

def plot_rolling_specrad(df, fpath, ncdt =3, momax=10, figname='rolling_specrad.pdf'):
    """
    Plot spectral radius along rolling window accross all subjects
    """
    cohort = list(df["subject"].unique())
    nsub = len(cohort)
    cdt = list(df["condition"].unique())
    ticks = [0, 0.8]
    # Plot rolling spectral radius
    f, ax = plt.subplots(ncdt, nsub, sharex=True, sharey=True)
    for c in range(ncdt):
        for s in range(nsub):
            time = df["time"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
            rho = df["rho"].loc[(df["subject"]==cohort[s]) & (df["condition"]==cdt[c])].to_numpy()
            ax[c,s].plot(time, rho, label="Spectral radius")
            ax[c,s].set_ylim(0.6,1)
            ax[0,s].set_title(cohort[s])
            ax[c,0].set_ylabel(cdt[c])
            ax[2,s].set_xlabel("Time (s)")
            ax[2,s].set_xticks(ticks)
    # Legend            
    handles, labels = ax[c,s].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    # Save figure
    fpath = fpath.joinpath(figname)
    plt.savefig(fpath)


#%% Plot mvgc results full stimuli

def sort_populations(populations, order= {'R':0,'O':1,'F':2}):
    """
    Sort visually responsive population along specific order
    Return sorted indices to permute GC/MI axis along wanted order
    """
    L = populations
    pop_order = [(i, idx, order[i]) for idx, i in enumerate(L)]
    L_sort = sorted(pop_order, key= lambda pop_order: pop_order[2])
    L_pair = [(L_sort[i][0], L_sort[i][1]) for i in range(len(L_sort))]
    pop_sort = [L_pair[i][0] for i in range(len(L_pair))]
    idx_sort = [L_pair[i][1] for i in range(len(L_pair))]
    return idx_sort, pop_sort


def plot_multi_fc(fc, populations, fpath, s=2, sfreq=250,
                                 rotation=90, tau_x=0.5, tau_y=0.8):
    """
    This function plot pairwise mutual information and transfer entropy matrices 
    as heatmaps against the null distribution for a single subject
    s: Subject index
    tau_x: translattion parameter for x coordinate of statistical significance
    tau_y: translattion parameter for y coordinate of statistical significance
    rotation: rotation of xticks and yticks labels
    te_max : maximum value for TE scale
    mi_max: maximum value for MI scale
    """
    idx_sort, populations = sort_populations(populations)
    (ncdt, nsub) = fc.shape
    fig, ax = plt.subplots(ncdt-1,2)
    for c in range(ncdt-1): # Consider resting state as baseline
        condition =  fc[c,s]['condition'][0]
        # Granger causality matrix
        f = fc[c,s]['F']
        sig_gc = fc[c,s]['sigF']
        # Mutual information matrix
        mi = fc[c,s]['MI']
        sig_mi = fc[c,s]['sigMI']     
        # Permutes axes along wanted order
        f = f[idx_sort,:]
        f = f[:, idx_sort]
        mi = mi[idx_sort,:]
        mi = mi[:, idx_sort]
        sig_gc = sig_gc[idx_sort,:]
        sig_gc = sig_gc[:,idx_sort]
        sig_mi = sig_mi[idx_sort,:]
        sig_mi = sig_mi[:,idx_sort]
        # Make ticks label
        pop_ticks = [0]*len(populations)
        pop_ticks[0] = populations[0]
        for i in range(1,len(populations)):
            if populations[i] == populations[i-1]:
                pop_ticks[i]='-'
            else:
                pop_ticks[i]=populations[i]
        # Plot MI as heatmap
        g = sns.heatmap(mi, xticklabels=pop_ticks,
                        yticklabels=pop_ticks, cmap='YlOrBr', ax=ax[c,0])
        g.set_yticklabels(g.get_yticklabels(), rotation = rotation)
        # Position xticks on top of heatmap
        ax[c, 0].xaxis.tick_top()
        ax[0,0].set_title('Mutual information (bit)')
        ax[c, 0].set_ylabel(condition)
        # Plot GC as heatmap
        g = sns.heatmap(f, xticklabels=pop_ticks,
                        yticklabels=pop_ticks, cmap='YlOrBr', ax=ax[c,1])
        g.set_yticklabels(g.get_yticklabels(), rotation = rotation)
        # Position xticks on top of heatmap
        ax[c, 1].xaxis.tick_top()
        ax[c, 1].set_ylabel('Target')
        ax[0,1].set_title('Transfer entropy (bit/s)')
        # Plot statistical significant entries
        for y in range(f.shape[0]):
            for x in range(f.shape[1]):
                if sig_mi[y,x] == 1:
                    ax[c,0].text(x + tau_x, y + tau_y, '*',
                             horizontalalignment='center', verticalalignment='center',
                             color='k')
                else:
                    continue
                if sig_gc[y,x] == 1:
                    ax[c,1].text(x + tau_x, y + tau_y, '*',
                             horizontalalignment='center', verticalalignment='center',
                             color='k')
                else:
                    continue
        plt.tight_layout()
        plt.savefig(fpath)


#%% Plot single trial GC results

# Compute z score of single distirbution
def single_pfc_stat(fc, cohort, subject ='DiAs', baseline= 'baseline', 
                    single='single_F', alternative='two-sided'):
    """
    Compare functional connectivity (GC or MI) during baseline w.r.t a specific
    condition such as Face or Place presentation.
    
    Parameters:
    single= 'single_F' or 'single_MI'
    cohort = ['AnRa',  'ArLa', 'DiAs']
    baseline = 'baseline' or 'Rest' 
    """
    # Index conditions
    cdt = {'Rest':0, 'Face':1, 'Place':2, 'baseline':3}
    # Make subject dictionary
    keys = cohort
    sub_dict = dict.fromkeys(keys)
    # Index subjects
    for idx, sub in enumerate(cohort):
        sub_dict[sub] = idx
    # Comparisons performed for FC
    comparisons = [(cdt[baseline],cdt['Face']), (cdt[baseline], cdt['Place']), 
                   (cdt[baseline], cdt['Face'])]
    ncomp = len(comparisons)
    # Subject index of interest
    s = sub_dict[subject]
    # FGet shape of functional connectivity matrix 
    f = fc[0,s][single]
    (n,n,N) = f.shape
    # Initialise statistics
    z = np.zeros((n,n,ncomp))
    pval =  np.zeros((n,n,ncomp))
    # Compare fc during baseline and one condition
    for icomp in range(ncomp):
        cb = comparisons[icomp][0]
        c = comparisons[icomp][1]
        # Baseline functional connectivity
        fb = fc[cb,s][single]
        # Condition-specific functional connectivity
        f = fc[c,s][single]
        # Compute z score and pvalues
        for i in range(n):
            for j in range(n):
                z[i,j, icomp], pval[i,j,icomp] = ranksums(f[i,j,:], fb[i,j,:], 
                 alternative=alternative)
    rejected, pval_corrected = fdr_correction(pval,alpha=0.05)
    sig = rejected
    return z, sig, pval



def plot_single_trial(z, sig, populations):
    """
    """
    (n,n,ncomp) = z.shape
    f, ax = plt.subplots(ncomp,2)
    zmax = np.amax(z)
    zmin = np.amin(z)
    for icomp in range(ncomp):
        g = sns.heatmap(z[:,:,icomp], vmin=zmin, vmax=zmax, xticklabels=populations,
                            yticklabels=populations, cmap='YlOrBr', ax=ax[icomp,0])
        g.set_yticklabels(g.get_yticklabels(), rotation = 90)
        # Position xticks on top of heatmap
        ax[icomp, 0].xaxis.tick_top()
        ax[icomp, 0].set_ylabel('Target')
        ax[0,0].set_title(' Z score')
        g = sns.heatmap(sig[:,:,icomp], xticklabels=populations,
                            yticklabels=populations, cmap='YlOrBr', ax=ax[icomp,1])
        g.set_yticklabels(g.get_yticklabels(), rotation = 90)
        # Position xticks on top of heatmap
        ax[icomp, 1].xaxis.tick_top()
        ax[0,1].set_title('Significance')
















