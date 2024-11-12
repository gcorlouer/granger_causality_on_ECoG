#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:28:23 2022
In this script we plot spectral mvgc in each condition and subjects
@author: guime
"""
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing_lib import EcogReader, parcellation_to_indices
from src.input_config import args
from scipy.io import loadmat
from pathlib import Path

# %%
# %matplotlib qt

conditions = [
    "Rest",
    "Face",
    "Place",
]
cohort = args.cohort
path = args.result_path
fname = "multi_trial_smvgc.mat"
fpath = path.joinpath(fname)
sgc = loadmat(fpath)
sgc = sgc["dataset"]
nsub = len(cohort)
ncdt = len(conditions)
# Saving figure
fig_path = Path("~", "PhD", "notes", "figures").expanduser()
fig_name = "cross_smvgc_ecog.png"

# %% Functions


def compute_freq_loss(s, freqs, frac=0.95):
    """
    Compute frequency after which psd lose more than 95% of its value
    """
    smax = np.amax(s)
    s_neglect = smax - frac * smax
    d = s - s_neglect
    d = np.abs(d)
    dmin = np.min(d)
    idx_neglect = np.where(d == dmin)[0][0]
    freq_loss = freqs[idx_neglect]
    return freq_loss


def plot_smvgc(sgc, args, pairs=["R->F", "F->R"]):
    """
    Plot spectral gc in each direction and subject, compare between
    preferred and non preferred stimuli
    """
    xticks = [0, 1, 10, 100]
    ng = len(pairs)
    fig, ax = plt.subplots(ng, nsub, sharex=True, sharey=True)
    for s, subject in enumerate(args.cohort):
        # Find retinotopic and face channels indices
        populations = ["R", "O", "F"]
        R_idx = populations.index("R")
        F_idx = populations.index("F")
        pair_idx = [(F_idx, R_idx), (R_idx, F_idx)]
        # Loop over pairs
        for ig, pair in enumerate(pairs):
            # Pick indices of pair we want to plot
            i = pair_idx[ig][0]
            j = pair_idx[ig][1]
            # Define spectral gc during rest as baseline
            f_b = sgc[0, s][2]["F"][0][0]
            # Loop over conditions
            for c, cdt in enumerate(conditions):
                f = sgc[c, s][2]["F"][0][0]
                # Baseline rescale
                f = np.divide(f, f_b, out=np.zeros_like(f), where=f_b != 0)
                freqs = sgc[c, s][2]["freqs"][0][0]
                ax[ig, s].plot(freqs, f[i, j, :], label=f"{cdt}")
                ax[ig, s].set_xscale("log")
                ax[ig, s].set_yscale("linear")
                ax[ig, s].set_xticks(xticks)
                ax[ig, s].set_xticklabels(["0", "1", "10", "100"])
                ax[ig, s].set_ylim(bottom=0.01, top=25)
                ax[ig, s].set_xlim(left=0.1, right=150)
                ax[ig, 0].set_ylabel(f"{pair}")
            ax[0, s].set_title(f"Spectral MVGC {subject}")
            ax[1, s].set_xlabel("frequency (Hz)")
    plt.legend()


# %%  Plot spectral GC along pairs x subjects (compare accross conditions)

pairs = ["R->F", "F->R"]
plot_smvgc(sgc, args, pairs=pairs)
fig_path = fig_path.joinpath(fig_name)
plt.savefig(fig_path)


# %% Compare top down sGC relative to bottum up accross conditions and subjects

# pairs = ['R->F', 'F->R']
# xticks = [0,1,10,100]
# ng = len(pairs)
# fig, ax = plt.subplots(1, nsub,sharex=True, sharey=True)
# for s, subject in enumerate(args.cohort):
#    reader = EcogReader(args.data_path, subject=subject)
#    df_visual = reader.read_channels_info(fname='visual_channels.csv')
#    # Find retinotopic and face channels indices
#    populations = parcellation_to_indices(df_visual, parcellation='group', matlab=False)
#    populations = list(populations.keys())
#    R_idx = populations.index('R')
#    F_idx = populations.index('F')
#    pair_idx = [(F_idx, R_idx),(R_idx, F_idx)]
#    for c, cdt in enumerate(conditions):
#        # Get smvgc
#        f = sgc[c,s][2]['F'][0][0]
#        # Top down smvgc
#        f_td = f[R_idx, F_idx, :]
#         # Bottum up smvgc
#        f_bu = f[F_idx, R_idx, :]
#        # Rescale top down by bottum up
#        f = np.divide(f_td,f_bu,out=np.zeros_like(f_td), where=f_bu!=0)
#        freqs =sgc[c,s][2]['freqs'][0][0]
#        freqs = freqs[:,0]
#        ax[s].plot(freqs, f, label = f'{cdt}')
#    ax[s].set_xscale('linear')
#    ax[s].set_yscale('linear')
#    ax[s].set_ylim(bottom = 0.01, top = 5)
#    ax[s].set_xlim(left = 0.1, right = 125)
#    ax[0].set_ylabel('Top down relative to bottom up smvgc')
#    ax[s].set_title(f"Spectral MVGC {subject}")
#    ax[s].set_xlabel('frequency (Hz)')
# plt.legend()
