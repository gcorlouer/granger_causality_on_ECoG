# Import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
# Import classes
from src.preprocessing_lib import EcogReader, Epocher
# Import functions
from src.preprocessing_lib import visual_indices, parcellation_to_indices
from mne.time_frequency import tfr_morlet
from mne.viz import centers_to_edges

# Import arguments
#%% Define functions


def compute_group_power(args, tf_args, subject='DiAs', group='R', 
                        condition='Face'):
    """
    Compute power of visually responsive in a specific group epochs 
    for time freq analysis
    Input: 
        args: arguments from input_config
        tf_args: arguments to run  cross_time_freq_analysis
        subject: subject name
        group: group of channels name
        condition: condition name
    """
    # Read ECoG
    reader = EcogReader(args.data_path, subject=subject, stage=args.stage,
                         preprocessed_suffix=args.preprocessed_suffix,
                         epoch=args.epoch)
    raw = reader.read_ecog()
    # Read visually responsive channels
    df_visual = reader.read_channels_info(fname='visual_channels.csv')
    visual_chans = df_visual['chan_name'].to_list()
    raw = raw.pick_channels(visual_chans)
    # Get visual channels from functional group
    indices = visual_indices(args)
    group_indices = indices[group]
    group_chans = [visual_chans[i] for i in group_indices]
    print(f'\n {group} channels are {group_chans} \n')
    # Epoch raw ECoG
    epocher = Epocher(condition=condition, t_prestim=args.t_prestim, t_postim = args.t_postim, 
                             baseline=None, preload=True, tmin_baseline=args.tmin_baseline, 
                             tmax_baseline=args.tmax_baseline, mode=args.mode)
    epochs = epocher.epoch(raw)
    # High pass filter
    epochs = epochs.filter(l_freq=tf_args.l_freq, h_freq=None)
    # Downsample
    epochs = epochs.decimate(args.decim)
    times = epochs.times
    # Pick channels
    epochs = epochs.pick(group_chans)
    # Compute time frequency with Morlet wavelet 
    freqs = get_freqs(tf_args)
    n_cycles = freqs/2
    power = tfr_morlet(epochs, freqs, n_cycles, return_itc=False)
    # Apply baseline correction
    baseline = (tf_args.tmin_baseline, tf_args.tmax_baseline)
    print(f"\n Morlet wavelet: rescaled with {tf_args.mode}")
    print(f"\n Condition is {condition}\n")
    power.apply_baseline(baseline=baseline, mode=tf_args.mode)
    power = power.data
    power = np.average(power, axis=0)
    return power, times, freqs


def get_freqs(tf_args):
    """Get frequencies for time frequency analysis"""
    fmin = tf_args.fmin
    nfreqs = tf_args.nfreqs
    sfreq = tf_args.sfreq
    fmax = sfreq/2
    fres = (fmax + fmin - 1)/nfreqs
    freqs = np.arange(fmin, fmax, fres)
    return freqs


def plot_tf(fpath, subject='DiAs',vmax=25):
    """Plot time frequency"""
    # Parameter specfics to plotting time frequency
    fname = "tf_power_dataframe.pkl"
    fpath = fpath.joinpath(fname)
    df = pd.read_pickle(fpath)
    conditions = ['Rest', 'Face', 'Place']
    groups = ['R','O','F']
    ngroup = 3
    ncdt = 3
    fig, ax = plt.subplots(ngroup, ncdt, sharex=True, sharey=True)
    # Loop over conditions and groups
    for i, condition in enumerate(conditions):
        for j, group in enumerate(groups):
            power = df['power'].loc[df['subject']==subject].loc[df['condition']==condition].loc[df['group']==group]
            freqs = df['freqs'].loc[df['subject']==subject].loc[df['condition']==condition].loc[df['group']==group]
            time = df['time'].loc[df['subject']==subject].loc[df['condition']==condition].loc[df['group']==group]
            power = power.iloc[0]
            freqs = freqs.iloc[0]
            time = time.iloc[0]
            x, y = centers_to_edges(time * 1000, freqs)
            mesh = ax[i,j].pcolormesh(time, freqs, power, cmap='RdBu_r', vmax=vmax, vmin=-vmax)
            ax[i,j].set_title(f'{group} Power during {condition}')
            ax[i,j].set(ylim=freqs[[0, -1]], xlabel='Time (ms)', ylabel='Freq (Hz)')
    fig.colorbar(mesh)
    plt.tight_layout()
    plt.show()

        