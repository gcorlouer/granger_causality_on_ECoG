# Import numpy
import numpy as np
# Import classes
from src.preprocessing_lib import EcogReader, Epocher
# Import functions
from src.preprocessing_lib import visual_indices, parcellation_to_indices
from mne.time_frequency import tfr_morlet
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
    return power, times


def get_freqs(tf_args):
    """Get frequencies for time frequency analysis"""
    fmin = tf_args.fmin
    nfreqs = tf_args.nfreqs
    sfreq = tf_args.sfreq
    fmax = sfreq/2
    fres = (fmax + fmin - 1)/nfreqs
    freqs = np.arange(fmin, fmax, fres)
    return freqs