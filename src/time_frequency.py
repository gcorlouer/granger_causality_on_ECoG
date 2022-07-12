

def visual_indices(args, subject='DiAs'):
    """
    Return indices of each functional group for a given subject
    Input: 
        - data_path (string): where data of cifar project is stored 
        - subject (string): subject name
    Output:
        - indices (dict): indices of each functional group
    """
    # Read visual channel dataframe
    reader = EcogReader(args.data_path, subject=subject)
    df_visual = reader.read_channels_info(fname=args.channels)
    # Return indices of functional groups from visual channel dataframe
    indices = parcellation_to_indices(df_visual, parcellation='group', matlab=False)
    return indices 

def compute_group_power(args, freqs, group='F', condition = 'Face', l_freq = 0.01,
                 baseline=(-0.5, 0), mode = 'zscore'):
    """Compute power of visually responsive in a specific group epochs 
    for time freq analysis"""
    # Read ECoG
    reader = EcogReader(args.data_path, subject=args.subject, stage=args.stage,
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
    epochs = epochs.filter(l_freq=l_freq, h_freq=None)
    # Downsample
    epochs = epochs.decimate(args.decim)
    times = epochs.times
    # Pick channels
    epochs = epochs.pick(group_chans)
    # Compute time frequency with Morlet wavelet 
    n_cycles = freqs/2
    power = tfr_morlet(epochs, freqs, n_cycles, return_itc=False)
    # Apply baseline correction
    baseline = (args.tmin_baseline, args.tmax_baseline)
    print(f"\n Computing group power from morlet wavelet: rescale with {mode}")
    print(f"\n Condition is {condition}\n")
    power.apply_baseline(baseline=baseline, mode=mode)
    power = power.data
    power = np.average(power,axis=0)
    return power, times