% EEG bands
bands = struct;
bands.theta = [4 7];
bands.alpha = [8 12];
bands.beta = [13 30];
bands.gamma = [32 60];
bands.hgamma = [60 120];

%% Functional connectivity on HFA

signal = 'hfa';
suffix = ['_condition_visual_' signal '.mat'];
band = [0 62]; % Band if downsampled to 125 Hz
connect = 'pairwise';
compare_condition_GC
connect = 'groupwise';
compare_condition_GC
compare_bu_td_gc_permtest
%% Functional connectivity on ECoG

signal = 'lfp';
suffix = ['_condition_visual_' signal '.mat'];
band_names = fieldnames(bands);
nband = length(band_names);
for ib=1:nband
    band = bands.(band_names{ib});
    connect = 'pairwise';
    compare_condition_GC
    connect = 'groupwise';
    compare_condition_GC
    compare_bu_td_gc_permtest
end