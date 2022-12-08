% In this script we run 2 chan GC analysis
bands = struct;
bands.delta = [1 4];
bands.theta = [4 7];
bands.alpha = [8 12];
bands.beta = [13 30];
bands.gamma = [32 60];
bands.hgamma = [60 120];
band_names = fieldnames(bands);
nband = length(band_names);
subject = 'DiAs';
% Analysis on HFA
signal ='hfa';
band = [0 62];
validate_analysis_2_chans
two_chan_TD_vs_BU
two_chans_compare_gc

% Analysis on lfp
signal ='lfp';
for ib=1:nband
    band = bands.(band_names{ib});
    validate_analysis_2_chans
    two_chan_TD_vs_BU
    two_chans_compare_gc
end
    