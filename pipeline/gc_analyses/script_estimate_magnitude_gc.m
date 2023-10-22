bands = struct;
bands.theta = [4 7];
bands.alpha = [8 12];
bands.beta = [13 30];
bands.gamma = [32 60];
bands.hgamma = [60 120];

signal = 'hfa';
connect = 'pairwise';
estimate_magnitude_band_gc
connect = 'groupwise';
estimate_magnitude_band_gc
signal = 'lfp';
band_names = fieldnames(bands);
nband = length(band_names);
for ib=1:nband
    band = bands.(band_names{ib});
    connect = 'pairwise';
    estimate_magnitude_band_gc
    connect = 'groupwise';
    estimate_magnitude_band_gc
end