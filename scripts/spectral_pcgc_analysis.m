%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script run spectral GC analysis on time series of one
% subject. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_parameters;
fres = 1024;

%% Load data

datadir = fullfile('~', 'projects', 'CIFAR', 'CIFAR_data', 'results');
fname = 'condition_ts_visual.mat';
fpath = fullfile(datadir, fname);

time_series = load(fpath);

X = time_series.data;
[nchan, nobs, ntrial, ncat] = size(X);
sub_id = time_series.sub_id;

fname = [sub_id fname];
fpath = fullfile(datadir, fname);
save(fpath, 'time_series');

%%

for i=1:ncat
    f(:,:,:,i) = ts_to_var_spcgc(X(:,:,:,i), 'regmode',regmode, 'morder',morder,...
        'fres',fres, 'fs', fs);
end
%% Save file

fname = [subject 'spectral_GC.mat'];
fpath = fullfile(datadir, fname);

save(fpath, 'f')