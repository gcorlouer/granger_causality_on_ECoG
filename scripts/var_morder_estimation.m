%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script estimate VAR model order from input time series
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input data parameters
if ~exist('subject', 'var') subject = 'DiAs'; end
if ~exist('momax', 'var') momax = 20; end
if ~exist('regmode', 'var') regmode = 'OLS'; end

alpha = 0.05;
pacf = true;
plotm = 0;
verb = 0;

%% Load data

datadir = fullfile('~', 'projects', 'CIFAR', 'CIFAR_data', 'iEEG_10', ... 
    'subjects', subject, 'EEGLAB_datasets', 'preproc');
fname = [subject, '_ts_visual.mat'];
fpath = fullfile(datadir, fname);

time_series = load(fpath);

X = time_series.data;
[nchan, nobs, ntrial, ncat] = size(X);

%% Estimate model order parameters

for i=1:ncat
    [moaic(i),mobic(i),mohqc(i),molrt(i)] = tsdata_to_varmo(X(:,:,:,i), ... 
        momax,regmode,alpha,pacf,plotm,verb);
end

