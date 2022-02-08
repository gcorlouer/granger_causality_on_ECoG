%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script run functional connectivity analysis on time series of one
% subject. Estimates mutual information, then pairwise conditional 
% Granger causality (GC), then spectral GC. Script returns dataset that
% is going to be plotted in python.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise parameters

input_parameters;
cohort = {'AnRa',  'ArLa', 'DiAs'};
condition = {'Rest', 'Face', 'Place', 'baseline'};
field = {'time',  'condition', 'pair', 'subject','F'};
ncdt = length(condition);
nsub = length(cohort);
dataset = struct;
%% Load data

datadir = fullfile('~', 'projects', 'cifar', 'results');


for s = 1:nsub
    subject = cohort{s};
    fname = [subject '_condition_visual_ts.mat'];
    fpath = fullfile(datadir, fname);
    
    % Meta data about time series
    time_series = load(fpath);
    time = time_series.time; fs = double(time_series.sfreq);

    % Functional group indices
    indices = time_series.indices; fn = fieldnames(indices);
    
    for c=1:ncdt
        % Read conditions specific time series
        X = time_series.(condition{c});
        [n, m, N] = size(X);


        %% Detrend

        [X,~,~,~] = mvdetrend(X,pdeg,[]);

        %% Pairwise conditional MI estimation

        [MI, sig_MI] = ts_to_MI(X, 'q', q, 'mhtc', mhtc, 'alpha', alpha);

        %% Pairwise conditional GC estimation

         F= ts_to_var_pcgc(X,'morder', morder,...
                'regmode', regmode,'alpha', alpha,'mhtc', mhtc, 'LR', LR);
        %% Build dataset
        
        dataset(c,s).condition = condition{c};
        dataset(c,s).subject = subject;
        dataset(c,s).F = F;
        dataset(c,s).MI = MI;
    end
end

%% Save dataset for plotting in python

fname = 'pairwise_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')
