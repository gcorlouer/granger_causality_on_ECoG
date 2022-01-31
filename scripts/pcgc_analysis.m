%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script run functional connectivity analysis on time series of one
% subject. Estimates mutual information, then pairwise conditional 
% Granger causality (GC), then spectral GC.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise parameters

input_parameters;
cohort = {'AnRa',  'ArLa', 'DiAs'};
condition = {'Rest', 'Face', 'Place', 'baseline'};
field = {'time',  'condition', 'pair', 'subject','F'};
ncdt = length(condition);
nsub = length(cohort);
dfc = struct;
%% Load data

datadir = fullfile('~', 'projects', 'cifar', 'results');


for s = 1:nsub
    subject = cohort{s};
    fname = [subject '_condition_ts.mat'];
    fpath = fullfile(datadir, fname);
    
    % Meta data about time series
    time_series = load(fpath);
    time = time_series.time; fs = double(time_series.sfreq);

    % Functional group indices
    indices = time_series.indices; fn = fieldnames(indices);

    % Initialise information criterion
    moaic = cell(ncdt,1);
    mobic =  cell(ncdt,1);
    mohqc =  cell(ncdt,1);
    molrt =  cell(ncdt,1);
    
    for c=1:ncdt
        % Read conditions specific time series
        X = time_series.(condition{c});
        [n, m, N] = size(X);


        %% Detrend and demean

        [X,~,~,~] = mvdetrend(X,pdeg,[]);

        %% Pairwise conditional MI estimation

        [MI, sig_MI] = ts_to_MI(X, 'q', q, 'mhtc', mhtc, 'alpha', alpha);

        %% Pairwise conditional GC estimation

         F= ts_to_var_pcgc(X,'morder', morder,...
                'regmode', regmode,'alpha', alpha,'mhtc', mhtc, 'LR', LR);

        %% Pairwise conditional Spectral GC estimation

         f = ts_to_var_spcgc(X, 'regmode',regmode, 'morder',morder,...
                'fres',fres, 'fs', fs);
        %% Save results
        dfc(c,s).condition = condition{c};
        dfc(c,s).subject = subject;
        dfc(c,s).F = F;
        dfc(c,s).MI = MI;
        dfc(c,s).f = f;
        dfc(c,s).sfreq = fs;
        fname = [sub_id '_pairwise_dfc.mat'];
        fpath = fullfile(datadir, fname);

        save(fpath, 'dfc')
    end
end
%% Plot results
% minF = 0;
% maxF = max(F,[],'all');
% for i=1:ncat
%     subplot(2,2,i)
%     imshow(imcomplement(mat2gray(F(:,:,i))))
% end