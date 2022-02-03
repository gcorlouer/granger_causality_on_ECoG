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
    fname = [subject '_condition_ts.mat'];
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

        %% Pairwise conditional Spectral GC estimation

         f = ts_to_var_spcgc(X, 'regmode',regmode, 'morder',morder,...
                'fres',fres, 'fs', fs);
        %% Build dataset
        
        dataset(c,s).condition = condition{c};
        dataset(c,s).subject = subject;
        dataset(c,s).F = F;
        dataset(c,s).MI = MI;
    end
end
        %% Build dataset
%         % Problem: ng, nk, nl vary across subjects!
%         ng = length(fn);
%         for i=1:ng
%             for j=1:ng
%                 % Indices of individual pairs within groups
%                 nk = length(indices.(fn{i}));
%                 nl = length(indices.(fn{j}));
%                 for k =1:nk
%                     for l=1:nl
%                         dataset(k,l,c,s).from = indices.(fn{i})(k);
%                         dataset(k,l,c,s).to = indices.(fn{j})(l);
%                         dataset(k,l,c,s).pair = [fn{j} '->' fn{i}];
%                         dataset(k,l,c,s).condition = condition{c};
%                         dataset(k,l,c,s).subject = subject;
%                         dataset(k,l,c,s).F = F(k,l);
%                         dataset(k,l,c,s).MI = MI(k,l);
%                     end
%                 end
%             end
%         end
%     end
% end

%% Save dataset for plotting in python

%lenData = numel(dataset);
%dataset = reshape(dataset, lenData, 1);

%df = struct2table(dataset);
fname = 'pairwise_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')
%writetable(df, fpath)