%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script run functional connectivity analysis on time series of one
% subject. Estimates mutual information, then pairwise conditional 
% Granger causality (GC) on multitrial and
% single trial. The script returns a dataset that is going to be anlayse 
% and plotted in python.
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

        pMI = ts_to_MI(X, 'q', q, 'mhtc', mhtc, 'alpha', alpha);
        
        %% Pairwise conditional GC estimation
        % VAR model estimation
        VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
        V = VAR.V;
        A = VAR.A;
        disp(VAR.info)
        % Pairwise conditional GC estimation
        % Single regression
        F = var_to_pwcgc(A,V);
        F(isnan(F))=0;
        % Compute significance against null distribution
        nx = 1; ny=1;nz = n-nx-ny; p=morder;
        % Dual regression
        stat = var_to_pwcgc_tstat(X,V,morder,regmode,tstat);
        pval = mvgc_pval(stat,tstat,nx,ny,nz,p,m,N);
        [sigF, pvalF] = significance(pval,alpha,mhtc,[]);
        sigF(isnan(sigF))=0;
        %% Estimate single trial distributions
        single_MI = zeros(n,n,N);
        single_F = zeros(n,n,N);
        for i=1:N
            [single_MI(:,:,i), ~] = ts_to_MI(X(:,:,i), 'q', q, ...
                'mhtc', mhtc, 'alpha', alpha);
            single_F(:,:,i)= ts_to_var_pcgc(X(:,:,i),'morder', morder,...
                'regmode', regmode,'alpha', alpha,'mhtc', mhtc, 'LR', LR);
            % Remove bias
            nx=1; ny=1; nz = n-nx-ny;
            bias = mvgc_bias('LR',nx,ny,nz,morder,m,N);
            single_F(:,:,i) = single_F(:,:,i) - bias;
        end
        %% Build dataset
        dataset(c,s).subject = subject;
        dataset(c,s).condition = condition{c};
        dataset(c,s).pMI = pMI; 
        dataset(c,s).single_MI = single_MI;
        dataset(c,s).F = F;
        dataset(c,s).sigF = sigF;
        dataset(c,s).pvalF = pvalF;
        dataset(c,s).single_F = single_F;
    end
end

%% Save dataset for plotting in python

fname = 'pairwise_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')
