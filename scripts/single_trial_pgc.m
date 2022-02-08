%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script run functional connectivity analysis on time series 
% Estimates mutual information, then pairwise conditional 
% Granger causality (GC) on single trial. Script returns dataset that
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

subject = 'DiAs';

fname = [subject '_condition_visual_ts.mat'];
fpath = fullfile(datadir, fname);

% Meta data about time series
time_series = load(fpath);
time = time_series.time; fs = double(time_series.sfreq);

% Functional group indices
indices = time_series.indices; fn = fieldnames(indices);

MI = cell(ncdt,1);
F = cell(ncdt,1);
bias = cell(ncdt,1);
for c=1:ncdt
    % Read conditions specific time series
    X = time_series.(condition{c});
    [n, m, N] = size(X);

    %% Detrend

    [X,~,~,~] = mvdetrend(X,pdeg,[]);
    
    
    MI{c} = zeros(n,n,N);
    F{c} = zeros(n,n,N);
    nx = 1; ny = 1; nz=n-nx-ny;
    bias{c} = mvgc_bias(tstat,nx,ny,nz,morder,m,N);
    for i=1:N
        %% Pairwise conditional MI estimation
        [MI{c}(:,:,i), ~] = ts_to_MI(X(:,:,i), 'q', q, 'mhtc', mhtc, 'alpha', alpha);

        %% Pairwise conditional GC estimation

         F{c}(:,:,i)= ts_to_var_pcgc(X(:,:,i),'morder', morder,...
                'regmode', regmode,'alpha', alpha,'mhtc', mhtc, 'LR', LR);
            
        % Debias by null distribution (because number of observations not 
        % the same
        F{c}(:,:,i) = F{c}(:,:,i) - bias{c};
    end
end

%% Run statistical tests to compare GC accross conditions

cFace = 2; cPlace = 3; cBaseline = 4;
comparisons = {[cFace cBaseline], [cPlace, cBaseline], [cFace cPlace]};
ncomp = length(comparisons);
p = cell(ncomp,1); h = cell(ncomp,1); stats=cell(ncomp,1);
for c = 1:ncomp  
    for i=1:n
        for j=1:n
            c1 = comparisons{c}(1,1);
            c2 = comparisons{c}(1,2);
            Fc1 = squeeze(F{c1}(i,j,:));
            Fc2 = squeeze(F{c2}(i,j,:));
            [p{c}(i,j), h{c}(i,j), stats{c}(i,j)] = ranksum(Fc1, Fc2, 'tail', 'right');
            
        end
    end
end

%% 

cFace = 2; cPlace = 3; cBaseline = 4;
comparisons = {[cFace cBaseline], [cPlace, cBaseline], [cFace cPlace]};
ncomp = length(comparisons);
z = cell(ncomp,1); pval = cell(ncomp,1); sig = cell(ncomp,1); 
pcrit = cell(ncomp,1); zcrit = cell(ncomp,1);
for c = 1:ncomp  
    for i=1:n
        for j=1:n
            c1 = comparisons{c}(1,1);
            c2 = comparisons{c}(1,2);
            Fc1 = squeeze(F{c1}(i,j,:));
            Fc2 = squeeze(F{c2}(i,j,:));
            z{c}(i,j) = mann_whitney(Fc1, Fc2);
            pval{c}(i,j) = 2*(1-normcdf(abs(z{c}(i,j)))); 
            [sig{c}, pcrit{c}] = significance(pval{c},alpha,mhtc,[]);
            zcrit{c} = -sqrt(2)*erfcinv(2*(1-pcrit{c}));
        end
    end
end

% Big problem. Need to run simulation.
