% Multitrial functional connectivity analysis

%% Input parameters
input_parameters;
ncdt = length(condition);
nsub = length(cohort);
dataset = struct;
c =2;
%% Read input time series and metadata

% Read condition specific time series
gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
    'suffix', suffix);
% Functional group indices
indices = gc_input.indices; fn = fieldnames(gc_input);
% Read conditions specific time series
X = gc_input.(condition{c});
[n, m, N] = size(X);
% Detrend
[X,~,~,~] = mvdetrend(X,pdeg,[]);
% Pairwise conditional MI estimation
pMI = ts_to_MI(X, 'q', q, 'mhtc', mhtc, 'alpha', alpha);
% Pairwise conditional GC
pGC = ts_to_dual_pgc(X, 'morder',morder, 'regmode', regmode, ...
            'tstat',tstat,'mhtc', mhtc, 'alpha', alpha);       
% Groupwise MI
gMI = ts_to_mvmi(X, 'gind', indices, ...
            'alpha', alpha, 'mhtc',mhtc);
% Groupwise GC
gGC = ts_to_dual_mvgc(X, 'gind', indices, 'morder',morder,...
        'regmode',regmode,'tstat', tstat,'alpha', alpha, 'mhtc',mhtc);