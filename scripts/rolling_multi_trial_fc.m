% Rolling multitrial pairwise and groupwise gc against the null. 
%% Input parameters
input_parameters;
ncdt = length(condition);
nsub = length(cohort);
dataset = struct;

%% Read input time series and metadata

% Read condition specific time series
gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
    'suffix', suffix);
% Functional group indices
indices = gc_input.indices; fn = fieldnames(gc_input);
% Read conditions specific time series
X = gc_input.(condition{c});
[n, m, N] = size(X);

%% Detrend
[X,~,~,~] = mvdetrend(X,pdeg,[]);

%% Rolling window

for w = 1:nwin
    o = (w-1)*nsobs;      % window offset
	W = X(:,o+1:o+nwobs,:);% the window
    for c=1:ncdt
        % Compute pairwise MI
        pMI  = ts_to_MI(W, 'q', q, ...
                'mhtc', mhtc, 'alpha', alpha);
        % Pairwise GC, dual regression
        pF = ts_to_dual_pgc(X, 'morder',morder, 'regmode', regmode, ...
            'tstat',tstat,'mhtc', mhtc, 'alpha', alpha);
        % Goup MI
        % Group GC
        [gF, sigF, pvalF] = ts_to_mvgc_stat(W, 'gind', indices, 'morder',morder,...
        'regmode',regmode,'tstat', tstat,'alpha', alpha, 'mhtc',mhtc);
        % Compute groupwise GC
    end
end

%%




















