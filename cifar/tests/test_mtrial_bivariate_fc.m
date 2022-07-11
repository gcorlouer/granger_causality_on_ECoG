% Multitrial functional connectivity analysis
%% Input parameters
input_parameters;
ncdt = length(condition);
nsub = length(cohort);
dataset = struct;
subject = 'DiAs';
suffix = '_condition_bivariate_ts.mat';
%% Loop multitrial functional connectivity analysis over each subjects

 % Loop over conditions
for c=1:ncdt
    % Read condition specific time series
    gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
        'condition',condition{c}, 'suffix', suffix);
    % Read conditions specific time series
    X = gc_input.X;
    % Pairwise conditional MI 
    pMI = ts_to_MI(X, 'q', q, 'mhtc', mhtc, 'alpha', alpha);
    % Pairwise conditional GC
    pGC = ts_to_dual_pgc(X, 'morder',morder, 'regmode', regmode, ...
                'tstat',tstat,'mhtc', mhtc, 'alpha', alpha);
    % Save dataset
    dataset(c).subject = subject;
    dataset(c).condition = condition{c};
    dataset(c).pMI = pMI;
    dataset(c).pGC = pGC;
end

%% Save dataset for plotting in python

fname = 'multi_trial_bivariate_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')