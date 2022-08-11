% Multitrial functional connectivity analysis
%% Input parameters
input_parameters;
ncdt = length(conditions);
nsub = length(cohort);
dataset = struct;
%% Loop multitrial functional connectivity analysis over each subjects
for s=1:nsub
     subject_id = cohort{s};
     % Loop over conditions
     for c=1:ncdt
        condition = conditions{c};
        % Read condition specific time series
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
            'condition',condition, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        % Detrend
        X = mvdetrend(X,[],[]);
        % Pairwise conditional MI 
        pMI = ts_to_MI(X, 'q', q, 'mhtc', mhtc, 'alpha', alpha);
        % Pairwise conditional GC
        pGC = ts_to_dual_pgc(X, 'morder',morder, 'regmode', regmode, ...
                    'tstat',tstat,'mhtc', mhtc, 'alpha', alpha);
        % Functional visual channels indices
        indices = gc_input.indices;
        % Groupwise MI
        gMI = ts_to_mvmi(X, 'gind', indices, ...
                    'alpha', alpha, 'mhtc',mhtc);
        % Groupwise GC
        gGC = ts_to_dual_mvgc(X, 'gind', indices, 'morder',morder,...
                'regmode',regmode,'tstat', tstat,'alpha', alpha, 'mhtc',mhtc);
        % Save dataset
        dataset(c,s).subject = subject_id;
        dataset(c,s).condition = condition;
        dataset(c,s).indices = indices;
        dataset(c,s).pMI = pMI;
        dataset(c,s).pGC = pGC;
        dataset(c,s).gMI = gMI;
        dataset(c,s).gGC = gGC;
     end
end

%% Save dataset for plotting in python

fname = 'multi_trial_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')