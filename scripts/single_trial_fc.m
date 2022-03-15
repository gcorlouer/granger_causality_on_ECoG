% Single trial connectivity
%% Input parameters
input_parameters;
ncdt = length(condition);
nsub = length(cohort);
dataset = struct;
%%
for s=1:nsub
     subject = cohort{s};
     % Loop over conditions
     for c=1:ncdt
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition{c}, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        [n , m, N] = size(X);
        % Pairwise MI single distribution
        pMI = ts_to_single_pMI(X, 'q', q, ...
                'mhtc', mhtc, 'alpha', alpha);
        % Pairwise GC single distribution
        pGC = ts_to_single_pGC(X,'morder', morder,...
                'regmode', regmode,'alpha', alpha,'mhtc', mhtc, 'tstat', tstat);
        
        % Groupwise Mutual information single distribution
        gMI = ts_to_single_mvmi(X, 'gind', indices);

        % Groupwise MVGC single distribution
        gGC = ts_to_single_mvgvc(X, 'gind', indices, 'morder',morder,...
                'regmode',regmode);
        % Save dataset
        dataset(c,s).subject = subject;
        dataset(c,s).condition = condition{c};
        dataset(c,s).pMI = pMI;
        dataset(c,s).pGC = pGC;
        dataset(c,s).gMI = gMI;
        dataset(c,s).gGC = gGC;
     end
end
%% Save dataset for plotting in python

fname = 'single_trial_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')