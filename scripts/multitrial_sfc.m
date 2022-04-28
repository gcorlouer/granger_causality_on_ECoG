% Multitrial functional connectivity analysis
%% Input parameters
input_parameters;
ncdt = length(condition);
nsub = length(cohort);
dataset = struct;
%% Loop multitrial functional connectivity analysis over each subjects
for s=1:nsub
     subject = cohort{s};
     % Loop over conditions
     for c=1:ncdt
        % Read condition specific time series
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition{c}, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        pGC = ts_to_spgc(X, 'morder',morder, 'regmode', regmode, ...
                    'dim',dim,'band',band ,'sfreq',sfreq,'fres',fres,...
                    'tstat',tstat,'mhtc', mhtc, 'alpha', alpha);
        % Save dataset
        dataset(c,s).subject = subject;
        dataset(c,s).condition = condition{c};
        %dataset(c,s).pMI = pMI;
        dataset(c,s).pGC = pGC;
        %dataset(c,s).gMI = gMI;
        %dataset(c,s).gGC = gGC;
     end
end

%% Save dataset for plotting in python

fname = 'multi_trial_sfc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')
