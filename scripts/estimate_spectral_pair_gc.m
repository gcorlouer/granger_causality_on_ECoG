% Multitrial spectral mvgc analysis
%% Input parameters
input_parameters;
conditions = {'Rest', 'Face', 'Place'};
ncdt = length(conditions);
nsub = length(cohort);
dataset = struct;
%% Loop multitrial functional connectivity analysis over each subjects
for s=1:nsub
     subject = cohort{s};
     % Loop over conditions
     for c=1:ncdt
        % Read condition specific time series
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',conditions{c}, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        % Functional visual channels indices
        indices = gc_input.indices;
        % Spectral pgc
        pGC = ts_to_spgc(X, 'morder',morder, 'regmode', regmode, ...
                    'dim',dim,'band',band ,'sfreq',sfreq,'nfreqs',nfreqs,...
                    'tstat',tstat,'mhtc', mhtc, 'alpha', alpha,...
                    'conditional',conditional);
        % Save dataset
        dataset(c,s).subject = subject;
        dataset(c,s).condition = conditions{c};
        dataset(c,s).pGC = pGC;
        dataset(c,s).indices = indices;
     end
end

%% Save dataset for plotting in python

fname = 'multi_trial_sfc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')

%% Plot pairwise spectral GC

