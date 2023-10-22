% Multitrial spectral mvgc analysis
%% Input parameters
input_parameters;
signal = 'hfa';
suffix = ['_condition_visual_' signal '.mat'];
ncdt = length(conditions);
nsub = length(cohort);
dataset = struct;
if strcmp(signal,'hfa')
    morder = 5;
    ssmo = 15;
else
    morder = 5;
    ssmo = 20;
end
%% Loop multitrial functional connectivity analysis over each subjects
for s=1:nsub
     subject = cohort{s};
     % Loop over conditions
     for c=1:ncdt
        condition = conditions{c};
        % Read condition specific time series
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',conditions{c}, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        indices = gc_input.indices;
        [n,m,N] = size(X);
        % Take same number of trials as Face (faster computation)
        if strcmp(condition, 'Rest')
            trial_idx = 1:N;
            Nt = 56;
            trials = datasample(trial_idx, Nt,'Replace',false);
            X = X(:,:, trials);
        end
        % Frequencies
        sfreq = gc_input.sfreq;
        freqs = sfreqs(nfreqs,sfreq);
        % Estimate SS model
        pf = 2 * morder;
        [model.A,model.C,model.K,model.V,~,~] = tsdata_to_ss(X,pf,ssmo);
        % Compute group spectral GC
        fn = fieldnames(indices);
        ng = length(fn);
        group = cell(ng,1);
        for k=1:length(fn)
            group{k} = double(indices.(fn{k}));
        end
        group = group';
        f = ss_to_sGC(model, 'connect', connect ,'group', group,'nfreqs', nfreqs);
        % Functional visual channels indices
        indices = gc_input.indices;
        sGC.(subject).(condition) = f;
        sGC.freqs = freqs;
        sGC.(subject).indices = indices;
     end
end

%% Save dataset for plotting in python

fname = ['spectral_group_GC_' signal '.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'sGC')
