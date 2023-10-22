% Multitrial spectral mvgc analysis
%% Input parameters
input_parameters;
suffix = ['_condition_visual_' signal '.mat'];
ncdt = length(conditions);
nsub = length(cohort);
dataset = struct;

if strcmp(signal,'hfa')
    morder = 5;
    ssmo = 15;
    band = [0 62]; % Band when downsampled to 125 Hz
else
    morder = 5;
    ssmo = 20;
end

bandstr = mat2str(band);
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
        % Compute band GC
        fn = fieldnames(indices);
        ng = length(fn);
        group = cell(ng,1);
        for k=1:length(fn)
            group{k} = double(indices.(fn{k}));
        end
        group = group';
        F = ss_to_GC(model, 'connect', connect ,'group', group, ..., 
            'nfreqs', nfreqs, 'band', band,'dim',dim);
        % Functional visual channels indices
        indices = gc_input.indices;
        GC.(subject).(condition) = F;
        GC.freqs = freqs;
        GC.band = band;
        GC.connect = connect;
        GC.(subject).indices = indices;
     end
end

%% Save dataset for plotting in python

fname = ['magnitude_', connect,'_',bandstr, '_GC.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'GC')
