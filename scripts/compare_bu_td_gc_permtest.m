%% Comparing GC between direction
% Scripts does takes about 17mnto run with 100 permutations
%% Input parameters
tic;
input_parameters;
nsub = length(cohort);
ncdt = length(conditions);
%% Compare top down and bottom up band-spacific GC with permutation testing
% For each Subjects and Conditions
for s=1:nsub
    subject = cohort{s};
    fprintf('Compare TD vs BU for subject %s \n', subject)
    for c=1:ncdt
        condition = conditions{c};
        fprintf('Condition %s for subject %s\n', condition, subject)
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        sfreq = gc_input.sfreq;
        [n,m,N] = size(X);
        if strcmp(condition, 'Rest')
            trial_idx = 1:N;
            Ntrial = 56; % Take same number of trials as Face (faster computation)
            trials = datasample(trial_idx, Ntrial,'Replace',false);
            X = X(:,:, trials);
        end
        indices = gc_input.indices;
        stat = compare_TD_BU_pgc(X, indices, 'morder', morder, 'ssmo', ssmo,...
            'Ns',Ns,'alpha',alpha, 'mhtc',mhtc, ...
            'sfreq',sfreq, 'nfreqs', nfreqs,'dim',dim, 'band',band);
        GC.(subject).(condition).('z') = stat.z;
        GC.(subject).(condition).('sig') = stat.sig;
        GC.(subject).(condition).('pval') = stat.pval;
        GC.(subject).(condition).('zcrit') = stat.zcrit;
        GC.('band') = band;
        GC.(subject).indices = indices;
    end
    fprintf('\n')
end
toc; 
%% Save dataset
bandstr = mat2str(band);
fname = ['compare_TD_BU_GC_' bandstr 'Hz.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'GC')

