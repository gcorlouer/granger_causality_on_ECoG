input_parameters
subject = 'DiAs';
conditions = {'Rest','Face', 'Place'};
ncdt = length(conditions);
suffix = ['_condition_two_chans_' signal '.mat'];
connect = 'pairwise';
morder = 10;  % suggested model order
nfreq = 1024;
ssmo = 39; % suggested state space model order
dim = 3;
GC = struct;

bandstr = mat2str(band);
fname = ['two_chans_TD_BU_GC_' bandstr 'Hz.mat'];
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
        GC.(condition).('z') = stat.z;
        GC.(condition).('sig') = stat.sig;
        GC.(condition).('pval') = stat.pval;
        GC.(condition).('zcrit') = stat.zcrit;
        GC.('band') = band;
        GC.indices = indices;
end
fpath = fullfile(datadir, fname);
save(fpath, 'GC')