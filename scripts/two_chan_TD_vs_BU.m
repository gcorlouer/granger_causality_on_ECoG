bands = struct;
bands.alpha = [8 12];
bands.beta = [13 30];
bands.gamma = [32 60];
bands.hgamma = [60 120];
band_names = fieldnames(bands);
nband = length(band_names);

input_parameters
subject = 'DiAs';
conditions = {'Face', 'Place'};
ncdt = length(conditions);
signal = 'hfa';
suffix = '_condition_pick_chans_hfa';
connect = 'pairwise';
morder = 10;  % suggested model order
nfreq = 1024;
ssmo = 39; % suggested state space model order
dim = 3;
GC = struct;

for ib=1:nband
    band = bands.(band_names{ib});
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
end