% Rolling multitrial pairwise and groupwise gc against the null. 
%% Input parameters
input_parameters;
ncdt = length(condition);
nsub = length(cohort);
dataset = struct;

%% Rolling window
% loop over subjects
for s=1:nsub
    subject = cohort{s};
    % Loop over conditions
    for c=1:ncdt
        % Read condition specific time series
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition{c}, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X; time = gc_input.time; 
        [n, m, N] = size(X);
        % Window sample to time
        fs = gc_input.sfreq;
        nwin = floor((m - mw)/shift +1);
        win_size = mw/fs;
        time_offset = shift/fs;
        win_time = zeros(nwin,mw);
        % Functional visual channels indices
        indices = gc_input.indices;
        ng = length(fieldnames(indices));
        % Initialise pairwise and grouwise FC arrays 
        % Pairwise MI
        pmi = zeros(n,n,nwin); sig_pmi = zeros(n,n,nwin); 
        pval_pmi = zeros(n,n,nwin); pcrit_pmi = zeros(nwin,1);
        % Pairwise GC
        pgc = zeros(n,n,nwin); sig_pgc = zeros(n,n,nwin); 
        pval_pgc = zeros(n,n,nwin); pcrit_pgc = zeros(nwin,1);
        % Groupwise MI
        gmi = zeros(ng,ng,nwin); sig_gmi = zeros(ng,ng,nwin); 
        pval_gmi = zeros(ng,ng,nwin); pcrit_gmi = zeros(nwin,1);
        % Groupwise GC
        ggc = zeros(ng,ng,nwin); sig_ggc = zeros(ng,ng,nwin); 
        pval_ggc = zeros(ng,ng,nwin); pcrit_ggc = zeros(nwin,1);
        % Loop over window
        for w=1:nwin
            o = (w-1)*shift; 
            win_time(w,:) = time(o+1:o+mw);
        end
        for w=1:nwin
            o = (w-1)*shift; 
            W = X(:,o+1:o+mw,:);% the window     
            % Compute pairwise MI       
            win_pMI  = ts_to_MI(W, 'q', q, ...
                    'mhtc', mhtc, 'alpha', alpha);
            % Pairwise GC, dual regression
            win_pGC = ts_to_dual_pgc(W, 'morder',morder, 'regmode', regmode, ...
                'tstat',tstat,'mhtc', mhtc, 'alpha', alpha);
            % Groupwise MI
            win_gMI = ts_to_mvmi(W, 'gind', indices, ...
                        'alpha', alpha, 'mhtc',mhtc);
            % Groupwise GC
            win_gGC = ts_to_dual_mvgc(W, 'gind', indices, 'morder',morder,...
                    'regmode',regmode,'tstat', tstat,'alpha', alpha, 'mhtc',mhtc);
            % Compute FC arrays
            % pMI
            pmi(:,:,w) = win_pMI.f; sig_pmi(:,:,w) = win_pMI.sig; 
            pval_pmi(:,:,w)= win_pMI.pval; pcrit_pmi(w) = win_pMI.pcrit;
            % pGC
            pgc(:,:,w) = win_pGC.f; sig_pgc(:,:,w) = win_pGC.sig; 
            pval_pgc(:,:,w)= win_pGC.pval; pcrit_pgc(w) = win_pGC.pcrit;
            % gMI
            gmi(:,:,w) = win_gMI.f; sig_gmi(:,:,w) = win_gMI.sig; 
            pval_gmi(:,:,w)= win_gMI.pval; pcrit_gmi(w) = win_gMI.pcrit;
            % gGC
            ggc(:,:,w) = win_gGC.f; sig_ggc(:,:,w) = win_gGC.sig; 
            pval_ggc(:,:,w)= win_gGC.pval; pcrit_ggc(w) = win_gGC.pcrit;
        end
        % Save FC in structure
        pMI.f = pmi; pMI.sig = sig_pmi; pMI.pval = pval_pmi; pMI.pcrit = pcrit_pmi;
        pGC.f = pgc; pGC.sig = sig_pgc; pGC.pval = pval_pgc; pGC.pcrit = pcrit_pgc;
        gMI.f = gmi; gMI.sig = sig_gmi; gMI.pval = pval_gmi; gMI.pcrit = pcrit_gmi;
        gGC.f = ggc; gGC.sig = sig_ggc; gGC.pval = pval_ggc; gGC.pcrit = pcrit_ggc;
        % Save dataset
        dataset(c,s).subject = subject;
        dataset(c,s).condition = condition{c};
        dataset(c,s).time = win_time(:,mw);
        dataset(c,s).indices = indices;
        dataset(c,s).pMI = pMI;
        dataset(c,s).pGC = pGC;
        dataset(c,s).gMI = gMI;
        dataset(c,s).gGC = gGC;
    end
end

%% Save dataset for plotting in python

fname = 'rolling_multi_trial_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')


















