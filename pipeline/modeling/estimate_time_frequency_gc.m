%% Input parameters
input_parameters;
conditions = {'Rest','Face','Place'};
ncdt = length(conditions);
nsub = length(cohort);
GC = struct;
subject = 'DiAs';

%% Read condition specific time series
for c=1:ncdt
    condition = conditions{c};
    gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
        'condition', condition , 'suffix', suffix);
    X = gc_input.X;
    sfreq  = gc_input.sfreq;
    gind = gc_input.indices;
    time = gc_input.time;
    % Sizes of input
    gi = fieldnames(gind); 
    ng = length(gi); % Number of groups
    [n,m,N] = size(X);
    % Window parameters
    nwin = floor((m - mw)/shift +1);
    tw = mw/sfreq; % Duration of time window

    %% Estimate time frequency pair and group GC

    nfreqs = 1024;
    pF = zeros(n, n, nwin,nfreqs+1);
    gF = zeros(ng, ng, nwin,nfreqs+1);
    fprintf('Rolling time window of size %4.2f \n', tw)
    for w=1:nwin
        fprintf('Time window number %d over %d \n', w,nwin)
        o = (w-1)*shift;      % window offset
        W = X(:,o+1:o+mw,:,:); % the window
        % Compute ss model of window
        [ss.A,ss.C,ss.K,ss.V,ss.Z,ss.E] = tsdata_to_ss(W, pf, ssmo);
        % Compute pair and group GC
        gF(:,:,w,:) = ss_to_sgGC(ss, 'gind',gind, 'sfreq',sfreq, 'nfreqs', nfreqs);
        pF(:,:,w,:) = ss_to_spwcgc(ss.A,ss.C,ss.K,ss.V,nfreqs);
    end

    roll_time = zeros(nwin,1);
    for w=1:nwin
        o = (w-1)*shift; 
        roll_time(w) = time(o+mw);
    end
    freqs = sfreqs(nfreqs, sfreq);
    GC.(subject).(condition).pF = pF; 
    GC.(subject).(condition).gF = gF;
    GC.(subject).(condition).time = roll_time; 
end
GC.freqs = freqs;
%% Save dataset

fname = 'time_frequency_gc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'GC')
