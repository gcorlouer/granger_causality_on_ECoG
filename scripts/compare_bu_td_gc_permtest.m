%% Comparing GC between direction
% This script computes the difference between top down and bottom up GC
% using permutation testing. 
% For each subjects and conditions, take condition specific time series X
% For each pairs of channels (i,j) running in the list RF of ret and face
% indices compute observed stat pairwise GC between channels X(i,:) and X(j,:)
% Concatenate X(i,:) and X(j,:) into a single time series Xc 
% Randomly allocate trials to ret and face channels without replacement
% Compute pairwise BU and TD GC between Xi and Xj and return test statistic Ti
% Repeat 1000 times, compute pvalues, signixficance and zscore for each pair
% Plot heatmap with ret chan and F chan showing BU vs TD for each sub and
% cdt (plot in python). 
%% Input parameters

input_parameters;
nsub = length(cohort);
ncdt = length(conditions);
%% Compare top down and bottom up band-spacific GC with permutation testing
% For each Subjects and Conditions
for s=1:nsub
    subject = cohort{s};
    for c=1:ncdt
        condition = conditions{c};
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        sfreq = gc_input.sfreq;
        [n,m,N] = size(X);
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
end

%% Save dataset
fname = 'compare_TD_BU_GC.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'GC')


