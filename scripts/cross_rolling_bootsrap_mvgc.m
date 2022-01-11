%% input_parameters;
cohort = {'AnRa',  'ArLa',  'BeFe',  'DiAs',  'JuRo'};
condition = {'rest', 'face', 'place'};
field = {'time',  'condition', 'pair', 'subject','F'};
dataset = struct;
datasetB = struct;
input_parameters;
nsample = 2;
shift = 80;
mw = 80;
nsub = length(cohort);
%% 
for s=1:nsub
    % Read data
    %datadir = fullfile('~', 'analysis','results');
    datadir = fullfile('~', 'projects','CIFAR', 'data','results');
    sub_id = cohort{s};
    fname = [sub_id '_condition_ts_visual.mat']; 
    fpath = fullfile(datadir, fname);
    time_series = load(fpath);
    
    % The time series
    X = time_series.data; sub_id = time_series.sub_id; time = time_series.time;
    fs = double(time_series.sfreq);
    
    % Functional group indices
    findices = time_series.functional_indices; fn = fieldnames(findices);
    [n, m, N, ncdt] = size(X);
    
    % Read baseline data
    Xb = time_series.baseline;
    timeb = time_series.timeb;
    [nb, mb, Nb, ncdtb] = size(Xb);

    Xb = time_series.baseline;
    timeb = time_series.timeb;
    [nb, mb, Nb, ncdtb] = size(Xb);

    %% Window sample to time
    %% Window sample to time

    nwin = floor((m - mw)/shift +1);
    win_size = mw/fs;
    time_offset = shift/fs;
    win_time = zeros(nwin,mw);
    for w=1:nwin
        o = (w-1)*shift; 
        win_time(w,:) = time(o+1:o+mw);
    end
     nwinb = floor((mb - mw)/shift +1);
    %% Estimate mvgc bootstrapp
    nv = length(fn);
    Fb = zeros(nv, nv, nsample, nwin, ncdt);
    Fbb = zeros(nv, nv, nsample, nwinb, ncdt);
    for c=1:ncdt
        Fb(:,:,:,:,c) = sliding_mvgc_bootstrap(X(:,:,:,c),'time', time, ...  
        'gind', findices,'mw',mw, 'morder', morder, 'regmode', regmode, ...
        'shift', shift, 'pdeg',pdeg, 'nsample', nsample);
    end
   
% Estimate baselibe mvgc
    for c=1:ncdt
            for b=1:nb
                Fbb(:,:,:,:,c) = sliding_mvgc_bootstrap(Xb(:,:,:,c),'time', time, ...  
        'gind', findices,'mw',mw, 'morder', morder, 'regmode', regmode, ...
        'shift', shift, 'pdeg',pdeg, 'nsample', nsample);
            end
            mFbb(:,:,:,c) = mean(Fbb(:,:,:,:,c),4);
    end
 % Compute Mann-Whitney test
   z = zeros(nv, nv, nwin, ncdt);
   pval = zeros(nv, nv, nwin, ncdt);

    for w=1:nwin
        for c=1:ncdt
            for i=1:nv
                for j=1:nv
                    z(i,j,w,c)    = mann_whitney(mFbb(i,j,:,c), Fb(i,j,:,w,c)); % z-score ~ N(0,1) under H0
                    % How to compute pvalue from z score?
                    pval(i,j,w,c) = 1 - 1/2*erfc(-abs(z(i,j,w,c))/sqrt(2)); % p-value (2-tailed test)        
                end
            end
        end
    end
    
    % Effect size and z score dataset
    for w=1:nwin
        for c=1:ncdt
            for i=1:nv
                for j=1:nv
                   dataset(w,i,j,c,s).time = win_time(w,mw);
                   dataset(w,i,j,c,s).pair =  [fn{j} '->' fn{i}];
                   dataset(w,i,j,c,s).condition = condition{c};
                   dataset(w,i,j,c,s).subject = sub_id;
                   dataset(w,i,j,c,s).z = z(i,j,w,c);
                   dataset(w,i,j,c,s).pval = pval(i,j,w,c);
                   dataset(w,i,j,c,s).Fbm = mean(Fb(i,j,:,w,c), 3);
                   dataset(w,i,j,c,s).Fbbm = mFbb(i,j,c);
                end
            end
        end
    end
    
    % Bootstrap dataset
    for k=1:nsample
        for w=1:nwin
            for c=1:ncdt
                for i=1:nv
                    for j=1:nv
                       datasetB(w,k,i,j,c,s).sample = k;
                       datasetB(w,k,i,j,c,s).time = win_time(w,mw);
                       datasetB(w,k,i,j,c,s).pair =  [fn{j} '->' fn{i}];
                       datasetB(w,k,i,j,c,s).condition = condition{c};
                       datasetB(w,k,i,j,c,s).subject = sub_id;
                       datasetB(w,k,i,j,c,s).Fb = Fb(i,j,k,w,c);
                    end
                end
            end
        end
    end
    
end
%% Write dataset
[sig, pcrit] = significance(pval,alpha,mhtc,false);
zcrit = -sqrt(2)*erfcinv(2*(1-pcrit)); % check that it is the right function
dataset(w,i,j,c,s).pcrit = pcrit;
dataset(w,i,j,c,s).zcrit = zcrit;
for s=1:nsub
    for w=1:nwin
                for c=1:ncdt
                    for i=1:nv
                        for j=1:nv
                            dataset(w,i,j,c,s).pcrit = pcrit;
                            dataset(w,i,j,c,s).zcrit = zcrit;
                        end
                    end
                end
    end
end            
%% Reshape dataset

lenData = numel(dataset);
dataset = reshape(dataset, lenData, 1);

lenDataB = numel(datasetB);
datasetB = reshape(datasetB, lenDataB, 1);

%% Save dataset

df = struct2table(dataset);
fname = 'cross_sliding_mvgc_bootstrapp_test.csv';
fpath = fullfile(datadir, fname);
writetable(df, fpath)

% Save bootstrap distribution
dfB = struct2table(datasetB);
fname = 'cross_sliding_mvgc_bootstrapp_distribution_test.csv';
fpath = fullfile(datadir, fname);
writetable(dfB, fpath)
