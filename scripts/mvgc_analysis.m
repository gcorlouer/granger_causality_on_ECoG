input_parameters;
cohort = {'AnRa',  'ArLa', 'DiAs'};
condition = {'Rest', 'Face', 'Place', 'baseline'};
field = {'time',  'condition', 'pair', 'subject','F'};
ncdt = length(condition);
nsub = length(cohort);
dataset = struct;
%% Load data

datadir = fullfile('~', 'projects', 'cifar', 'results');

for s = 1:nsub
    subject = cohort{s};
    fname = [subject '_condition_visual_ts.mat'];
    fpath = fullfile(datadir, fname);
    
    % Meta data about time series
    time_series = load(fpath);
    time = time_series.time; fs = double(time_series.sfreq);

    % Functional group indices
    indices = time_series.indices; fn = fieldnames(indices);
    
    for c=1:ncdt
        % Read conditions specific time series
        X = time_series.(condition{c});
        [n, m, N] = size(X);


        %% Detrend

        [X,~,~,~] = mvdetrend(X,pdeg,[]);
        
        % VAR estimation
        VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
        V = VAR.V;
        A = VAR.A;
        disp(VAR.info)
        
        % MVGC multitrial stat estimation
        [F, sigF, pcrit] = ts_to_mvgc_stat(X, 'gind', indices, 'morder',morder,...
        'regmode',regmode,'tstat', tstat,'alpha', alpha, 'mhtc',mhtc);
        % Single trial MVGC
        singleF = ts_to_single_mvgvc(X, 'gind', indices, 'morder',morder,...
        'regmode',regmode);
        %% Build dataset
        dataset(c,s).subject = subject;
        dataset(c,s).condition = condition{c};
        dataset(c,s).F = F;
        dataset(c,s).sigF = sigF;
        dataset(c,s).pcrit = pcrit;
        dataset(c,s).singleF = singleF;
    end
end
%% Save dataset for plotting in python

fname = 'mvgc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        