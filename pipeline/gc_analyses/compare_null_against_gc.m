% Multitrial functional connectivity analysis
%% Input parameters
input_parameters;
ncdt = length(conditions);
nsub = length(cohort);
dataset = struct;
%% Loop multitrial functional connectivity analysis over each subjects
for s=1:nsub
     subject = cohort{s};
     % Loop over conditions
     for c=1:ncdt
        condition = conditions{c};
        % Read condition specific time series
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        % Detrend
        % X = mvdetrend(X,pdeg,[]);
        % Functional group indices
        indices = gc_input.indices;
        fn = fieldnames(indices);
        ng = length(fn);
        group = cell(ng,1);
        for k=1:length(fn)
            group{k} = double(indices.(fn{k}));
        end
        % Estimate covariance matrix
        [n,m,N] = size(X);
        Nsample = m * N;
        V = tsdata_to_autocov(X,q);
        % Compute mutual information
        MI = cov_to_MI(V, 'connect', connect,'group', group ,'Nsample', Nsample, ...
           'alpha',alpha, 'mhtc', mhtc);
        % Estimate VAR model
        VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
        % Compute GC
        GC = var_to_dualGC(X, VAR, 'connect', connect,'group', group , 'morder',morder,...
                'regmode',regmode,'test', test);
        % Save Functional connectivity dataset
        FC.(subject).(condition).('GC')= GC;
        FC.(subject).(condition).('MI') = MI;
     end
     FC.(subject).indices = indices;
end
FC.('connectivity') = connect;
%% Save dataset for plotting in python

fname = ['null_' connect '_fc.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'FC')