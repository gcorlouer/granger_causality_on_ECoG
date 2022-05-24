% Single trial connectivity
%% Input parameters
input_parameters;
ncdt = length(condition);
nsub = length(cohort);
I = cell(ncdt, nsub);
F = cell(ncdt, nsub);

%%
for s=1:nsub
     subject = cohort{s};
     % Loop over conditions
     for c=1:ncdt
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition{c}, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        [n , m, N] = size(X);
        % Functional visual channels indices
        indices = gc_input.indices;
        % Groupwise Mutual information single distribution
        I{c,s} = ts_to_single_mvmi(X, 'gind', indices);
        % Groupwise MVGC single distribution
        F{c,s} = ts_to_single_mvgvc(X, 'gind', indices, 'morder',morder,...
                'regmode',regmode);
     end
end
%% Return Mutual information and GC in R->F and F->R directions


%% Compute group Z-score
zI = zeros(ncdt,ncdt);
zF = zeros(ncdt,ncdt);
for i=1:ncdt
    for j=1:ncdt
        zI(i,j) = mann_whitney_group(I(i,:),I{j,:});
        zF(i,j) = mann_whitney_group(F{i,:},F{j,:});
    end
end
%% Save dataset for plotting in python

fname = 'single_trial_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')