%% Compare GC between conditions 
% In this script we compare pairwise or groupwise GC between conditions
% Comparisons are done for each subjects. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  -Subject and condition-specific mutltirial HFA #X=(n x m x N)
% Output: -Subject,comparison,direction-specific Z-score testing 
%           dominance of GC between conditions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input parameters
input_parameters;
connect = 'groupwise';
comparisons = {{'Face' 'Rest'}, {'Place' 'Rest'}, {'Face' 'Place'}};
compare = {'FR' 'PR' 'FP'};
% Condition, comparison sized
ncdt = length(conditions);
nsub = length(cohort);
nComp = size(comparisons,2);
% Prepare cell arrays
F = cell(2,1);
Xc = cell(2,1);
Ntrial = zeros(2,1);
% Number of permutations
Ns = 500;

%% Compare GC between conditions

for s=1:nsub
       subject = cohort{s};
    for c=1:nComp
            % Concatenate comparisons
            comparison = comparisons{c};
            comparison_name = [comparison{1}(1) 'vs' comparison{2}(1)];
            for i=1:2
                condition = comparison{i};
                gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                    'condition',condition, 'suffix', suffix);
                Xc{i} = gc_input.X;
                [n,m,Ntrial(i)] = size(Xc{i});
                indices = gc_input.indices;
                % Estimate SS model
                pf = 2 * morder;
                [model.A,model.C,model.K,model.V,~,~] = tsdata_to_ss(Xc{i},pf,ssmo);
                % Compute GC
                fn = fieldnames(indices);
                ng = length(fn);
                group = cell(ng,1);
                for k=1:length(fn)
                    group{k} = double(indices.(fn{k}));
                end
                group = group';
                F{i} = ss_to_GC(model, 'connect', connect ,'group', group,...
        'dim', dim, 'sfreq', sfreq, 'nfreqs', nfreqs, 'band',band);
            end
            % Observed statistic
            obsStat = F{1} - F{2};
            % Concatenate for permutation testing
            X = cat(3, Xc{1},Xc{2});
            % Compute permutation test statistic
            tstat = permute_condition_GC(X, 'connect', connect ,'group', group,...
                'Ntrial', Ntrial, 'Ns', Ns, 'morder', morder, 'ssmo',ssmo, ...
                'sfreq',sfreq, 'nfreqs', nfreqs,'dim',dim, 'band',band);
            % Permutation testing
            stat = permtest(tstat, 'obsStat', obsStat, 'alpha', alpha, 'mhtc','FDRD');
            % Save into GC structure for python plotting
            GC.(subject).(comparison_name).('z') = stat.z;
            GC.(subject).(comparison_name).('sig') = stat.sig;
            GC.(subject).(comparison_name).('pval') = stat.pval;
            GC.(subject).(comparison_name).('zcrit') = stat.zcrit;
            GC.(subject).indices = indices;
            GC.connect = connect;
    end
end
%% Save dataset for plotting in python

fname = 'compare_condition_GC.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'GC')
