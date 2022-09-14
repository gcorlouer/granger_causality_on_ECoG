%% Compare condition specific GC
% In this script we compare GC and MI during condition 1 and condition 2
% Comparison are done for each subjects and directions. In addition to an
% indivisual subject analysis, we also pool z score across subjects. 
% We consider all pairs of condition among Rest, Face and Place
% presentation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  -Subject and condition-specific mutltirial HFA #X=(n x m x N)
% Output: -Subject,comparison,direction-specific Z-score testing 
%           stochastic dominance of GC in a pair of conditions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input parameters
input_parameters;
ncdt = length(conditions);
nsub = length(cohort);
comparisons = {{'Face' 'Rest'}, {'Place' 'Rest'}, {'Face' 'Place'}};
compare = {'FR' 'PR' 'FP'};
nComp = size(comparisons,2);
% Prepare cell arrays  I and Xc to compute z score
F = cell(2,1);
Xc = cell(2,1);
N = zeros(2,1);
dataset = struct;
% Number of permutations
Ns = 100;

%% Compare groupwise GC between conditions

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
                [n,m,N(i)] = size(Xc{i});
                indices = gc_input.indices;
                % Estimate SS model
                [A,C,K,V,~,~] = tsdata_to_ss(Xc{i},pf,ssmo);
                ss.A = A; ss.C = C; ss.K = K; ss.V =V;
                % Compute observed group GC 
                F{i} = ss_to_gGC(ss, 'gind', indices);
            end
            observedF = F{1} - F{2};
            X = cat(3, Xc{1},Xc{2});
            stat = compare_condition_gGC(X, 'obsStat', observedF, 'N', N, ...
                'Ns', Ns, 'morder', morder, 'ssmo',ssmo, 'gind', indices);
            gGC.(subject).(comparison_name).('z') = stat.z;
            gGC.(subject).(comparison_name).('sig') = stat.sig;
            gGC.(subject).(comparison_name).('pval') = stat.pval;
            gGC.(subject).(comparison_name).('zcrit') = stat.zcrit;
            gGC.(subject).indices = indices;
    end
end
%% Save dataset for plotting in python

fname = 'compare_condition_gGC.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'gGC')
