%% Compare condition specific group MI
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
I = cell(2,1);
Xc = cell(2,1);
N = zeros(2,1);
dataset = struct;
% Number of permutations
Ns = 500;
%% Compare pairwise MI between conditions

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
                % Estimate autocov matrix
                debias = [];
                q = 0;
                G = tsdata_to_autocov(Xc{i},q,debias);
                % Compute pairwise MI 
                I{i} = cov_to_gMI(G, 'gind', indices);
            end
            observedI = I{1} - I{2};
            X = cat(3, Xc{1},Xc{2});
            statI = compare_condition_gMI(X, 'obsStat', observedI, 'N', N, ...
                'Ns', Ns, 'gind', indices);
            gMI.(subject).(comparison_name).('z') = statI.z;
            gMI.(subject).(comparison_name).('sig') = statI.sig;
            gMI.(subject).(comparison_name).('pval') = statI.pval;
            gMI.(subject).(comparison_name).('zcrit') = statI.zcrit;
            gMI.(subject).indices = indices;
    end
end
%% Save dataset for plotting in python

fname = 'compare_condition_gMI.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'gMI')
