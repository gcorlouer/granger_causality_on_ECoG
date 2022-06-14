%% Compare top down vs bottom up GC
% In this script we compare top down (F -> R) vs bottom-up (R -> F) GC 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  -Subject and condition-specific mutltirial HFA #X=(n x m x N)
% Output: -3x1 Z-score testing stochastic dominance of top down vs
%         bottom up GC in each condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

input_parameters;
ncdt = length(conditions);
nsub = length(cohort);
% Cross subjects top down and bottom up GC
F_td = cell(nsub,1);
F_bu = cell(nsub,1);

Zscore = struct;
pZscore = struct;
%% Estimate group td vs bu by pooling td and bu
for c=1:ncdt
    condition = conditions{c};
    for s=1:nsub
        subject_id = cohort{s};
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        indices = gc_input.indices;
        F = ts_to_single_pair_unconditional_gc(X, 'morder',morder, 'regmode', regmode);
        R_idx = indices.('R');
        F_idx = indices.('F');
        F_t = F(R_idx, F_idx,:);
        F_b = F(F_idx, R_idx,:);
        F_t = reshape(F_t, numel(F_t),1);
        F_b = reshape(F_b, numel(F_b),1);
        
        F_td{s} = F_t;
        F_bu{s} = F_b;
    end
    z = mann_whitney_group(F_td,F_bu);
    zcrit = sqrt(2)*erfcinv(alpha);
    % Compute pcrit, zcrit, significance.
    Zscore.(condition).val = z;
    Zscore.(condition).crit = zcrit;
end

%% Estimate pairwise top down vs bottom up

for s=1:nsub
    subject_id = cohort{s};
    for c=1:ncdt
        condition = conditions{c};
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        indices = gc_input.indices;
        F = ts_to_single_pair_unconditional_gc(X, 'morder',morder, 'regmode', regmode);
        R_idx = indices.('R');
        F_idx = indices.('F');
        F_t = F(R_idx, F_idx,:);
        F_b = F(F_idx, R_idx,:);
        % Loop over pairs of retinotopic and face channels
        nR = length(R_idx);
        nF = length(F_idx);
        z = zeros(nR,nF);
        pvals = zeros(nR,nF);
        for i=1:nR
            for j =1:nF
                % Compute test statistics using Wilcoxon signed rank
                
                [p,h,stats] = signrank(squeeze(F_t(i,j,:)),squeeze(F_b(j,i,:)));
                z(i,j) = stats.zval;
                pvals(i,j) = p;
            end
        end
        % Correct for multiple hypothesis
        [sigs,pcrit] = significance(pvals,alpha,mhtc);
        zcrit = sqrt(2)*erfcinv(pcrit);
        % Add to structure
        pZscore.(subject_id).(condition).zval = z;
        pZscore.(subject_id).(condition).zcrit = zcrit;
        
        pZscore.(subject_id).(condition).sig = sigs;
    end
end

%% Save for python analysis

fname = 'top_down_vs_bottom_up.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'pZscore')





