%% Compare condition specific spectral GC
% In this script we compare spectral GC during condition 1 and condition 2
% Comparison are done for each subjects and directions. In addition to an
% indivisual subject analysis, we also pool z score across subjects. 
% We consider all pairs of condition among Rest, Face and Place
% presentation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  -Subject and condition-specific mutltirial HFA #X=(n x m x N)
% Output: -Subject,comparison,direction-specific Z-score testing 
%           stochastic dominance of GC in a pair of conditions
%        - comparison,direction-specific Z-scores pooled across subjects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input parameters
input_parameters;
ncdt = length(conditions);
nsub = length(cohort);
comparisons = {{'Face' 'Rest'}, {'Place' 'Rest'}, {'Face' 'Place'}};
nComparisons = size(comparisons,2);
Subject = struct;
CrossSubject = struct;
% Prepare cell array of F and I to compute group z score
F = cell(nsub,1);

%%

for s=1:nsub
    subject_id = cohort{s};
    for c=1:ncdt
        condition = conditions{c};
        % Compute MI and F
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        indices = gc_input.indices;
        [n, m, N] = size(X);
        f = zeros(n,n,nfreqs+1,N);
        F = zeros(n,n,N);
        F{s} = ts_to_single_spgc(X,'morder',morder,'band', band, ...
               'regmode',regmode, 'sfreq', sfreq);
        % Store single trial results in subject structure
        Subject.(subject_id).(condition).('single_GC') = F{s};
    end
    Subject.(subject_id).indices = indices;
end

%%