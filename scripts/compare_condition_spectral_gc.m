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
% Prepare cell array of F across subjects
pF = cell(nsub,1);
gF = cell(nsub,1);

% EEG bands
bands = struct;
bands.delta = [1 4];
bands.theta = [4 7];
bands.alpha = [8 12];
bands.beta = [13 30];
bands.gamma = [32 60];
bands.hgamma = [60 120];
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
        pF{s} = ts_to_single_spgc(X,'morder',morder,'band', band, ...
               'regmode',regmode, 'sfreq', sfreq);
        gF{s} = ts_to_single_smvgc(X,'gind',indices,'morder',morder,'band', band, ...
               'regmode',regmode, 'sfreq', sfreq);
        % Store single trial results in subject structure
        Subject.(subject_id).(condition).('pair_GC') = pF{s};
        
    end
    Subject.(subject_id).indices = indices;
end

%%