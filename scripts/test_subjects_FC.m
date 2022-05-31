%% Input parameters
input_parameters;
ncdt = length(conditions);
nsub = length(cohort);
comparisons = {{'Face' 'Rest'}, {'Place' 'Rest'}, {'Face' 'Place'}};
nComparisons = size(comparisons,2);
Subject = struct;
%% For each subjects compute GC and MI single trial distribution for all conditions

for s=1:nsub
    subject_id = cohort{s};
    for c=1:ncdt
        condition = conditions{c};
        % Compute MI and F
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        indices = gc_input.indices;
        I = ts_to_single_mvmi(X, 'gind', indices);
        F = ts_to_single_mvgvc(X, 'gind', indices, 'morder',morder,...
                'regmode',regmode);
        % Store single trial resu;ts in subject structure
        Subject.(subject_id).(condition).('single_MI') = I;
        Subject.(subject_id).(condition).('single_GC') = F;
    end
    Subject.(subject_id).indices = indices;
end

%% For each subject compute Z scores with significance

for s=1:nsub
    subject_id = cohort{s};
    for icomp=1:nComparisons
        comparison = comparisons{icomp};
        single_FC = {'single_MI', 'single_GC'};
        for j=1:2
        FC = single_FC{j};
        tstat = tstat_singleFC(Subject, 'subject_id',subject_id, ...
            'comparison', comparison, 'FC', FC, 'alpha',alpha, 'mhtc', mhtc);
        comparison_name = [comparison{1} comparison{2}];
        Subject.(subject_id).(comparison_name).(FC) = tstat;
        end
    end
end
%% Save for python

fname = 'test_singleFC.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'Subject');










