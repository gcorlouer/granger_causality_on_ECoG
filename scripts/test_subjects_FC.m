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
I = cell(nsub,1);
%% For each subjects compute GC and MI single trial distribution for all conditions

for s=1:nsub
    subject_id = cohort{s};
    for c=1:ncdt
        condition = conditions{c};
        % Compute MI and F
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        indices = gc_input.indices;
        I{s} = ts_to_single_mvmi(X, 'gind', indices);
        F{s} = ts_to_single_mvgvc(X, 'gind', indices, 'morder',morder,...
                'regmode',regmode);
        % Store single trial results in subject structure
        Subject.(subject_id).(condition).('single_MI') = I{s};
        Subject.(subject_id).(condition).('single_GC') = F{s};
    end
    Subject.(subject_id).indices = indices;
end
%%
% Loop conditons to build cross subject single FC
CrossSubject.indices = indices;
for c=1:ncdt
    % Cross subject
    condition = conditions{c};
    for s=1:nsub 
        % Compute MI and F
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        indices = gc_input.indices;
        I{s} = ts_to_single_mvmi(X, 'gind', indices);
        F{s} = ts_to_single_mvgvc(X, 'gind', indices, 'morder',morder,...
                'regmode',regmode);
    end
    CrossSubject.(condition).('single_MI')= I;
    CrossSubject.(condition).('single_GC')= F;
end

%% For each subject compute Z scores with significance
group = false;
for s=1:nsub
    subject_id = cohort{s};
    for icomp=1:nComparisons
        % Get conditions that we compare
        comparison = comparisons{icomp};
        % Loop over MI and GC functional connectivity
        single_FC = {'single_MI', 'single_GC'};
        for j=1:2
        FC = single_FC{j};
        % Compute z score from comparing single MI and GC distrib in 
        % different conditions
        z = singleFC_to_z(Subject, 'group',group, 'subject_id',subject_id, ...
            'comparison', comparison, 'FC', FC);
        comparison_name = [comparison{1} comparison{2}];
        % Return p values
        pvals = erfc(abs(z)/sqrt(2));
        % Sidak correction over top down/bottom-up directions and nsubjects
        mhtc = 'Sidak';
        nopv = true;
        ndir = 2; % directions for which we whish to test hypotheses
        nhyp = ndir*nsub;
        % Return significance and pvalues
        pcrit = significance(nhyp,alpha,mhtc, nopv);
        % Return zcrit
        zcrit = sqrt(2)*erfcinv(pcrit);
        % Return significant array
        sig = z > zcrit;
        % Return statistics
        zstat.z = z;
        zstat.zcrit = zcrit;
        zstat.pcrit = pcrit;
        zstat.sig = sig;
        % Build subject structure for python plotting
        Subject.(subject_id).(comparison_name).(FC).zstat = zstat;
        end
    end
end

%% Compute Z score accross all subjects
group = true;
for icomp=1:nComparisons
        % Get conditions that we compare
        comparison = comparisons{icomp};
        % Loop over MI and GC functional connectivity
        single_FC = {'single_MI', 'single_GC'};
        for j=1:2
        FC = single_FC{j};
        % Compute z score from comparing single MI and GC distrib in 
        % different conditions
        z = singleFC_to_z(CrossSubject, 'group',group, 'subject_id',subject_id, ...
            'comparison', comparison, 'FC', FC);
        comparison_name = [comparison{1} comparison{2}];
        % Return p values
        pvals = erfc(abs(z)/sqrt(2));
        % Sidak correction over top down/bottom-up directions and nsubjects
        mhtc = 'Sidak';
        nopv = true;
        ndir = 2; % directions for which we whish to test hypotheses
        nhyp = ndir;
        % Return significance and pvalues
        pcrit = significance(nhyp,alpha,mhtc, nopv);
        % Return zcrit
        zcrit = sqrt(2)*erfcinv(pcrit);
        % Return significant array
        sig = z > zcrit;
        % Return statistics
        zstat.z = z;
        zstat.zcrit = zcrit;
        zstat.pcrit = pcrit;
        zstat.sig = sig;
        % Build subject structure for python plotting
        CrossSubject.(comparison_name).(FC).zstat = zstat;
        end
end

%% Save for python

fname = 'test_singleFC.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'Subject','CrossSubject');










