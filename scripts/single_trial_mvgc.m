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
%% Compute individual subject Z scores from GC.

comparisons = [1 2; 1 3; 2 3];
nComparisons = size(comparisons,1);
ncomp = length(comparisons);
GC = struct;
for s=1:nsub
    subject = cohort{s};
    gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition{2}, 'suffix', suffix);
    % Get indices of visually responsive populations to account for 
    % different indices in different subjects
    indices = gc_input.indices;
    populations = fieldnames(indices);
    ng = length(populations);
    z = zeros(ng,ng);
    % Loop over comparisons
    for c=1:nComparisons
        i1 = comparisons(c,1);
        i2 = comparisons(c,2);
        F1 = F{i1,s};
        F2 = F{i2,s};
        % Test wether F1>F2
        for i=1:ng
            for j=1:ng
                z(i,j) = mann_whitney(F2(i,j,:), F1(i,j,:));
            end
        end
        % Multiple hypothesis correction
        mhtc = 'Sidak'; % Other methods like FDR or FDRD are too conservative
        pvals = erfc(abs(z)/sqrt(2));
        % Number of hypotheses
        nhyp = numel(pvals)*nsub;
        % No pvalues
        nopv = true;
        % Pcrit and Zcrit
        pcrit = significance(nhyp,alpha,mhtc,nopv);
        zcrit = sqrt(2)*erfcinv(pcrit);
        % Prepare dataset for plotting in python
        GC(s,c).subject = subject;
        % Condition i1 > condition i2
        GC(s,c).pair = {condition{i1}, condition{i2}};
        GC(s,c).populations = populations;
        GC(s,c).z = z;
        % Z critique for threshold p=0.05
        GC(s,c).zcrit = zcrit;
    end
end

%% Compute group Z score from GC comparisons

gGC = struct;
for c=1:ncomp
    i1 = comparisons(c,1); 
    i2 = comparisons(c,2);
    F1 = cell(nsub,1);
    F2 = cell(nsub,1);
    z = zeros(ng,ng);
    for i=1:ng
        for j=1:ng
            for s = 1:nsub
                subject = cohort{s};
                gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                        'condition',condition{2}, 'suffix', suffix);
                % Get indices of visually responsive populations
                indices = gc_input.indices;
                populations = fieldnames(indices);
                F1{s} = squeeze(F{i1,s}(i,j,:));
                F2{s} = squeeze(F{i2,s}(i,j,:));
            end
            z(i,j) = mann_whitney_group(F2,F1);
        end
    end
   % Multiple hypothesis correction
    mhtc = 'Sidak'; % Other methods like FDR or FDRD are too conservative
    pvals = erfc(abs(z)/sqrt(2));
    % Number of hypotheses
    nhyp = numel(pvals);
    % No pvalues
    nopv = true;
    % Pcrit and Zcrit
    pcrit = significance(nhyp,alpha,mhtc,nopv);
    zcrit = sqrt(2)*erfcinv(pcrit);
    % Prepare dataset for plotting in python
    % Condition i1 > condition i2
    gGC(c).pair = {condition{i1}, condition{i2}};
    gGC(c).populations = populations;
    gGC(c).z = z;
    % Z critique for threshold p=0.05
    gGC(c).zcrit = zcrit;
end

%% Save dataset for plotting in python

fname = 'compare_condition_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'GC','gGC')