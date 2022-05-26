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
%% Conpute per subjects Z-score
% Note just for testing on individual subjects 
% but we will only return group Z scores
% z = cell(nsub,1);
% for s=1:nsub
%     F_face = F{1,s};
%     F_place = F{2,s};
%     T = zeros(ng,ng);
%     for i=1:ng
%         for j =1:ng
%             T(i,j) = mann_whitney(F_place(i,j,:),F_face(i,j,:));
%         end
%     end
%     z{s} = T;
% end

%% Compute group Z-score
% Face and place corss subjects GC across conditions
F_face = cell(nsub,1);
F_place = cell(nsub,1);
% Mutual information
I_face = cell(nsub,1);
I_place = cell(nsub,1);
% Groupw Mann whitney Z statistics
zI = zeros(ng,ng);
zF = zeros(ng,ng);
ng = length(fieldnames(indices));
for i=1:ng
    for j=1:ng
        for s = 1:nsub
            I_face{s} = I{1,s}(i,j,:);
            I_place{s} = I{2,s}(i,j,:);
            F_face{s} = F{1,s}(i,j,:);
            F_place{s} = F{2,s}(i,j,:);
        end
        zI(i,j) = mann_whitney_group(I_place,I_face);
        zF(i,j) = mann_whitney_group(F_place,F_face);
    end
end
%% Create dataset

dataset.zI = zI;
dataset.zF = zF;

%% Save dataset for plotting in python

fname = 'compare_condition_fc.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'dataset')