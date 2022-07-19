input_parameters;
ncdt = length(conditions);
nsub = length(cohort);
subject_id = 'DiAs';
%%
F = cell(3,1);
mosvc = cell(3,1);
Fvar = cell(3,1);

for c=1:3
    % Read condition specific time series
    gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
        'condition',conditions{c}, 'suffix', suffix);
    % Read conditions specific time series
    X = gc_input.X;
    % VAR model estimation
    [moaic,mobic,mohqc,molrt] = tsdata_to_varmo(X, ... 
                        momax,regmode,alpha,[],[],[]);
    pf = 2 * moaic;
    % SS model estimation
    plotm = 1;
    [mosvc{c},rmax] = tsdata_to_ssmo(X,pf,plotm);
    %%
    % SS model estimation
    [A,C,K,V,~,~] = tsdata_to_ss(X,pf,mosvc{c});
    % GC estimation
    F{c} = ss_to_pwcgc(A,C,K,V);
    %% VAR model estimation
    [A,V,E] = tsdata_to_var(X,moaic,regmode); 
    Fvar{c} = var_to_pwcgc(A,V);
end

%% Plot SS model GC
n = size(F{1},1);
xtick = cell(1,n);
indices = gc_input.indices;
for i=1:n
    xtick{i} = 'F';
end

figure
plotm = 0;
ptitle = {'Rest', 'Face', 'Place'};
plot_gc(F',ptitle,[],plotm,0)

%% Plot VAR model GC
n = size(F{1},1);
xtick = cell(1,n);
indices = gc_input.indices;
for i=1:n
    xtick{i} = 'F';
end

figure
plotm = 0;
ptitle = {'Rest', 'Face', 'Place'};
plot_gc(Fvar',ptitle,[],plotm,0)