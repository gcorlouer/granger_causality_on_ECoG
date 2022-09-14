input_parameters
nobs = 250;
tsdim = 10;
N = 20;
rho = 0.9;
p = 5;
plotm = [];
fres = 1024;
[X,~, ~, C] = var_simulation(tsdim, ...
    'nobs',nobs, 'ntrials', N, 'specrad', rho, 'morder', p);

%% Estimate var model order with multiple information criterion

[A, V, ~] = tsdata_to_var(X,p,regmode);

g1 = [1 2 3];
g2 = [4 5 6];
g3 = [7 8 9];

%% Compute group GC 


groups = {g1, g2, g3};

f = var_to_sgwcgc(A,V,groups,fres);

%% Compute group global GC

gf = var_to_sgwcggc(A,V,groups,fres);

%% Test single trial pair and group GC 

indices.g1 = g1;
indices.g2 = g2;
indices.g3 = g3;

%% Single trial pair GC

tic
pF = ts_to_single_spgc(X,'morder',morder,'band', band, ...
               'regmode',regmode, 'sfreq', sfreq);
toc
    
%% Single trial group GC

tic
gF = ts_to_single_smvgc(X,'gind',indices,'morder',morder,'band', band, ...
               'regmode',regmode, 'sfreq', sfreq, 'dim', dim);
toc