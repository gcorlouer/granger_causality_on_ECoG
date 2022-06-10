%% Compare top down and bottom up GC with pairs (would need to be done 
% on an individual subject basis)

nsim = 100;
z = zeros(nsim,1);
pval = zeros(nsim,1);
input_parameters;
morder = 3;
specrad = 0.9;
m = 300;
N = 56;
tsdim = 10;

[X,var_coef, E, connectivity_matrix] = var_simulation(tsdim, 'ntrials',N, 'nobs',m,...
        'morder',morder);

%% Estimate single trial pcgc

F = ts_to_single_pGC(X,'morder',morder, 'regmode', regmode);

%% Get top down and bottom up GC

nv = 10;
nR = 4;
nF = nv - nR;
R_idx = randi([1 nv], nR,1)';
F_idx = setdiff(1:10, R_idx);

F_td = F(R_idx, F_idx,:);

F_bu = F(F_idx, R_idx,:);

%% Estimate paired z score
p = zeros(nR,nF);
for i=1:nR
    for j=1:nF
        x = squeeze(F_td(i,j,:));
        y = squeeze(F_bu(j,i,:));
        p(i,j) = signrank(x, y);
    end
end
[sig,pcrit] = significance(p, alpha, 'FDRD');