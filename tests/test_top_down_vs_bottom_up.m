%% Test Compare top down vs bottom up GC

% Input parameters
nsim = 10;
input_parameters;
morder = 3;
specrad = 0.5;
m = 500;
N = 56;
tsdim = 10;

%% Build connectivity matrix (top down direction)

nv = tsdim;
nR = 2;
nF = nv - nR;
R_idx = randperm(nv, nR)';
F_idx = setdiff(1:nv, R_idx);
C = zeros(nv, nv);
C(R_idx, F_idx) = ones(nR,nF);

%% Simulate time series

A = var_rand(C,morder,specrad,[]);
E = corr_rand(tsdim,[]); 
X = var_to_tsdata(A,E,m,N);

%% Estimate single trial pcgc

F = ts_to_single_pair_unconditional_gc(X,'morder',morder, 'regmode', regmode);

%% Get top down and bottom up GC

F_td = F(R_idx, F_idx,:);
F_td = reshape(F_td, numel(F_td),1);

F_bu = F(F_idx, R_idx,:);
F_bu = reshape(F_bu, numel(F_bu),1);


%% Compare top down and bottom up GC
% Note that we will use groupwise z scores between subjects in the real
% case

z = mann_whitney(F_bu,F_td);
pval = erfc(abs(z)/sqrt(2));
zcrit = sqrt(2)*erfcinv(alpha);
if z>zcrit
    fprintf('Top down dominates\n')
elseif z<-zcrit
    fprintf('Bottom up dominate\n')
else
    fprintf('No direction dominates\n')
end









