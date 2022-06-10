    %% Test Compare top down vs bottom up GC
nsim = 100;
z = zeros(nsim,1);
pval = zeros(nsim,1);
input_parameters;
morder = 3;
specrad = 0.9;
m = 300;
N = 56;
tsdim = 10;
for i=1:nsim
    %% Simulate time series

    [X,var_coef, E, connectivity_matrix] = var_simulation(tsdim, 'ntrials',N, 'nobs',m,...
        'morder',morder);

    %% Estimate single trial pcgc

    F = ts_to_single_pGC(X,'morder',morder, 'regmode', regmode);

    %% Get top down and bottom up GC
    nv = 10;
    nR = 4;
    R_idx = randi([1 nv], nR,1)';
    F_idx = setdiff(1:10, R_idx);

    F_td = F(R_idx, F_idx,:);
    F_td = reshape(F_td, numel(F_td),1);

    F_bu = F(F_idx, R_idx,:);
    F_bu = reshape(F_bu, numel(F_bu),1);


    %% Compare top down and bottom up GC
    % Note that we will use groupwise z scores between subjects in the real
    % case
    z(i) = mann_whitney(F_bu,F_td);
    pval(i) = erfc(abs(z(i))/sqrt(2));
    zcrit = sqrt(2)*erfcinv(alpha);
    if z(i)>zcrit
        fprintf('Top down dominates')
    elseif z(i)<-zcrit
        fprintf('Bottom up dominate')
    else
        fprintf('No direction dominates')
    end
end

zmean = mean(z);
if zmean>zcrit
        fprintf('Top down dominates')
elseif zmean<-zcrit
    fprintf('Bottom up dominate')
else
    fprintf('No direction dominates')
end









