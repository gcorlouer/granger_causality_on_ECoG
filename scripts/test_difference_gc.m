%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, we test the use of wilcoxon sum rank test to test
% difference in GC accross conditions on simulated VAR models.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialise parameters
% Restrict to n=2 channels for the example
% Time series parameters
morder = 3; specrad = 0.92; n=2;
% Observations and trials
m = {1000, 1000}; N = {56, 120};
% Chose distinct connectivity matrices
cmatrix1 = [[0 ; 1] [0 ; 0]];
cmatrix2 = [[0 ; 1] [0 ; 0]];

cmatrix = {cmatrix1 cmatrix2} ;
ncomp = length(cmatrix);
ts = cell(ncomp,1); F = cell(ncomp,1);

%% Simulate VAR in both conditions and compute F

for c=1:ncomp
    % Simulate time series from connectivity matrices
    cm = cmatrix{c};
    var_coef = var_rand(cm,morder, specrad, []);
    corr_res = corr_rand(n,[]); 
    ts{c} = var_to_tsdata(var_coef,corr_res,m{c},N{c});
    % Estimate pcgc
    for i=1:N{c}
        X = ts{c}(:,:,i);
        f = ts_to_var_pcgc(X,'morder', morder,...
                    'regmode', regmode,'alpha', alpha,'mhtc', mhtc, 'LR', LR);
        F{c}(:,:,i) = f;
    end
end

%% Compute statistics to compare F

z_mann = zeros(n,n); pval = zeros(n,n); p = zeros(n,n); h =zeros(n,n);
zval = zeros(n,n);
for i=1:n
    for j=1:n
        Fc1 = squeeze(F{1}(i,j,:));
        Fc2 = squeeze(F{2}(i,j,:));
        z_mann(i,j) = mann_whitney(Fc1, Fc2);
        pval(i,j) = 2*(1-normcdf(abs(z_mann(i,j)))); 
        [sig, pcrit] = significance(pval,alpha,mhtc,[]);
        zcrit = -sqrt(2)*erfcinv(2*(1-pcrit));
        [p(i,j),h(i,j),stats(i,j)] = ranksum(Fc2, Fc1, 'tail','right');
        zval(i,j) = stats(i,j).zval;
    end
end

%% Plot 

diff = cmatrix2 - cmatrix1;
subplot(3,2,1)
imagesc(diff)
title('True difference')
colormap(gray)
colorbar
subplot(3,2,2)
imagesc(pval)
title('pval Mann')
colormap(gray)
colorbar
subplot(3,2,3)
imagesc(p)
title('pval Wilcoxon')
colormap(gray)
colorbar
subplot(3,2,4)
imagesc(z_mann)
title('Z Mann')
colormap(gray)
colorbar
subplot(3,2,5)
imagesc(zval)
title('Z Wilcoxon')
colormap(gray)
colorbar



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%