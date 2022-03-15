function pGC = ts_to_single_pGC(X, args)

arguments
    X double;
    args.morder double = 3;
    args.regmode = 'OLS';
    args.tstat char = 'LR' % GC statistics
    args.mhtc char = 'FDRD';
    args.alpha double = 0.05;
end

morder = args.morder; regmode = args.regmode; tstat = args.tstat; 
alpha = args.alpha; mhtc = args.mhtc;

% Initialise variables
[n,m,N] = size(X);
pval = zeros(n,n,N);
F = zeros(n,n,N);
sig = zeros(n,n,N);
pval = zeros(n,n,N);
pcrit = zeros(N,1);
for i=1:N
    trial = X(:,:,i);
    VAR = ts_to_var_parameters(trial, 'morder', morder, 'regmode', regmode);
    V = VAR.V;
    A = VAR.A;
    disp(VAR.info)
    % Pairwise conditional GC estimation
    % Single regression
    F(:,:,i) = var_to_pwcgc(A,V);
    F(isnan(F))=0;
    % Dual regression (return number of significant pairs along rolling
    % window
    nx = 1; ny=1;nz = n-nx-ny; p=morder;
    stat = var_to_pwcgc_tstat(X,V,morder,regmode,tstat);
    pval(:,:,i) = mvgc_pval(stat,tstat,nx,ny,nz,p,m,N);
    [sig(:,:,i), pcrit(i)] = significance(pval(:,:,i),alpha,mhtc,[]);
    sig(isnan(sig))=0;
end

pGC.gc = F; pGC.sig = sig; pGC.pval = pval; pGC.pcrit = pcrit;
