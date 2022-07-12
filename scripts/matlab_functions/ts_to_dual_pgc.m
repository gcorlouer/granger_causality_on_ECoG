function pF = ts_to_dual_pgc(X, args)

arguments
    X double
    args.morder double = 3;
    args.regmode char = 'OLS';
    args.tstat char = 'LR';
    args.alpha double = 0.05;
    args.mhtc char = 'FDRD';
end
morder = args.morder; regmode = args.regmode; 
tstat = args.tstat; alpha = args.alpha; mhtc = args.mhtc;

[n, m, N] = size(X);
% Compute significance against null distribution
nx = 1; ny=1; nz = n-nx-ny; p=morder;
% Dual regression
F = var_to_pwcgc_tstat(X,[],morder,regmode,tstat);
pval = mvgc_pval(F,tstat,nx,ny,nz,p,m,N);
[sig, pcrit] = significance(pval,alpha,mhtc,[]);
% Return pairwise GC and statitics
pF.F = F; pF.sig = sig; pF.pval = pval; pF.pcrit = pcrit; 
end