function pF = test_ts_to_spgc(X, args)

arguments
    X double
    args.morder double = 3;
    args.regmode char = 'OLS';
    args.fres double = 1024; %frequency bins
    args.band double = [0 20];
    args.alpha double = 0.05;
    args.mhtc char = 'FDRD';
end
morder = args.morder; regmode = args.regmode; 
tstat = args.tstat; alpha = args.alpha; mhtc = args.mhtc;

[n, m, N] = size(X);
% Compute significance against null distribution
nx = 1; ny=1; nz = n-nx-ny; p=morder;
VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
V = VAR.V;
A = VAR.A;
disp(VAR.info)
% Dual regression
f = var_to_spwcgc(A,V,fres);
pval = mvgc_pval(F,tstat,nx,ny,nz,p,m,N);
[sig, pcrit] = significance(pval,alpha,mhtc,[]);
% Compute F crit
% d = morder*nx*ny;
% F_crit = icdf('chi2',1-pcrit,d);
% Return pairwise GC and statitics
pF.f = F; pF.sig = sig; pF.pval = pval; pF.pcrit = pcrit; 
end