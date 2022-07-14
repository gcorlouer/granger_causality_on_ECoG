function pF = ts_to_spgc(X, args)
% Compute spectral GC and integrate over a frequency band
arguments
    X double
    args.morder double = 3;
    args.regmode char = 'OLS';
    args.dim double = 3;
    args.band double = [0 40];
    args.sfreq double = 250;
    args.fres double = 1024
    args.tstat char = 'LR';
    args.alpha double = 0.05;
    args.mhtc char = 'FDRD';
end

morder = args.morder; regmode = args.regmode;
dim = args.dim; band = args.band; sfreq = args.sfreq; fres = args.fres;
tstat = args.tstat; alpha = args.alpha; mhtc = args.mhtc;
% Compute VAR model
[n, m, N] = size(X);
VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
V = VAR.V;
A = VAR.A;
disp(VAR.info)
% Compute spectral GC
f = var_to_spwcgc(A,V,fres);
% Integrate over frequency band
F = bandlimit(f,dim, sfreq, band);
% Test against null of no GC
nx = 1; ny=1; nz = n-nx-ny; p=morder;
pval = mvgc_pval(F,tstat,nx,ny,nz,p,m,N);
[sig, pcrit] = significance(pval,alpha,mhtc,[]);
% Return pairwise GC and statitics
pF.F = F; pF.sig = sig; pF.pval = pval; pF.pcrit = pcrit; 
end