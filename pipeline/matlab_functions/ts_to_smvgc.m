function sGC = ts_to_smvgc(X, args)
% Compute spectral GC and integrate over a frequency band
arguments
    X double
    args.gind struct % group indices 
    args.morder double = 5;
    args.regmode char = 'OLS';
    args.sfreq double = 250;
    args.fres double = 1024;
    args.dim double = 3;
    args.band double = [60 80];
    args.alpha double = 0.05;
    args.tstat char = 'F';
    args.mhtc char = 'FDRD';
end

% Input parameters
% VAR modeling
gind = args.gind; morder = args.morder; regmode = args.regmode;
gi = fieldnames(gind);
% Frequency stuff
sfreq = args.sfreq; fres = args.fres; band=args.band; dim = args.dim; 
% Statistics
alpha = args.alpha; mhtc = args.mhtc; tstat = args.tstat;

% Compute VAR model
[n, m, N] = size(X);
VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
disp(VAR.info)

% Get frequency vector
freqs = sfreqs(fres,sfreq);
nfreq = size(freqs,1);

% Get arrays dimensions
ng = length(gi); % Number of groups
[n,m,N]=size(X);
f = zeros(ng, ng, nfreq);
F =  zeros(ng, ng);
sig = zeros(ng, ng);
% Estimate mvgc stat
for i=1:ng
    for j=1:ng
        % Get indices of specific groups
        x = gind.(gi{i});
        y = gind.(gi{j});
        nx = length(x); ny = length(y); nz = n -nx - ny;
        if i==j
            % Return multi-information for diagonal elements
            fk = zeros(nx,nfreq);
            for k =1:nx
                xk = x;
                xk(k) = [];
                fk(k,:) = var_to_smvgc(VAR.A,VAR.V,x(k), xk,fres);
            end
            f(i,j,:) = sum(fk,1);
        else 
            % Return mvgc between group of populations
            f(i,j,:) = var_to_smvgc(VAR.A,VAR.V,x,y,fres);
        end
    end
end
% Integrate over frequency band
F = bandlimit(f,dim, sfreq, band);
% Test against null of no GC
pval = mvgc_pval(F,tstat,nx,ny,nz,morder,m,N);
[sig, pcrit] = significance(pval,alpha,mhtc,[]);
sGC.F=F; sGC.f = f; sGC.sig=sig; sGC.pcrit = pcrit;
sGC.freqs = freqs; sGC.indices = gind; 
end