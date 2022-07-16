function pF = ts_to_spgc(X, args)
% Compute spectral GC and integrate over a frequency band
arguments
    X double
    args.morder double = 3;
    args.regmode char = 'OLS';
    args.dim double = 3;
    args.conditional logical = false;
    args.band double = [0 40];
    args.sfreq double = 250;
    args.nfreqs double = 1024
    args.tstat char = 'LR';
    args.alpha double = 0.05;
    args.mhtc char = 'FDRD';
    
end

morder = args.morder; regmode = args.regmode;
dim = args.dim; band = args.band; sfreq = args.sfreq; nfreqs = args.nfreqs;
tstat = args.tstat; alpha = args.alpha; mhtc = args.mhtc; 
conditional = args.conditional;

[n, m, N] = size(X);
f = zeros(n,n,nfreqs+1);
F = zeros(n,n);
sig = zeros(n,n);

% Get frequency vector
freqs = sfreqs(nfreqs,sfreq);


% Estimate spectral pairwise GC
if conditional == false
    for i=1:n
        for j=i:n
            if i==j
                f(i,j,:)=0;
            else
                x = X([i j],:,:);
                VAR = ts_to_var_parameters(x, 'morder', morder, 'regmode', regmode);
                V = VAR.V;
                A = VAR.A;
                disp(VAR.info)
                % Compute spectral GC
                pf = var_to_spwcgc(A,V,nfreqs);
                f(i,j,:) = pf(1,2,:);
                f(j,i,:) = pf(2,1,:);
            end
        end
        % Integrate over frequency band
        F = bandlimit(f,dim, sfreq, band);
        % Test against null of no GC
        nx = 1; ny=1; nz = 0; p=morder;
        pval = mvgc_pval(F,tstat,nx,ny,nz,p,m,N);
        [sig, pcrit] = significance(pval,alpha,mhtc,[]);
    end
else     
    VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
    V = VAR.V;
    A = VAR.A;
    disp(VAR.info)
    % Compute spectral GC
    f = var_to_spwcgc(A,V,nfreqs);
    % Integrate over frequency band
    F = bandlimit(f,dim, sfreq, band);
    % Test against null of no GC
    nx = 1; ny=1; nz = n-nx-ny; p=morder;
    pval = mvgc_pval(F,tstat,nx,ny,nz,p,m,N);
    [sig, pcrit] = significance(pval,alpha,mhtc,[]);
end
% Return pairwise GC and statitics
pF.f=f; pF.freqs = freqs; pF.F = F; pF.sig = sig; pF.pval = pval; pF.pcrit = pcrit; 
end