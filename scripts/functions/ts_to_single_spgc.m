function F = ts_to_single_spgc(X, args)
% Compute single trial pairwise band-specific pair conditional GC distribution
arguments
    X double
    args.morder double = 3;
    args.regmode char = 'OLS';
    args.dim double = 3;
    args.band double = [0 40];
    args.sfreq double = 250;
    args.nfreqs double = 1024
end

morder = args.morder; regmode = args.regmode;
dim = args.dim; band = args.band; sfreq = args.sfreq; nfreqs = args.nfreqs;

[n, m, N] = size(X);
f = zeros(n,n,nfreqs+1,N);
F = zeros(n,n,N);

% Get frequency vector
freqs = sfreqs(nfreqs,sfreq);
for i=1:N
    % VAR model
    VAR = ts_to_var_parameters(X(:,:,i), 'morder', morder, 'regmode', regmode);
    V = VAR.V;
    A = VAR.A;
    disp(VAR.info)
    % Compute spectral GC
    f(:,:,:,i) = var_to_spwcgc(A,V,nfreqs);
end
% Integrate over frequency band
F = bandlimit(f,dim, sfreq, band);
end
