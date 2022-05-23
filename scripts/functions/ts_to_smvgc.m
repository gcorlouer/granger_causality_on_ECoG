function sGC = ts_to_smvgc(X, args)
% Compute spectral GC and integrate over a frequency band
arguments
    X double
    args.gind struct % group indices 
    args.morder double = 3;
    args.regmode char = 'OLS';
    args.sfreq double = 250;
    args.fres double = 1024
end

% Input parameters
gind = args.gind; morder = args.morder; regmode = args.regmode;
gi = fieldnames(gind); sfreq = args.sfreq; fres = args.fres;
% Compute VAR model
[n, m, N] = size(X);
VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
V = VAR.V;
A = VAR.A;
disp(VAR.info)

% Get frequency vector
freqs = sfreqs(fres,sfreq);
nfreq = size(freqs,1);
% Get arrays dimensions
ng = length(gi); % Number of groups
[n,m,N]=size(X);
F = zeros(ng, ng, nfreq);

% Estimate mvgc stat
for i=1:ng
    for j=1:ng
        % Get indices of specific groups
        x = gind.(gi{i});
        y = gind.(gi{j});
        nx = length(x); ny = length(y); nz = n -nx - ny;
        if i==j
            % Return causal density for diagonal elements
            % Compute pairwise conditional GC
            F(i,j,:) = 0;
        else 
            % Return mvgc between group of populations
            F(i,j,:) = var_to_smvgc(VAR.A,VAR.V,x,y,fres);
        end
    end
end
sGC.F = F; sGC.freqs = freqs;
end