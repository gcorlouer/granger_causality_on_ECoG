function f = ts_to_var_spcgc(X, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate pairwise conditional Granger causality (GC) from VAR model with
% given model order. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defaultMorder = 5;
defaultRegmode = 'OLS';
defaultFres = 1024;
p = inputParser;
defaultFs = 250;

addRequired(p,'X');
addParameter(p, 'regmode', defaultRegmode);  
addParameter(p, 'morder', defaultMorder);  
addParameter(p, 'fres', defaultFres);  
addParameter(p, 'fs', defaultFs);  

parse(p, X, varargin{:});

X = p.Results.X;
morder = p.Results.morder;
regmode = p.Results.regmode;
fres = p.Results.fres;
fs = p.Results.fs;

%% VAR modeling

VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
disp(VAR.info)
%% Pairwise conditional GC

V = VAR.V;
A = VAR.A;

f = var_to_spwcgc(A,V, fres);
f(isnan(f)) = 0;
