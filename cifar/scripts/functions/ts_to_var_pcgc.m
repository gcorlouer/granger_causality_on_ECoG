function F = ts_to_var_pcgc(X, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate pairwise conditional Granger causality (GC) from VAR model with
% given model order. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defaultMorder = 5;
defaultRegmode = 'OLS';
defaultAlpha = 0.05;
defaultMhtc = 'FDRD';
defaultLR = true;

p = inputParser;

addRequired(p,'X');
addParameter(p, 'regmode', defaultRegmode);  
addParameter(p, 'morder', defaultMorder);  
addParameter(p, 'alpha', defaultAlpha)
addParameter(p, 'mhtc', defaultMhtc)
addParameter(p, 'LR', defaultLR)

parse(p, X, varargin{:});

X = p.Results.X;
morder = p.Results.morder;
regmode = p.Results.regmode;
alpha = p.Results.alpha;
mhtc =  p.Results.mhtc;
LR = p.Results.LR;

%% VAR modeling

VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
disp(VAR.info)
%% Pairwise conditional GC

V = VAR.V;
A = VAR.A;

F = var_to_pwcgc(A,V);
% Put diagonal terms which are NaN by default to 0
F(isnan(F))=0;
% Chose statistical test
% if LR == true
%     stats = pval.LR;
% else 
%     stats = pval.FT;
% end
% % Return statistical significance
% sig = significance(stats,alpha,mhtc);
end