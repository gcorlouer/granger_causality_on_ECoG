function [MI, sig] = ts_to_MI(X, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate mutual information MI from covariance matrix at a given lag
% Works best with normally distributed time series
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defaultQ = 0;
defaultAlpha = 0.05;
defaultMhtc = 'FDRD';

p = inputParser;

addRequired(p,'X');
addParameter(p, 'q', defaultQ);
addParameter(p, 'mhtc', defaultMhtc);
addParameter(p, 'alpha', defaultAlpha);

parse(p, X, varargin{:});
X = p.Results.X;
q = p.Results.q;
mhtc = p.Results.mhtc;
alpha = p.Results.alpha;


% Estimate covariance matrix
[n,m,N] = size(X);
V = tsdata_to_autocov(X,q);

% Given Gaussianity, compute mutual information
nobs = m * N;
[MI,pval] = cov_to_pwcmi(V,nobs);
MI(isnan(MI)) = 0;
% Return LR test statistics
% pval = stats.LR.pval;
sig = significance(pval,alpha,mhtc);

end