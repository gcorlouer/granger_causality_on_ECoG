%% Simulate MVAR model
function [tsdata,var_coef, corr_res, connectivity_matrix] = var_simulation(tsdim, varargin)
%% Argunents
% connect_matrix: matrix of G causal connections
% specrad: spectral radius
% nobs : number of observations
% g = -log(det(R)) where R is the correlation variance exp see corr_rand_exponent
% g is residual multi-information (g = -log|R|): g = 0 yields zero correlation
% w : decay factor of var coefficients

%% 

defaultMorder = 5;
defaultSpecrad = 0.98;
defaultW = []; % decay weighting parameter: empty (default) = don't weight
defaultG = []; % multi-information (g = -log|R|): g = 0 yields zero correlation
defaultNtrials = 1;
defaultNobs = 1000;

p = inputParser;

addRequired(p, 'tsdim')
addParameter(p, 'morder', defaultMorder)
addParameter(p, 'specrad', defaultSpecrad)
addParameter(p, 'w', defaultW)
addParameter(p, 'g', defaultG)
addParameter(p, 'ntrials', defaultNtrials)
addParameter(p, 'nobs', defaultNobs)

parse(p, tsdim, varargin{:});

tsdim = p.Results.tsdim;
morder = p.Results.morder;
specrad = p.Results.specrad;
w = p.Results.w;
g = p.Results.g;
ntrials = p.Results.ntrials;
nobs = p.Results.nobs;

connectivity_matrix = random_connections(tsdim);
var_coef = var_rand(connectivity_matrix,morder,specrad,w);
corr_res = corr_rand(tsdim,g); 
tsdata = var_to_tsdata(var_coef,corr_res,nobs,ntrials);

end

function connectivity_matrix = random_connections(tsdim)

connectivity_matrix = randi([0 1], tsdim);
% self transfer entropy is zero

for i = 1:tsdim
    connectivity_matrix(i,i) = 0;
end

end