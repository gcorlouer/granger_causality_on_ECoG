input_parameters
nobs = 250;
tsdim = 10;
N = 20;
rho = 0.9;
p = 5;
plotm = [];
fres = 1024;
[X,~, ~, C] = var_simulation(tsdim, ...
    'nobs',nobs, 'ntrials', N, 'specrad', rho, 'morder', p);

%%

% Pick n trials amond N at random without replacement
% Or form n-tuple of trials without replacement
% Given integer n. 
% Take n trials of X without replacement. 
% Form n trial sub time series
% Estimate GC on time series 
% Return n-trial GC distribution

N = size(X,3);
nSubSample = floor(N/k);
for i=1:nSubsample
    