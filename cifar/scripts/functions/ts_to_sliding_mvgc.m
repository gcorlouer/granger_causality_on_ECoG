function [F, wtime] = ts_to_sliding_mvgc(X, args)
%%%
% Compute mvgc along sliding window
%%%

arguments
X double % Input time series
args.time double % Time
args.gind struct % group indices
args.pdeg double = 1 % Detrending degre
args.mw double = 80 % Number of observations in window
args.shift double = 10 % Observation shift (controls window overlaps)
args.morder double = 5 % model order
args.regmode char = 'OLS' % model regression (OLS or LWR)
end

gind = args.gind; pdeg = args.pdeg; mw = args.mw; shift = args.shift; 
morder = args.morder; regmode = args.regmode; time = args.time;


[n,m,N] = size(X);
nwin = floor((m - mw)/shift +1);

% Estimate mvgc
gi = fieldnames(gind);
ng = length(gi); % Number of groups
F = zeros(ng, ng, nwin);

for w=1:nwin
    o = (w-1)*shift;      % window offset
    W = X(:,o+1:o+mw,:); % the window
    [W,~,~,~] = mvdetrend(W,pdeg,[]);
    F(:,:,w) = ts_to_mvgc(W, 'gind', gind,'morder', morder,...
        'regmode', regmode);
end

%% Sample to time

wtime = zeros(nwin,mw);
for w=1:nwin
    o = (w-1)*shift; 
    wtime(w,:) = time(o+1:o+mw);
end

end