function Fb = sliding_mvgc_bootstrap(X, args)

%%%
% Compute mvgc along sliding window
%%%

arguments
X double % Input time series
args.time double % Time
args.gind struct % group indices
args.mw double = 80 % Number of observations in window
args.nsample double = 100 % Number of observations in window
args.shift double = 10 % Observation shift (controls window overlaps)
args.pdeg double = 1 % Detrending degre
args.morder double = 5 % model order
args.regmode char = 'OLS' % model regression (OLS or LWR)
end

gind = args.gind; mw = args.mw; shift = args.shift; 
morder = args.morder; regmode = args.regmode; time = args.time;
nsample = args.nsample; pdeg = args.pdeg;


[n,m,N] = size(X);
nwin = floor((m - mw)/shift +1);

%% Estimate mvgc
gi = fieldnames(gind);
ng = length(gi); % Number of groups
Fb = zeros(ng, ng, nsample, nwin);

for w=1:nwin
    o = (w-1)*shift;      % window offset
    W = X(:,o+1:o+mw,:); % the window
    [W,~,~,~] = mvdetrend(W,pdeg,[]);
    for i=1:ng
        for j=1:ng
            x = gind.(gi{i});
            y = gind.(gi{j});
            if i == j
               % Compute causal density
               pFb = pwcgc_bootstrap(W,morder,nsample,regmode);
               pFb(isnan(pFb))=0;
               Fb(i,j,:,w) = mean(pFb(x,y),'all');
            else
                % Compute mvgc
               Fb(i,j,:,w) = mvgc_var_bootstrap(W, x,y, morder, nsample, regmode);
            end
        end
    end
end

end