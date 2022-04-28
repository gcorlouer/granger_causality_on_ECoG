function gMI = ts_to_mvmi(X, args)
%%%%
% Computes groupwise MI between all populations of a time series. 
%%%%
arguments
    X double % Time series
    args.gind struct % group indices 
    args.tstat char = 'LR' % GC statistics
    args.alpha double = 0.05 % Signifiance threshold
    args.mhtc char = 'FDRD' % multiple hypothesis test
end

% Assign input parameters
gind = args.gind; 
gi = fieldnames(gind); alpha = args.alpha;
mhtc = args.mhtc;

% Get arrays dimensions
ng = length(gi); % Number of groups
[n,m,N]=size(X);

I = zeros(ng, ng); pval = zeros(ng, ng); sigI = zeros(ng, ng);

% Estimate 0-lag Autocovariance
q=0;
V = tsdata_to_autocov(X,q);

% Estimate Mutual information
% Estimate mvgc stat
for i=1:ng
    for j=1:ng
        % Get indices of specific groups
        x = gind.(gi{i});
        y = gind.(gi{j});
        if i==j
            pI = cov_to_pwcmi(V,m);
            pI(isnan(pI))=0;
            I(i,j) = mean(pI(x,y), 'all'); 
        else
            group = {x,y};
            % Compute groupwise mutual information
            % Output is 2x2 symmetric matrix
            [C,stats] = cov_to_gwcmi(V,group,m,N);
            % We take off diagonal entries which interest us
            I(i,j) = C(1,2);
            % Compute p value
            pval(i,j) = stats.LR.pval(1,2);
        end
    end
end
% Compute significance against null and correct for multiple hypotheses.
[sig, pcrit] = significance(pval,alpha,mhtc,[]);
gMI.F = I; gMI.sig = sig; gMI.pval = pval; gMI.pcrit = pcrit;
end
