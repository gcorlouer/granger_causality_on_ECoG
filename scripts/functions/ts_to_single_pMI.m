function MI = ts_to_single_pMI(X, args)

arguments
    X double;
    args.q double = 0;
    args.mhtc char = 'FDRD';
    args.alpha double = 0.05;
end

q = args.q; mhtc = args.mhtc; alpha = args.alpha;

% Initialise variables
[n,m,N] = size(X);
V = zeros(n,n,N);
MI = zeros(n,n,N);
sig = zeros(n,n,N);
pval = zeros(n,n,N);
pcrit = zeros(N,1);
for i=1:N
    % Estimate covariance matrix
    V(:,:,i) = tsdata_to_autocov(X(:,:,i),q);
    % Given Gaussianity, compute mutual information
    [MI(:,:,i),pval(:,:,i)] = cov_to_pwcmi(V(:,:,i),m);
    MI(isnan(MI)) = 0;
end
end 