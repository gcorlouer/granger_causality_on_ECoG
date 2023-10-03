function stat = permtest(tstat, obsStat, args)
% Compute pvalue and zscore from permutation test statistics
arguments
    tstat double % test statistic
    obsStat double % observed statistic
    args.Ns double = 500; % Number of permutations
    args.alpha double = 0.05; 
    args.mhtc char = 'FDRD'; % multiple correction
end

alpha = args.alpha; mhtc = args.mhtc; Ns = args.Ns ;

% Count number of time observed statistic is "extreme"
count = abs(tstat) > abs(obsStat);
count = sum(count, 3);
% Compute p value and significance
pval = (count +1)/(Ns + 1); % continuity correction
[sig, pcrit] = significance(pval,alpha,mhtc,[]);

% Compute z score
mT = mean(tstat,3);
sT = std(tstat,0,3);
z = (obsStat - mT)./sT;
zcrit = 1.96;

% Return statistics
stat.T = tstat;
stat.pval = pval;
stat.sig = sig;
stat.z = z;
stat.zcrit = zcrit;
end