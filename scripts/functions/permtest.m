function stat = permtest(tstat, args)
% Compute pvalue and zscore from permutation test statistics
arguments
    tstat double % test statistic
    args.obsStat struct % observed statistic
    args.alpha double = 0.05; 
    args.mhtc double = 'FDRD'; % multiple correction
end

obsStat = args.obsStat; alpha = args.alpha; mhtc = args.mhtc;

% Count number of time observed statistic is "extreme"
count = abs(tstat) > abs(obsStat);
count = sum(count, 'all');

% Compute p value and significance
pval = count/Ns;
[sig, pcrit] = significance(pval,alpha,mhtc,[]);

% Compute z score
mT = mean(tstat,3);
sT = std(tstat,0,3);
z = (obsStat - mT)./sT;
zcrit = sqrt(2)*erfcinv(pcrit);

% Return statistics
stat.pval = pval;
stat.sig = sig;
stat.z = z;
stat.zcrit = zcrit;
end