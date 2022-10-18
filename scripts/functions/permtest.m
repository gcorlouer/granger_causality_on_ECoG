function stat = permtest(tstat, args)
% Compute pvalue and zscore from permutation test statistics
arguments
    tstat double % test statistic
    args.obsStat char % observed statistic
    args.Ns double = 500; % Number of permutations
    args.alpha double = 0.05; 
    args.mhtc char = 'FDRD'; % multiple correction
end

obsStat = args.obsStat; alpha = args.alpha; mhtc = args.mhtc; Ns = args.Ns ;

% Count number of time observed statistic is "extreme"
count = abs(tstat) > abs(obsStat);
count = sum(count, 3);

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