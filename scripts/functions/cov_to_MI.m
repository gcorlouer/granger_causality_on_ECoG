function MI = cov_to_MI(V, args)

arguments
    V double % ss model
    args.connect char % connectivity: 'groupwise' or 'pairwise'
    args.group cell % group indices
    args.Nsample double % number of samples (trials x observations)
    args.alpha double = 0.05;
    args.mhtc char = 'FDRD';
end

group = args.group; connect = args.connect; Nsample = args.Nsample;
alpha = args.alpha; mhtc = args.mhtc;


ng = length(group);

if strcmp(connect, 'groupwise')
    % Compute between group spectral GC
    F = cov_to_gwcmi(V, group);
    % Compute within group spectral GC
    pval = zeros(ng, ng);
    for ig=1:ng
        nx = length(group{ig});
        F(ig,ig,:) = cov_to_cmii(V, group{ig});
        pval(ig,ig) = 1 - chi2cdf(F(ig,ig) * Nsample, nx*(nx-1)/2);        
    end
    for a = 1:ng
         for b = a+1:ng
             nga = length(group{a});
             ngb = length(group{b});
             pval(a,b) = 1 - chi2cdf(F(a,b) * Nsample, nga * ngb);
         end
    end
    pval = symmetrise(pval);
else
    % Compute pairwise conditional spectral GC
    F = cov_to_pwcmi(V);
    pval = 1 - chi2cdf(F * Nsample, 1);
end

[sig, pcrit] = significance(pval,alpha,mhtc,[]);

% Return statistics
MI.F = F; % effect size
MI.pcrit = pcrit;
MI.sig = sig;

end