function GC = var_to_dualGC(X, VAR, args)
% Test pairwise and groupwise GC against the null 
arguments
    X double
    VAR struct % Var model
    args.connect char % connectivity: 'groupwise' or 'pairwise'
    args.group cell % group indices
    args.morder double = 5 % group indices
    args.regmode char = 'LWR' % group indices
    args.test char = 'F';
    args.alpha double = 0.05;
    args.mhtc char = 'FDRD';
end

group = args.group; connect = args.connect; morder = args.morder;
regmode = args.regmode; alpha = args.alpha; mhtc = args.mhtc; test = args.test;

[n,m,N] = size(X);
ng = length(group); % number of groups
pval = zeros(ng,ng);

if strcmp(connect, 'groupwise')
    % Compute between group GC
    tstat = var_to_gwcgc_tstat(X,VAR.V,group,morder,regmode,test);     
    % Compute within group GC
    for ig=1:ng
        tstat(ig,ig) = var_to_cggc_tstat(X,VAR.V, group{ig},morder,regmode,test);
    end
    % Compute pvalue
    for i=1:ng
        for j=1:ng
            if i==j
                nx = length(group{i});
                nz = n - nx;
                cdf = ggc_cdf(test,nx,nz,morder,m,N);
                pval(i,j) = 1-cdf(tstat(i,j));
            else
                nx = length(group{i});
                ny = length(group{j});
                nz = n - nx - ny;
                pval(i,j) = mvgc_pval(tstat(i,j),test,nx,ny,nz,morder,m,N);
            end
        end
    end
else
    % Compute pairwise GC
    tstat = var_to_pwcgc_tstat(X,VAR.V,morder,regmode,test);
    pval = mvgc_pval(tstat,test,1,1,n-2,morder,m,N);
end
% Calculate significance
[sig, pcrit] = significance(pval,alpha,mhtc,[]);

GC.F = tstat;
GC.sig = sig;
GC.pcrit = pcrit;

end