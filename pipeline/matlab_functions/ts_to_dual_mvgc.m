function gGC = ts_to_dual_mvgc(X, args)
%%%%
% Computes mvgc between all populations of a time series. This function 
% also returns causal density for diagonal terms
%%%%
arguments
    X double % Time series
    args.gind struct % group indices 
    args.morder double = 3 % model order
    args.regmode char = 'OLS' % model regression (OLS or LWR)
    args.tstat char = 'LR' % GC statistics
    args.alpha double = 0.05 % Signifiance threshold
    args.mhtc char = 'FDRD' % multiple hypothesis test
end

% Assign input parameters
gind = args.gind; morder = args.morder; regmode = args.regmode;
gi = fieldnames(gind); tstat = args.tstat; alpha = args.alpha;
mhtc = args.mhtc;

% Get arrays dimensions
ng = length(gi); % Number of groups
[n,m,N]=size(X);
F = zeros(ng, ng); pval = zeros(ng, ng); sig = zeros(ng, ng);
bias = zeros(ng, ng);

% Estimate VAR model
VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);

% Estimate mvgc stat
for i=1:ng
    for j=1:ng
        % Get indices of specific groups
        x = gind.(gi{i});
        y = gind.(gi{j});
        nx = length(x); ny = length(y); nz = n -nx - ny;
        if i==j
            % Return causal density for diagonal elements
            % Compute pairwise conditional GC
            pF = var_to_pwcgc_tstat(X,VAR.V,morder,regmode,tstat);
            % Return 0 when group of channel is singleton
            pF(isnan(pF))=0;
            % Average over each type of populations
            F(i,j) = mean(pF(x,y),'all');
        else 
            % Return mvgc between group of populations
            F(i,j) = var_to_mvgc_tstat(X,VAR.V,x,y,morder,regmode,tstat);
        end
        % Compute corrected pvalue and significance
        pval(i,j) = mvgc_pval(F(i,j),tstat,nx,ny,nz,morder,m,N);
    end
end
[sig, pcrit] = significance(pval,alpha,mhtc,[]);
gGC.F = F; gGC.sig = sig; gGC.pval = pval; gGC.pcrit = pcrit; 
end