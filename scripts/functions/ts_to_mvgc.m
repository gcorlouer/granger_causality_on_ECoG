function F = ts_to_mvgc(X, args)
%%%%
% Computes mvgc between all populations of a time series. This function 
% also returns causal density for diagonal terms
%%%%
arguments
    X double % Time series
    args.gind struct % group indices 
    args.morder double = 5 % model order
    args.regmode char = 'OLS' % model regression (OLS or LWR)
end

gind = args.gind; morder = args.morder; regmode = args.regmode;
gi = fieldnames(gind);
ng = length(gi); % Number of groups

[n,m,N]=size(X);
F = zeros(ng, ng);
VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
for i=1:ng
    for j=1:ng
        x = gind.(gi{i});
        y = gind.(gi{j});
        if i==j
            % Return causal density for diagonal elements
            % Compute pairwise conditional GC
            pF = var_to_pwcgc(VAR.A,VAR.V);
            % Return 0 when group of channel is singleton
            pF(isnan(pF))=0;
            F(i,j) = mean(pF(x,y),'all');
        else 
            % Return mvgc between group of populations
            F(i,j) = var_to_mvgc(VAR.A,VAR.V,x,y);
        end
    end
end
end