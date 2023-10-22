function I = ts_to_single_mvmi(X, args)

arguments
    X double % Time series
    args.gind struct % group indices 
end

% Define input arguments
gind = args.gind;
gi = fieldnames(gind);
ng = length(gi); % Number of groups
[n,m,N]=size(X);
q = 0;

% Output argument
I = zeros(ng, ng, N); 

% Estimate groupwise mutual information on single trials
for k=1:N
    % Esimtate autocovariance
    V = tsdata_to_autocov(X(:,:,k),q);
    % Loop over groups of channels
    for i=1:ng
        for j=1:ng
            % Take specific groups
            x = gind.(gi{i});
            y = gind.(gi{j});
            % Correlation density
            if i==j
                pI = cov_to_pwcmi(V,m);
                pI(isnan(pI))=0;
                I(i,j,k) = mean(pI(x,y), 'all');
            % Groupwise mutual information    
            else
                % Return MI between group of populations
                group = {x, y};
                C = cov_to_gwcmi(V,group,m,1);
                I(i,j,k) = C(1,2);
            end            
        end
    end
end
end