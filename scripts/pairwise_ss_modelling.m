%% In this script we explore SS modelling on pair of channels
%% Input parameters 
input_parameters;
ncdt = length(conditions);
nsub = length(cohort);

%% Estimate pairwise VAR model order

subject = 'DiAs';
condition = 'Face';

        
gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
    'condition',condition, 'suffix', suffix);
X = gc_input.X;
indices = gc_input.indices;
[n,m,N] = size(X);

moaic =  zeros(n,n);
mobic =  zeros(n,n);
mohqc =  zeros(n,n);
molrt =  zeros(n,n);

for i=1:n
    for j=i+1:n
        x = X([i,j],:,:);
        % Estimate VAR model order
        [moaic(i,j),mobic(i,j),mohqc(i,j),molrt(i,j)] = tsdata_to_varmo(x, ... 
            momax,regmode,alpha,pacf,plotm,verb);
    end
end

%% Estimate pairwise SS model order
pf = 2*morder;
mosvc = zeros(n,n);
for i=1:n
    for j=i+1:n
        x = X([i,j],:,:);
        [mosvc(i,j),rmax] = tsdata_to_ssmo(x,pf,plotm);
        [A,C,K,V,Z,E] = tsdata_to_ss(x,pf,mosvc(i,j));
    end
end
mosvcMean = mean(mosvc, 'all');
mosvcMean = 2*(n^2/(n * (n-1))) * mosvcMean;
mosvcMean = round(mosvcMean);