function stat = compare_TD_BU_pgc(X, indices, args)
% This function compare top down with bottom up pariwise unconditional GC 
% by permuting pairs of face and retinotopic channels, estimating pGC
% from state space model and comparing the statistic with observed difference
% in TD and BU GC

arguments
    X double % Input time series
    indices struct 
    args.morder double = 5 % VAR model order
    args.ssmo double = 20 % SS model order
    args.Ns double = 100 % Number of permutations
    args.alpha double = 0.05;
    args.mhtc char = 'FDRD'; 
end

morder = args.morder; ssmo = args.ssmo;
Ns = args.Ns; alpha = args.alpha; mhtc = args.mhtc;

pf = 2 * morder;

R_idx = indices.('R');
F_idx = indices.('F');

% Sizes
nR = length(R_idx);
nF = length(F_idx);
count = zeros(nR,nF);
T = zeros(nR,nF,Ns);
Ta = zeros(nR,nF);

[~,~,N] = size(X);

for i=1:nR
    for j=1:nF
        iR = R_idx(i);
        iF = F_idx(j);
        x = X(iR,:,:);
        y = X(iF,:,:);
        % Compute observed TD - BU GC
        [A,C,K,V,~,~] = tsdata_to_ss(X([iR iF],:,:), pf, ssmo);
        F = ss_to_pwcgc(A,C,K,V);
        Ta(i,j) = F(1,2) - F(2,1);
        % Concatenate in one multitrial time series
        xRF = cat(3,x,y);
        [~,~,Nt] = size(xRF);
        for s=1:Ns
            fprintf('Permutation sample %d of %d',s,Ns);
            % Permute trial index
            trials = randperm(Nt);
            trialsR = trials(1:N);
            trialsF = trials(N+1:Nt);
            xR = xRF(:,:,trialsR);
            xF = xRF(:,:,trialsF);
            % Concatenate pairwise time series
            xRFp = cat(1,xR,xF);
            % Estimate permutation GC
            [A,C,K,V,~,~] = tsdata_to_ss(xRFp, pf, ssmo);
            Fp =  ss_to_pwcgc(A,C,K,V);
            % Compute permutation statistic TD - BU
            T(i,j,s) = Fp(1,2)-Fp(2,1);
            fprintf('\n');
            if abs(T(i,j,s))>abs(Ta(i,j))
                count(i,j) = count(i,j)+1;
            else
                continue 
            end
        end
     end
end

% Compute p value and significance
pval = count./Ns;
[sig, pcrit] = significance(pval,alpha,mhtc,[]);

% Compute z score
mT = mean(T,3);
sT = std(T,0,3);
z = (Ta - mT)./sT;
zcrit = sqrt(2)*erfcinv(pcrit);

% Return statistics
stat.pval = pval;
stat.sig = sig;
stat.z = z;
stat.zcrit = zcrit;

end
