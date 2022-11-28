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
    args.dim double = 3; % Integration dimension
    args.sfreq double = 250; % Sampling frequency
    args.nfreqs double = 1024; % Number of frequency bins
    args.band double = [0 40]; % frequency band
end

morder = args.morder; ssmo = args.ssmo;
Ns = args.Ns; alpha = args.alpha; mhtc = args.mhtc;
sfreq = args.sfreq; dim = args.dim; band = args.band; nfreqs = args.nfreqs;

bandstr = mat2str(band);

pf = 2 * morder;

% Get retonotopic and Face channels indices
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
        f = ss_to_spwcgc(A,C,K,V, nfreqs);
        F = bandlimit(f,dim, sfreq, band);
        Ta(i,j) = F(1,2) - F(2,1);
        % Concatenate in one multitrial time series
        xRF = cat(3,x,y);
        [~,~,Nt] = size(xRF);
        for s=1:Ns
            if mod(s, Ns/10) == 0
                fprintf('TD vs BU %s GC permutation sample %d of %d \n',bandstr,s,Ns);
            end
            % Permute trial index
            trials = randperm(Nt);
            trialsR = trials(1:N);
            trialsF = trials(N+1:Nt);
            % Permuted Retinotopic time series
            xR = xRF(:,:,trialsR);
            % Permuted Face time series
            xF = xRF(:,:,trialsF);
            % Concatenate in 2D time series
            xRFp = cat(1,xR,xF);
            % Estimate permutation GC
            [A,C,K,V,~,~] = tsdata_to_ss(xRFp, pf, ssmo);
            fp =  ss_to_spwcgc(A,C,K,V, nfreqs);
            Fp = bandlimit(fp,dim, sfreq, band);
            % Compute permutation statistic TD - BU
            T(i,j,s) = Fp(1,2)-Fp(2,1);
            if abs(T(i,j,s))>abs(Ta(i,j))
                count(i,j) = count(i,j)+1;
            else
                continue 
            end
        end
        fprintf('\n');
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
stat.T = T;
stat.pval = pcrit;
stat.sig = sig;
stat.z = z;
stat.zcrit = zcrit;

end
