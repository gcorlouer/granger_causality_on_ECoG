function stat = compare_condition_gGC(X, args)
% compare group Mutual information between conditions using
% permutation testing

arguments
    X double % Concatenated condition time series (along trials dimensions)
    args.gind struct % group indices 
    args.obsStat double % Observed difference in MI between conditions
    args.morder double = 5 % VAR model order
    args.ssmo double = 20 % SS model order
    args.N double = [56 56] % Number of trials per condition
    args.Ns double = 500; % Number of permutations
    args.alpha double = 0.05;
    args.mhtc char = 'FDRD'; 
end

gind = args.gind; gi = fieldnames(gind); 
N = args.N; obsStat = args.obsStat; morder = args.morder; ssmo = args.ssmo;
Ns = args.Ns; alpha = args.alpha; mhtc = args.mhtc;


ng = length(gi); % Number of groups
[~,~,Nt] = size(X); % Number of concatenated trials
trial_idx = 1:Nt;
F = zeros(ng,ng,Ns);
perm_F = cell(2,1);
pf = 2 * morder;
% Compute permutation pairwise MI
for i=1:2
    for s=1:Ns
        fprintf('MVGC: permutation sample %d of %d',s,Ns);
        trials = datasample(trial_idx, N(i),'Replace',false);
        Xp = X(:,:,trials);
        % Estimate SS model
        [A,C,K,V,~,~] = tsdata_to_ss(Xp,pf,ssmo);
        ss.A = A; ss.C = C; ss.K = K; ss.V =V;
        % Compute groupwise GC 
        F(:,:,s) = ss_to_gGC(ss, 'gind', gind);
        fprintf('\n');
    end
    perm_F{i} = F;
end

tstat = perm_F{1} - perm_F{2};
count = zeros(ng,ng);

% Compute statistics
for s=1:Ns
    for i=1:ng
        for j=1:ng
            if abs(tstat(i,j,s))>abs(obsStat(i,j))
                count(i,j)=count(i,j)+1;
            else
                continue 
            end
        end
    end
end

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