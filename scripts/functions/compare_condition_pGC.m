function stat = compare_condition_pGC(X, args)
% Compare pairwise conditional GC between conditions using
% permutation testing
arguments
    X double % Concatenated condition time series (along trials dimensions)
    args.obsStat double % Observed difference in F between conditions
    args.N double = [56 56] % Number of trials per condition
    args.morder double = 5 % VAR model order
    args.ssmo double = 20 % SS model order
    args.Ns double = 500; % Number of permutations
    args.alpha double = 0.05;
    args.mhtc char = 'FDRD'; 
end

N = args.N; obsStat = args.obsStat; morder = args.morder; ssmo = args.ssmo;
Ns = args.Ns; alpha = args.alpha; mhtc = args.mhtc;

[n,~,Nt] = size(X);
trial_idx = 1:Nt;
F = zeros(n,n,Ns);
perm_F = cell(2,1);
pf = 2 * morder;

% Compute permutation pairwise GC
for i=1:2
    for s=1:Ns
        fprintf('MVGC: permutation sample %d of %d',s,Ns);
        trials = datasample(trial_idx, N(i),'Replace',false);
        Xp = X(:,:,trials);
        % SS model
        [A,C,K,V,~,~] = tsdata_to_ss(Xp,pf,ssmo);
        % Compute pairwise MI 
        F(:,:,s) = ss_to_pwcgc(A,C,K,V);
        fprintf('\n');
    end
    perm_F{i} = F;
end
tstat = perm_F{1} - perm_F{2};
count = zeros(n,n);

% Compute statistics
for s=1:Ns
    for i=1:n
        for j=1:n
            if tstat(i,j,s)>obsStat(i,j)
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