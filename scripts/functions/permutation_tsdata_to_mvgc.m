function Fp = permutation_tsdata_to_mvgc(X,x,y,args)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate permutation distribution GC in a 
% given conditions from permuting trials from 2 conditions.

arguments
    X double % Concatenated time series from two conidtions
    x double
    y double
    args.morder double = 5 % model order
    args.regmode char = 'OLS' % model regression (OLS or LWR)
    args.Ns double = 100 % Number of permutations
    args.N double = 56 % Number of trials in a given condition
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = args.morder; regmode = args.regmode; Ns = args.Ns; N = args.N;

[n,m,Nt] = size(X);
trial_idx = 1:Nt;

Fp = zeros(Ns,1);

for s=1:Ns
    fprintf('MVGC: permutation sample %d of %d',s,Ns);
    trials = datasample(trial_idx, N,'Replace',false);
    Xp = X(:,:,trials);
    VAR = ts_to_var_parameters(Xp, 'morder', p, 'regmode', regmode);
    Fp(s) = var_to_mvgc(VAR.A, VAR.V,x,y);
    fprintf('\n');
end

end
