function testStat = permutation_tsdata_to_mvgc(X,x,y,args)
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

%F1 = zeros(Ns,1);
%F2 = zeros(Ns,1);
testStat = zeros(Ns,1);

for s=1:Ns
    fprintf('MVGC: permutation sample %d of %d',s,Ns);
    trials = randperm(Nt);
    trial1 = trials(1:N);
    trial2 = trials(N+1:Nt);
    %trials = datasample(trial_idx, N,'Replace',false);
    %Xp = X(:,:,trials);
    X1 = X(:,:,trial1);
    X2 = X(:,:,trial2);
    VAR1 = ts_to_var_parameters(X1, 'morder', p, 'regmode', regmode);
    VAR2 = ts_to_var_parameters(X2, 'morder', p, 'regmode', regmode);
    F1 = var_to_mvgc(VAR1.A, VAR1.V,x,y);
    F2 = var_to_mvgc(VAR2.A, VAR2.V,x,y);
    testStat(s) = F1 - F2;
    fprintf('\n');
end

end
