function F = pwcgc_bootstrap(X,morder,S,regmode)

tic;
[n,m,N] = size(X);
s = randi(N,N,S); % subsample trials with replacement
F = zeros(n,n,S);
S10 = round(S/10);
for i = 1:S
	if rem(i,S10) == 0, fprintf('.'); end             % progress indicator
	Xs      = X(:,:,s(:,i));                          % select i-th bootstrap sample
    VAR = ts_to_var_parameters(Xs, 'morder', morder, 'regmode', regmode);
	F(:,:,i) = var_to_pwcgc(VAR.A,VAR.V); % GC 
end
et = toc; % elapsed time