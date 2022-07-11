function [F,et] = mvgc_var_sample_tstat(A,V,x,y,m,N,S,regm,tstat)

% Generates an empirical distribution of F or likelihood
% ratio (chi^2) VAR GC test statistics based on (known)
% VAR model parameters.
%
% A,V       VAR model parameters
% x         target variable indices
% y         source variable indices
% m         observations per trial
% N         number of trials
% S         number of samples
% regm      regression mode ('OLS' or 'LWR')
% tstat     test statistic: F or likelihood ratio (chi^2)
%
% F         empirical GC sample

tic;
p = size(A,3);
F = zeros(S,1);
S10 = round(S/10);
for i = 1:S
	if rem(i,S10) == 0, fprintf('.'); end             % progress indicator
	Xs   = var_to_tsdata(A,V,m,N);                    % generate i-th data sample
	F(i) = var_to_mvgc_tstat(Xs,[],x,y,p,regm,tstat); % GC test statistic (F or likelihood-ratio)
end
et = toc; % elapsed time
