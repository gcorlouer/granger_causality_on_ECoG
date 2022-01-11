function [F,et] = mvgc_var_sample(A,V,x,y,m,N,S,regm)

% Generates an empirical distribution of single-regression
% VAR GC estimates based on (known) VAR model parameters.
%
% A,V       VAR model parameters
% x         target variable indices
% y         source variable indices
% m         observations per trial
% N         number of trials
% S         number of samples
% regm      regression mode ('OLS' or 'LWR')
%
% F         empirical GC sample

tic;
p = size(A,3);
F = zeros(S,1);
S10 = round(S/10);
for i = 1:S
	if rem(i,S10) == 0, fprintf('.'); end % progress indicator
	Xs      = var_to_tsdata(A,V,m,N);     % generate i-th data sample
	[As,Vs] = tsdata_to_var(Xs,p,regm);   % estimated model
	F(i)    = var_to_mvgc(As,Vs,x,y);     % sample GC
end
et = toc; % elapsed time
