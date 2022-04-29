%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Rather than a bootstrap, this demonstrates use of the single-trial empirical
% GC distribution to test for stochastic dominance between conditions for multi-
% trial (epoched) data.
%
% In this demo the SAME (random) VAR model is used to generate two multi-trial
% time series, to represent the situation where the population GC is the same
% under the two conditions - thus the test should (at leat 1-alpha of the time,
% where alpha is the significance level), infer NO stochastic dominance.
%
% A limitation is that the number of observations-per-trial must be "sufficiently
% large" (whatever that means!)
%
% IMPORTANT NOTE: I haven't yet tested this in the arguably more realistic case
% where the VAR models are actually different, but the population GCs the same.
%
%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation parameters - override on command line

if ~exist('p',      'var'), p       = 7;           end % VAR model order
if ~exist('rho',    'var'), rho     = 0.9;         end % VAR spectral radius
if ~exist('cdw',    'var'), cdw     = 1;           end % VAR coefficients decay weighting factor
if ~exist('mii',    'var'), mii     = 2;           end % VAR residuals multi-information (generalised correlation)
if ~exist('regm',   'var'), regm    = 'OLS';       end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('nx',     'var'), nx      = 3;           end % number of target variables
if ~exist('ny',     'var'), ny      = 5;           end % number of source variables
if ~exist('nz',     'var'), nz      = 4;           end % number of conditioning variables
if ~exist('m',      'var'), m       = 100;         end % number of observations per trial
if ~exist('N',      'var'), N       = [90,120];    end % numbers of trials (can be different)
if ~exist('alpha',  'var'), alpha   = 0.05;        end % Significance level
if ~exist('hbins',  'var'), hbins   = 30;          end % histogram bins
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note: For this test to work, the number of observations-per-trial must be the
% SAME under both conditions (if the number of observations-per-trial is too small,
% though -- whatever "too small" means! -- the reduced regression is likely to fail
% with a DARE error. The number of trials in the respective conditions need not be
% the same; if you want, you may enter N as a 2-vector, e.g., N = [90,120].

if isscalar(N), N = [N,N]; end

rng_seed(seed);

n = nx+ny+nz;
x = 1:nx;
y = nx+1:nx+ny;

% Create a random VAR model

A = var_rand(n,p,rho,cdw);
V = corr_rand(n,mii);

% Calculate actual GC y -> x

Fa = var_to_mvgc(A,V,x,y);

% Generate two time series from (same!) model, corresponding to two "conditions",
% although the underlying generative process is actually THE SAME for both
% "conditions".

X = cell(2,1);
for c = 1:2 % conditions 1 and 2
	X{c} = var_to_tsdata(A,V,m,N(c));
end

% At this point we "throw away" the model, and pretend we don't know it; all we have
% are the two time series.

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate full-data estimated GCs y -> x. We cheat slightly, by using the (known)
% model order p; empirically, the model orders would be estimated in the usual way
% separately for each condition (since we may not be able to assume they will be
% the same under the two conditions).

% Full-data sample estimates

Fs = zeros(2,1);
for c = 1:2 % conditions 1 and 2
	[As,Vs] = tsdata_to_var(X{c},p,regm);
	Fs(c) = var_to_mvgc(As,Vs,x,y);
end

% Now we calculate the empirical single-trial distribution for each time series.
% Again, we cheat on the model orders, which should be the same as those selected
% for the full-data VAR model estimates.

Fb = cell(2,1);
fprintf('\n');
for c = 1:2 % conditions 1 and 2
	fprintf('Calculating single-trial empirical distribution (condition %d) ',c);
	[Fb{c},et] = single_trial_distribution(X{c},x,y,p,regm);
	fprintf(' %.2f seconds\n',et);
end

% Single-trial GC distribution medians and mean absolute deviations (mad)

Fbm = zeros(2,1);
Fbd = zeros(2,1);
for c = 1:2 % conditions 1 and 2
	Fbm(c) = median(Fb{c});
	Fbd(c) = mad(Fb{c},1);
end

% Summary statistics

fprintf('\n--------------------------------------------------\n');
fprintf('GC                      Condition 1    Condition 2\n');
fprintf('--------------------------------------------------\n');
fprintf('Actual                :   %6.4f         %6.4f     <--- same in both conditions!\n',Fa,Fa);
fprintf('Estimated             :   %6.4f         %6.4f\n', Fs(1), Fs(2) );
fprintf('Single-trial median   :   %6.4f         %6.4f\n', Fbm(1),Fbm(2));
fprintf('Single-trial mad      :   %6.4f         %6.4f\n', Fbd(1),Fbd(2));
fprintf('--------------------------------------------------\n');

% Plot histograms of empirical single-trial GC distributions

figure(fignum); clf;
histogram(Fb{1},hbins,'facecolor','g');
hold on
histogram(Fb{2},hbins,'facecolor','r');
hold off
title(sprintf('\nGC single-trial empirical distributions\n'));
xlabel('GC (green = Condition 1, red = Condition 2)')

% "Unpaired t-test" (Mann-Whitney test) between single-trial GC
% distributions.The null hypothesis is that neither distribution
% stochastically dominates the other (equivalently, that their medians
% are the same). A significant positive value means GC in Condition 2
% stochastically dominates (i.e. "is bigger than") GC in Condition 1.

z    = mann_whitney(Fb{1},Fb{2}); % z-score ~ N(0,1) under H0
pval = 2*(1-normcdf(abs(z)));     % p-value (2-tailed test)
sig  = pval < alpha;              % significant (reject H0)?

% Report inference result

fprintf('\n----------------------------------------\n');
fprintf('Stochastic dominance test (Mann-Whitney)\n');
fprintf('----------------------------------------\n');
if sig
	if z > 0, sigstr = 'YES (Condition 2 > Condition 1) - WRONG!';
	else,     sigstr = 'YES (Condition 1 > Condition 2) - WRONG!';
	end
else
	sigstr = 'NO - CORRECT!';
end
fprintf('z-score     :   %6.4f\n',z     );
fprintf('p-value     :   %g\n',   pval  );
fprintf('Significant :   %s\n',   sigstr);
fprintf('----------------------------------------\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [F,et] = single_trial_distribution(X,x,y,p,regm)

	% Single-trial GC empirical distribution
	%
	% X         multi-trial time-series data
	% x         target variable indices
	% y         source variable indices
	% p         VAR model order
	% regm      regression mode ('OLS' or 'LWR')
	%
	% F         single-trial GC estimates
	% et        elapsed time

	tic;
	N = size(X,3);
	F = zeros(N,1);
	N10 = round(N/10);
	for i = 1:N
		if rem(i,N10) == 0, fprintf('.'); end      % progress indicator
		[As,Vs] = tsdata_to_var(X(:,:,i),p,regm);  % estimated model based on single trial
		F(i)    = var_to_mvgc(As,Vs,x,y);          % single-trial GC estimate
	end
	et = toc; % elapsed time

end

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
