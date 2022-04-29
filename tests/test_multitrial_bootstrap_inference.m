%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Illustrates (within-subject) VAR GC inference between conditions
%
% Note that we use the test statistics (F or likelihood ratio) for statistical
% inference. The only reason for this (as opposed to using the more accurate
% single-regression estimator) is that we can (at least approximately) de-bias
% the estimates. This is important if the VAR parameters, and/or number of
% observations/trials, varies between the conditions, as it may well do in
% practice: while bias does not effect null-hypothesis statistical significance
% testing, it may well effect non-null (nonparametric) t-tests.
%
% Statistical inference uses a Mann-Whitney U-test (a non-parametric unpaired
% t-test) for "statistical dominance".
%
%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulation parameters - override on command line

if ~exist('nx',     'var'), nx      = 3;           end % number of target variables
if ~exist('ny',     'var'), ny      = 5;           end % number of source variables
if ~exist('nz',     'var'), nz      = 2;           end % number of conditioning variables
if ~exist('p',      'var'), p       = [4,6];       end % model orders
if ~exist('m',      'var'), m       = [50,60];     end % numbers of observations per trial
if ~exist('N',      'var'), N       = [56,56];   end % numbers of trials
if ~exist('rho',    'var'), rho     = [0.9,0.95];  end % spectral radii
if ~exist('wvar',   'var'), wvar    = [0.9,0.7];   end % var coefficients decay weighting factors
if ~exist('rmi',    'var'), rmi     = [0.8,1.2];   end % residuals log-generalised correlations (multi-information):
if ~exist('regm',   'var'), regm    = 'OLS';       end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('tstat',  'var'), tstat   = 'LR';        end % GC test statistic: F or LR (likelihood ratio)
if ~exist('debias', 'var'), debias  = true;        end % Debias GC statistics? (recommended for inference)
if ~exist('alpha',  'var'), alpha   = 0.05;        end % Significance level
if ~exist('S',      'var'), S       = [1000,1000];  end % bootdstrap sample sizes
if ~exist('hbins',  'var'), hbins   = 50;          end % histogram bins
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = nx+ny+nz;
x = 1:nx;
y = nx+1:nx+ny;

rng_seed(seed);

Fb  = cell(2,1);
FF  = zeros(2,1);
Fs  = zeros(2,1);
Fbm = zeros(2,1);
Fbd = zeros(2,1);
ab  = zeros(2,1);

for c = 1:2 % conditions 1 and 2

	% Random VAR model

	AA = var_rand(n,p(c),rho(c),wvar(c));
	VV = corr_rand(n,rmi(c));

	% The data

	XX = var_to_tsdata(AA,VV,m(c),N(c));

	% Calculate estimated GC y -> x | z test statistic

	Fs(c) = var_to_mvgc_tstat(XX,[],x,y,p(c),regm,tstat);

	% GC multi-trial bootstrap distribution

	fprintf('\nCalculating multi-trial bootstrap distribution (condition %d) ',c);
	[Fb{c},et] = mvgc_var_bootstrap_tstat(XX,x,y,p(c),S(c),regm,tstat);
	fprintf(' %.2f seconds\n',et);

	% GC bootstrap median and median absolute deviation

	Fbm(c) = median(Fb{c});
	Fbd(c) = mad(Fb{c},1);

	if debias

		% Debias (approximately) statistics - recommended if VAR parameters and/or
		% numbers of observations/trials differ between the two conditions

		ab(c) = mvgc_bias(tstat,nx,ny,nz,p(c),m(c),N(c)); % approximate bias

		Fs     = Fs     - ab(c);
		Fb{c}  = Fb{c}  - ab(c);
		Fbm(c) = Fbm(c) - ab(c);

	end
end

% Summary statistics (Note: you can use the median absolute deviations in Fbd
% as you would standard deviation, for error bars, etc.)

fprintf('\n--------------------------------------------\n');
if debias
	fprintf('GC (de-biased)    Condition 1    Condition 2\n');
else
	fprintf('GC (biased)       Condition 1    Condition 2\n');
end
fprintf('--------------------------------------------\n');
fprintf('Estimated        : %6.4f         %6.4f\n', Fs(1),  Fs(2) );
fprintf('Bootstrap median : %6.4f         %6.4f\n', Fbm(1), Fbm(2));
fprintf('Bootstrap mad    : %6.4f         %6.4f\n', Fbd(1), Fbd(2));
fprintf('Approx. bias     : %6.4f         %6.4f\n', ab(1),  ab(2) );
fprintf('--------------------------------------------\n\n');

% Unpaired "t-test" (Mann-Whitney U-test) between bootstrap samples.
% Null hypothesis is that neither group stochastically dominates the
% other. A significant positive value means GC in Condition 2
% stochastically dominates (i.e. is "bigger than") GC in Condition 1.
%
% Note: if you are doing this for multiple GCs, you should use a multiple-
% hypothesis adjustment; see MVGC2 routine 'significance'.

z    = mann_whitney(Fb{1},Fb{2}); % z-score ~ N(0,1) under H0
pval = erfc(abs(z)/sqrt(2));      % p-value (2-tailed test)
sig  = pval <= alpha;             % significant (reject H0)?

if sig
	if z > 0
		sigstr = 'YES (Condition 2 > Condition 1)';
	else
		sigstr = 'YES (Condition 1 > Condition 2)';
	end
else
	sigstr = 'NO';
end

fprintf('z-score     : %6.4f\n',z);
fprintf('p-value     : %6.4f\n',pval);
fprintf('Significant : %s\n\n', sigstr);

% Plot histograms of bootstrap distributions

figure(fignum); clf;
histogram(Fb{1},hbins,'facecolor','g');
hold on
histogram(Fb{2},hbins,'facecolor','r');
hold off
if debias
	title(sprintf('Bootstrap distributions (de-biased)\n'));
else
	title(sprintf('Bootstrap distributions (biased)\n'));
end
xlabel('GC (green = Condition 1, red = Condition 2)')
