%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Illustrates (within-subject) VAR GC inference between conditions
%
%
% Simulation parameters - override on command line

if ~exist('nx',     'var'), nx      = 3;           end % number of target variables
if ~exist('ny',     'var'), ny      = 5;           end % number of source variables
if ~exist('nz',     'var'), nz      = 2;           end % number of conditioning variables
if ~exist('p',      'var'), p       = 5;       end % model orders
if ~exist('m',      'var'), m       = 100;     end % numbers of observations per trial
if ~exist('N',      'var'), N       = [56,58];   end % numbers of trials
if ~exist('rho',    'var'), rho     = [0.9,0.95];  end % spectral radii
if ~exist('wvar',   'var'), wvar    = [0.9,0.7];   end % var coefficients decay weighting factors
if ~exist('rmi',    'var'), rmi     = [0.8,1.2];   end % residuals log-generalised correlations (multi-information):
if ~exist('regm',   'var'), regmode    = 'OLS';       end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('tstat',  'var'), tstat   = 'LR';        end % GC test statistic: F or LR (likelihood ratio)
if ~exist('debias', 'var'), debias  = true;        end % Debias GC statistics? (recommended for inference)
if ~exist('alpha',  'var'), alpha   = 0.05;        end % Significance level
if ~exist('Ns',      'var'), Ns     = 100;        end % Permutation sample sizes
if ~exist('hbins',  'var'), hbins   = 50;          end % histogram bins
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = nx+ny+nz;
x = 1:nx;
y = nx+1:nx+ny;

rng_seed(seed);

Fp = cell(2,1);
Fpm = zeros(2,1);
Fpd = zeros(2,1);
Fa  = zeros(2,1);
Fs  = zeros(2,1);


% Generate one ground truth connectivity matrix
connectivity_matrix = randi([0 1], n);
for i = 1:n
    connectivity_matrix(i,i) = 0;
end

% Random VAR model 
A = var_rand(connectivity_matrix,p,rho(1),wvar(1));
V = corr_rand(n,rmi(1));

X1 = var_to_tsdata(A,V,m,N(1));
X2 = var_to_tsdata(A,V,m,N(2));

% Concatenate X1 and X2
Xcat = cat(3, X1,X2);

%% Calculate permutation, actual and estimated GC for each condition


for c=1:2   
    Fp{c} = permutation_tsdata_to_mvgc(Xcat,x,y,'Ns',Ns,'morder',p);
    % Permutation median and mad
    Fpm(c) = median(Fp{c});
    Fpd(c) = mad(Fp{c},1);
    % Calculate actual GC
    Fa(c) = var_to_mvgc(A,V,x,y);
end

% Calculate estimated GC y -> x | z test statistic
VAR = ts_to_var_parameters(X1, 'morder', p, 'regmode', regmode);
Fs(1) = var_to_mvgc(VAR.A, VAR.V,x,y);
VAR = ts_to_var_parameters(X2, 'morder', p, 'regmode', regmode);
Fs(2) = var_to_mvgc(VAR.A, VAR.V,x,y);
%% Summary statistics 

fprintf('\n--------------------------------------------\n');
fprintf('Actual        : %6.4f         %6.4f\n', Fa(1),  Fa(2) );
fprintf('Estimated        : %6.4f         %6.4f\n', Fs(1),  Fs(2) );
fprintf('Permuted median : %6.4f         %6.4f\n', Fpm(1), Fpm(2));
fprintf('Permuted mad    : %6.4f         %6.4f\n', Fpd(1), Fpd(2));
fprintf('--------------------------------------------\n\n');

%% Compute p value and Z scores (with pvalue from permtest)

testStat = Fp{1} - Fp{2};
observedStat = Fs(1) - Fs(2);

count = abs(testStat) > abs(observedStat);
count = sum(count, 1);

pval = count/Ns;
mT = mean(testStat);
sT = std(testStat);
z = (observedStat - mT)/sT;

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

%% Test for gaussianity
t = (testStat - mT)/sT;
[kt,p,ksstat] = kstest(t);
q = skewness(testStat);
k = kurtosis(testStat);
% xn = randn(1000,1);
% kx = kurtosis(xn);
% [ksx,px,ksstat] = kstest(xn);
% 
%% Plot histograms of test statistics

% figure(fignum); clf;
% histogram(testStat,hbins,'facecolor','g');
% hold on
% xline(observedStat,'-','Observed statistic')
% hold off
% 
% title(sprintf('Permutation distribution of difference in GC between 2 conditions'));
