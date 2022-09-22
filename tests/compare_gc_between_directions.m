%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Test GC difference between directions with permutation testing
%
%
% Simulation parameters - override on command line

if ~exist('nx',     'var'), nx      = 3;           end % number of target variables
if ~exist('ny',     'var'), ny      = 5;           end % number of source variables
if ~exist('nz',     'var'), nz      = 2;           end % number of conditioning variables
if ~exist('p',      'var'), p       = 5;       end % model orders
if ~exist('m',      'var'), m       = 100;     end % numbers of observations per trial
if ~exist('N',      'var'), N       = 56;   end % numbers of trials
if ~exist('rho',    'var'), rho     = 0.9;  end % spectral radii
if ~exist('wvar',   'var'), wvar    = 0.9;   end % var coefficients decay weighting factors
if ~exist('rmi',    'var'), rmi     = 0.8;   end % residuals log-generalised correlations (multi-information):
if ~exist('regm',   'var'), regmode    = 'LWR';       end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('tstat',  'var'), tstat   = 'LR';        end % GC test statistic: F or LR (likelihood ratio)
if ~exist('alpha',  'var'), alpha   = 0.05;        end % Significance level
if ~exist('S',      'var'), Ns       = 1000;        end % Permutation sample sizes
if ~exist('hbins',  'var'), hbins   = 50;          end % histogram bins
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 2;

rng_seed(seed);

%% Simulare time series

% Generate one ground truth connectivity matrix
connectivity_matrix = [[0;1],[1;0]];

% Random VAR model 
A = var_rand(1,p,rho,wvar);
V = corr_rand(n);
V = V(1,2);
% Simulate time series 
% Same VAR model for each time series
x1 = var_to_tsdata(A,V,m,N);
x2 = var_to_tsdata(A,V,m,N);
X = cat(1,x1,x2);
%% Compute actual and estimate pairwise GC
% Estimate GC
%VAR = ts_to_var_parameters(Xp, 'morder', p, 'regmode', regmode);
%Fa = var_to_pwcgc(VAR.A, VAR.V);
% Actual differnce
%Ta = Fa(1,2)-Fa(2,1); 
% Estimate GC
VAR = ts_to_var_parameters(X, 'morder', p, 'regmode', regmode);
Fest = var_to_pwcgc(VAR.A,VAR.V);
% Estimated difference
Test = Fest(1,2) - Fest(2,1);
%% Compute permutation statistic

% Concatenate channels into one multitrial time series

Xc = cat(3, X(1,:,:), X(2,:,:));

% Permute channels

[n,m,Nt] = size(Xc);

T = zeros(N,1);

for s=1:Ns
    fprintf('MVGC: permutation sample %d of %d',s,Ns);
    % Permut trial index
    trials = randperm(Nt);
    trials1 = trials(1:N);
    trials2 = trials(N+1:Nt);
    x1 = Xc(:,:,trials1);
    x2 = Xc(:,:,trials2);
    % Concatenate pairwise time series
    Xp = cat(1,x1,x2);
    % Estimate permutation GC
    VAR = ts_to_var_parameters(Xp, 'morder', p, 'regmode', regmode);
    Fp = var_to_pwcgc(VAR.A, VAR.V);
    % Compute permutation statistic
    T(s) = Fp(1,2)-Fp(2,1);
    fprintf('\n');
end

%% Compute p value and Z scores

count = 0;
for s=1:Ns
    if abs(T(s))>abs(Test)
        count=count+1;
    else
        continue 
    end
end

pval = count/Ns;
mT = mean(T);
sT = std(T);
z = (Test - mT)/sT;

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

%% Plot histograms of test statistics

figure(fignum); clf;
histogram(T,hbins,'facecolor','g');
hold on
xline(Test,'-','Observed statistic')
hold off

title(sprintf('Permutation distribution of TD vs BU GC'));








