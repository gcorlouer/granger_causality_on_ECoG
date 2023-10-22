% In this script we do a power analysis to test when TD vs BU GC
% breaks down on perturbed VAR models

% Simulations parameters

if ~exist('p',      'var'), p       = 5;       end % model orders
if ~exist('ssmo',      'var'), ssmo       = 20;       end % model orders
if ~exist('m',      'var'), m       = 150;     end % numbers of observations per trial
if ~exist('N',      'var'), N       = 56;      end % numbers of trials
if ~exist('M',      'var'), M       = 500;      end % numbers of time series to draw
if ~exist('rho',    'var'), rho     = 0.9;       end % spectral radii
if ~exist('wvar',   'var'), wvar    = 0.9;   end % var coefficients decay weighting factors
if ~exist('perturbation', 'var'), perturbation = 100; end % perturbation

if ~exist('rmi',    'var'), rmi     = 0.8;   end % residuals log-generalised correlations (multi-information):
if ~exist('regm',   'var'), regmode    = 'LWR';       end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('debias', 'var'), debias  = true;        end % Debias GC statistics? (recommended for inference)
if ~exist('alpha',  'var'), alpha   = 0.05;        end % Significance level
if ~exist('Ns',      'var'), Ns       = 250;        end % Permutation sample sizes
if ~exist('sfreqs',      'var'), sfreq       = 250;        end % Sampling rate
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize 
input_parameters
rng_seed(seed);
n = 2; % Since we are only concerned with pairwise unconditonal GC, we only need 2 channels
%% Generate VAR model

A = var_rand(n,p, rho, wvar);
V = corr_rand(n,rmi);
%% Calculate true TD vs BU
    
F = var_to_pwcgc(A,V);
dFa = F(1,2) - F(2,1);
%% Simulate M time series
T = zeros(M,1);

for i=1:M
    fprintf("Sample %i over %i\n", i, M)
    X = var_to_tsdata(A,V,m,N);
    %X(:,1,:) = X(:,1,:) + perturbation * X(:,1,:); % perturbate
    
    %% Estimate TD vs BU (note we assume fix model order. Further analysis could estimate model order as well)

    ssmo = tsdata_to_ssmo(X, 2*p, []);
    [Ae,Ce,Ke,Ve,~,~] = tsdata_to_ss(X, 2*p, ssmo);
    Fe = ss_to_pwcgc(Ae,Ce,Ke,Ve);
    dFe = Fe(1,2) - Fe(2,1);
    %% Compute test statistic
    
    T(i) = dFe - dFa;
end

[pval, h, stats] = signrank(T); % A t-test could be reasonable here since each T(x_i) is i.i.d
fprintf("Estimated difference between TD and BU does NOT correspond to the truth: " + ...
    "%i, with pvalue %.4f", h, pval);