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
if ~exist('Ns',      'var'), Ns       = 100;        end % Permutation sample sizes
if ~exist('hbins',  'var'), hbins   = 50;          end % histogram bins
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


