%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Illustrates (within-subject) VAR GC inference between conditions
%
%
% Simulation parameters - override on command line


if ~exist('p',      'var'), p       = 5;       end % model orders
if ~exist('ssmo',      'var'), ssmo       = 20;       end % model orders
if ~exist('m',      'var'), m       = 100;     end % numbers of observations per trial
if ~exist('N',      'var'), N       = 20;      end % numbers of trials
if ~exist('rho',    'var'), rho     = 0.9;       end % spectral radii
if ~exist('wvar',   'var'), wvar    = 0.9;   end % var coefficients decay weighting factors
if ~exist('rmi',    'var'), rmi     = 0.8;   end % residuals log-generalised correlations (multi-information):
if ~exist('regm',   'var'), regmode    = 'OLS';       end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('debias', 'var'), debias  = true;        end % Debias GC statistics? (recommended for inference)
if ~exist('alpha',  'var'), alpha   = 0.05;        end % Significance level
if ~exist('Ns',      'var'), Ns       = 250;        end % Permutation sample sizes
if ~exist('sfreqs',      'var'), sfreq       = 250;        end % Sampling rate
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize 
input_parameters

connectivity_matrix = [[1;1],[0;1]];
[n, ~] = size(connectivity_matrix);
A = var_rand(connectivity_matrix,p,rho);
V = corr_rand(n,rmi);
GC = struct;
rng_seed(seed);
% Generate 2 conditions specific time series
X = var_to_tsdata(A,V,m,N);
indices.R = 1;
indices.F = 2;
stat = compare_TD_BU_pgc(X, indices, 'morder', morder, 'ssmo', ssmo,...
            'Ns',Ns,'alpha',alpha, 'mhtc',mhtc, ...
            'sfreq',sfreq, 'nfreqs', nfreqs,'dim',dim, 'band',band);
%% Save into GC structure for python plotting
GC.('z') = stat.z;
GC.('sig') = stat.sig;
GC.('pval') = stat.pval;
GC.('zcrit') = stat.zcrit;
GC.('band') = band;
GC.('connectivity') = connect;
fprintf('\n')

bandstr = mat2str(band);
fname = ['compare_ts_bu_simulated_GC_' connect '_' bandstr 'Hz.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'GC')