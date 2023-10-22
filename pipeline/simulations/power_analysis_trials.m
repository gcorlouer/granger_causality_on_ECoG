if ~exist('mMin', 'var'), mMin       = 20;       end % Min number of observations
if ~exist('mMax', 'var'), mMax       = 150;      end % Max number of observations
if ~exist('M', 'var'), M       = 100;      end % number of time series samples


if ~exist('dF',      'var'), dF       = 0;       end % True difference
if ~exist('p',      'var'), p       = 5;       end % model order
if ~exist('ssmo',      'var'), ssmo       = 20;       end % SS model order
if ~exist('n',      'var'), n       = 2;       end % Number of channels
if ~exist('m',      'var'), m       = 50;     end % numbers of observations per trial
if ~exist('N',      'var'), N       = 20;      end % numbers of trials
if ~exist('rho',    'var'), rho     = 0.9;       end % spectral radii
if ~exist('wvar',   'var'), wvar    = 0.9;   end % var coefficients decay weighting factors
if ~exist('perturbation', 'var'), perturbation = 2; end % perturbation
if ~exist('noise_factor', 'var'), noise_factor = 1; end % perturbation


if ~exist('rmi',    'var'), rmi     = 0.8;   end % residuals log-generalised correlations (multi-information):
if ~exist('regm',   'var'), regmode    = 'LWR';       end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('debias', 'var'), debias  = true;        end % Debias GC statistics? (recommended for inference)
if ~exist('alpha',  'var'), alpha   = 0.05;        end % Significance level
if ~exist('Ns',      'var'), Ns       = 250;        end % Permutation sample sizes
if ~exist('sfreqs',      'var'), sfreq       = 250;        end % Sampling rate
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number
%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_parameters;
N0 = 2;
Nt = 50; % nummber of trials
noise_factor = 1;
step = 2;

indices.R = 1:1:floor(n/2);
indices.F = floor(n/2)+1:n;

% Get retonotopic and Face channels indices
R_idx = indices.('R');
F_idx = indices.('F');

nR = length(R_idx);
nF = length(F_idx);

C = eye(n); 
C(R_idx,F_idx) = ones(nR,nF);

A = var_rand(C,p,rho,wvar);
V = corr_rand(n,rmi);

%% Compute Statistical power by number trials
parray = zeros(nR,nF,M);
Nspace = (Nt - N0)/step + 1;
power = zeros(Nspace,1);
for j = 1:Nspace
    N = N0 + step*(j-1);
    for i = 1:M
        if mod(i,20) == 0
            fprintf("Sample %i over %i\n", i, M)
        end
        X = var_to_tsdata(A,V,m,N, noise_factor);
        % Compute TD vs BU GC
        stat = compare_TD_BU_pgc(X, indices, 'morder', morder, 'ssmo', ssmo,...
                    'Ns',Ns,'alpha',alpha, 'mhtc',mhtc, ...
                    'sfreq',sfreq, 'nfreqs', nfreqs,'dim',dim, 'band',band);
        parray(i) = stat.pval;
    end
    power(j) = sum(parray < alpha)/M;
    fprintf("Statistical power is %.4f\n", power(j))
end

datadir = fullfile('~', 'projects', 'cifar', 'results');
fname = ['statistical_power_analysis_' num2str(Nt) '_trials.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'power')