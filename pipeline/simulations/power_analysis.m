if ~exist('mMin', 'var'), mMin       = 20;       end % Min number of observations
if ~exist('mMax', 'var'), mMax       = 150;      end % Max number of observations
if ~exist('M', 'var'), M       = 100;      end % number of time series samples


if ~exist('signal',      'var'), signal       = 'hfa';       end % hfa or lfp
if ~exist('dF',      'var'), dF       = 0;       end % True difference
if ~exist('p',      'var'), p       = 5;       end % model order
if ~exist('ssmo',      'var'), ssmo       = 20;       end % SS model order
if ~exist('n',      'var'), n       = 2;       end % Number of channels
if ~exist('m',      'var'), m       = 50;     end % numbers of observations per trial
if ~exist('N',      'var'), N       = 20;      end % numbers of trials
if ~exist('rho',    'var'), rho     = 0.9;       end % spectral radii
if ~exist('wvar',   'var'), wvar    = 0.9;   end % var coefficients decay weighting factors
if ~exist('perturbation', 'var'), perturbation = 2; end % perturbation


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

%% Estimate GC on ECoG data

subject = 'DiAs';
fname = [subject '_condition_visual_' signal '.mat'];
fpath = fullfile(datadir, fname);
time_series = load(fpath);
X = time_series.('Face');
[~, m, N] = size(X);

pf = 2*morder;
[A,C,K,V,Z,E] = tsdata_to_ss(X,pf,ssmo);
F = ss_to_pwcgc(A,C,K,V);

%% Compute average top down
R_idx = time_series.indices.('R');
F_idx = time_series.indices.('F');
F_td = mean(F(R_idx,F_idx), 'all');
F_bu = 0;
%% Compute Statistical power by number trials
n = 2;
indices.R = 2;
indices.F = 1;
N0 = 1;
band = [0 120];
Nt = 56;% nummber of trials
noise_factor = 1;
power = zeros(Nt,1);
parray = zeros(M,1);
for j = Nt
    N = Nt;
    fprintf("Trial %i over %i\n", N, Nt)
    for i = 1:M
        fprintf("Sample %i over %i\n", i, M)
        % generate a random n-channel VAR(p)

			A = var_rand(n,p,rho);

			% generate an n-channel residuals correlation matrix

			V = corr_rand(n,rmi);

			% impose given BU and TD GC.

			A = var_adjust_mvgc(A,V,2,1,F_td);
            A = var_adjust_mvgc(A,V,1,2,F_bu); 

			% generate time-series data

			X = var_to_tsdata(A,V,m,N);

            % Compute TD vs BU GC
            stat = compare_TD_BU_pgc(X, indices, 'morder', morder, 'ssmo', ssmo,...
                        'Ns',Ns,'alpha',alpha, 'mhtc',mhtc, ...
                        'sfreq',sfreq, 'nfreqs', nfreqs,'dim',dim, 'band',band);
            parray(i) = stat.pval;
    end
    power = sum(parray < alpha)/M;
    fprintf("Statistical power is %4.f\n", power(j))
end
 
datadir = fullfile('~', 'projects', 'cifar', 'results');
fname = ['statistical_power_analysis_56_trials_' signal '.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'power')