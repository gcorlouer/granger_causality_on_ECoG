%% Simulate time frequency GC

if ~exist('nx',     'var'), nx      = 3;           end % number of target variables
if ~exist('ny',     'var'), ny      = 5;           end % number of source variables
if ~exist('nz',     'var'), nz      = 2;           end % number of conditioning variables
if ~exist('p',      'var'), p       = 5;       end % model order
if ~exist('m',      'var'), m       = 2000;     end % numbers of observations per trial
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

%% Simulate time series
n = nx + ny +nz;
A = var_rand(n,p,rho,wvar);
V = corr_rand(n,rmi);

X = var_to_tsdata(A,V,m,N);

%% Estimate rolling mosvc
pf = 2 * p;
fs = 500;
shift = 40; % How much to shift window in observations
mw = 100; % Observations per window
tw = mw/fs; % Duration of time window
nwin = floor((m - mw)/shift +1);
mosvc = zeros(nwin,1);
fprintf('Rolling time window of size %4.2f \n', tw)
for w=1:nwin
    fprintf('Time window number %d over %d \n', w,nwin)
    o = (w-1)*shift;      % window offset
	W = X(:,o+1:o+mw,:,:);% the window
    mosvc(w) = tsdata_to_ssmo(W, pf , []);
end

ssmo_average = floor(mean(mosvc));
fprintf('SS average model oder is %d ', ssmo_average);

%% Estimate rolling group GC

x = 1; y=2;
nfreqs = 1024;
F = zeros(nwin,nfreqs);
fprintf('Rolling time window of size %4.2f \n', tw)
for w=1:nwin
    fprintf('Time window number %d over %d \n', w,nwin)
    o = (w-1)*shift;      % window offset
	W = X(:,o+1:o+mw,:,:);% the window
    [A,C,K,V,Z,E] = tsdata_to_ss(W, pf, ssmo_average);
    F(w,:) = ss_to_smvgc(A,C,K,V,x,y,nfreqs-1);
end

%% Time and freqs array

freqs = sfreqs(nfreqs,fs);
time = zeros(nwin,1);
for w=1:nwin
    time(w) = w * shift/fs;
end

%% Plot time frequency graph
figure
imagesc(F');

