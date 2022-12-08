%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Compare single trial GC with permutation testing on simulated VAR
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
if ~exist('regm',   'var'), regmode = 'OLS';       end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('tstat',  'var'), tstat   = 'LR';        end % GC test statistic: F or LR (likelihood ratio)
if ~exist('debias', 'var'), debias  = true;        end % Debias GC statistics? (recommended for inference)
if ~exist('alpha',  'var'), alpha   = 0.05;        end % Significance level
if ~exist('Ns',      'var'), Ns       = 500;        end % Permutation sample sizes
if ~exist('hbins',  'var'), hbins   = 50;          end % histogram bins
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize
n = nx+ny+nz;
x = 1:nx;
y = nx+1:nx+ny;
stat = struct;
rng_seed(seed);
perm_F = cell(2,1);
single_F = cell(2,1); % single F
Fs = zeros(2,1); % estimated F
Fpm = zeros(2,1);
Fpd = zeros(2,1);
Fa  = zeros(2,1);

% Generate one ground truth connectivity matrix
connectivity_matrix = randi([0 1], n);
for i = 1:n
    connectivity_matrix(i,i) = 0;
end

% Random VAR model  (same model for both conditions)
A = var_rand(connectivity_matrix,p,rho(1),wvar(1));
V = corr_rand(n,rmi(1));

% Generate 2 time conditions
X1 = var_to_tsdata(A,V,m,N(1));
X2 = var_to_tsdata(A,V,m,N(2));

% Concatenate X1 and X2
Xcat = cat(3, X1,X2);

%% Calculate permutation GC for each condition

testStat = permutation_tsdata_to_mvgc(Xcat,x,y,'Ns',Ns,'morder',p);

% for c=1:2   
%     perm_F{c} = permutation_tsdata_to_mvgc(Xcat,x,y,'Ns',Ns,'morder',p);
%     % Permutation median and mad
%     Fpm(c) = median(perm_F{c});
%     Fpd(c) = mad(perm_F{c},1);
%     % Calculate actual GC
%     Fa(c) = var_to_mvgc(A,V,x,y);
% end

% Calculate estimated GC y -> x | z test statistic
VAR = ts_to_var_parameters(X1, 'morder', p, 'regmode', regmode);
Fs(1) = var_to_mvgc(VAR.A, VAR.V,x,y);
VAR = ts_to_var_parameters(X2, 'morder', p, 'regmode', regmode);
Fs(2) = var_to_mvgc(VAR.A, VAR.V,x,y);

% Compute p value and Z scores

%testStat = perm_F{1} - perm_F{2};
observedStat = Fs(1) - Fs(2);
count = 0;
for s=1:Ns
    if abs(testStat(s))>abs(observedStat)
        count=count+1;
    else
        continue 
    end
end

pval = count/Ns;
mT = mean(testStat);
sT = std(testStat);
perm_z = (observedStat - mT)/sT;
perm_sig  = pval <= alpha;             % significant (reject H0)?

%% Compute single trial GC

X = {X1, X2};

for c=1:2
    % Generate a random VAR model for condition 1
    single_F{c} = zeros(N(c),1);
    % Estimate single trial GC in each condition
    tic
    fprintf('Calculating single-trial empirical distribution (condition %d) ',c)
    for i=1:N(c)
        [A,V] = tsdata_to_var(X{c}(:,:,i),p,regmode);
        %var_info(A,V)
        single_F{c}(i) = var_to_mvgc(A,V,x,y);
    end
    fprintf(' %.2f seconds\n',toc);
end


% Compute statistics
single_z    = mann_whitney(single_F{1},single_F{2}); % z-score ~ N(0,1) under H0
pval = 2*(1-normcdf(abs(single_z)));     % p-value (2-tailed test)
single_sig  = pval < alpha;       % significant (reject H0)?

%% Create dataset to plot histogram in python

stat.estimated = Fs;
stat.permutation.t = testStat;
stat.permutation.obs = observedStat;   
stat.permutation.z = perm_z;
stat.permutation.sig = perm_sig;
stat.single_trial.t = single_F;
stat.single_trial.z = single_z;
stat.single_trial.sig = single_sig;
% Save statistics
datadir = fullfile('~', 'projects', 'cifar', 'results');
fname = 'edited_permtest_vs_single_trial.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'stat')
