% Simulation parameters - override on command line

if ~exist('p',      'var'), p       = 5;           end % VAR model order
if ~exist('rho',    'var'), rho     = 0.9;         end % VAR spectral radius
if ~exist('cdw',    'var'), cdw     = 1;           end % VAR coefficients decay weighting factor
if ~exist('mii',    'var'), mii     = 2;           end % VAR residuals multi-information (generalised correlation)
if ~exist('regm',   'var'), regm    = 'OLS';       end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('nx',     'var'), nx      = 3;           end % number of target variables
if ~exist('ny',     'var'), ny      = 5;           end % number of source variables
if ~exist('nz',     'var'), nz      = 4;           end % number of conditioning variables
if ~exist('m',      'var'), m       = 100;         end % number of observations per trial
if ~exist('N',      'var'), N       = 100;    end % numbers of trials (can be different)
if ~exist('alpha',  'var'), alpha   = 0.05;        end % Significance level
if ~exist('hbins',  'var'), hbins   = 30;          end % histogram bins
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)


n = nx+ny+nz;
x = 1:nx;
y = nx+1:nx+ny;

% Generate one ground truth connectivity matrix
connectivity_matrix = randi([0 1], n);
for i = 1:n
    connectivity_matrix(i,i) = 0;
end
%% 
X = cell(2,1);
F = cell(2,1);
Fa = cell(2,1);

for c=1:2
    % Generate a random VAR model for condition 1
    A = var_rand(connectivity_matrix,p,rho,cdw);
    V = corr_rand(n,mii); 
    X{c} = var_to_tsdata(A,V,m,N);

    % Calculate actual GC y -> x
    Fa{c} = var_to_mvgc(A,V,x,y);
    F{c} = zeros(N,1);

    % Estimate single trial GC in each condition

    tic
    fprintf('Calculating single-trial empirical distribution (condition %d) ',c)
    for i=1:N
        [A,V] = tsdata_to_var(X{c}(:,:,i),p,regm);
        %var_info(A,V)
        F{c}(i) = var_to_mvgc(A,V,x,y);
    end
    fprintf(' %.2f seconds\n',toc);
end


%% Compute statistics
z    = mann_whitney(F{1},F{2}); % z-score ~ N(0,1) under H0
pval = 2*(1-normcdf(abs(z)));     % p-value (2-tailed test)
sig  = pval < alpha;              % significant (reject H0)?

% Report statistics
fprintf('\n----------------------------------------\n');
fprintf('Stochastic dominance test (Mann-Whitney)\n');
fprintf('----------------------------------------\n');
if sig
	if z > 0, sigstr = 'YES (Condition 2 > Condition 1) - WRONG!';
	else,     sigstr = 'YES (Condition 1 > Condition 2) - WRONG!';
	end
else
	sigstr = 'NO - CORRECT!';
end
fprintf('z-score     :   %6.4f\n',z     );
fprintf('p-value     :   %g\n',   pval  );
fprintf('Significant :   %s\n',   sigstr);
fprintf('----------------------------------------\n\n');

%% Plot histograms of empirical single-trial GC distributions

figure(fignum); clf;
histogram(F{1},hbins,'facecolor','g');
hold on
histogram(F{2},hbins,'facecolor','r');
hold off
title(sprintf('\nGC single-trial empirical distributions\n'));
xlabel('GC (green = Condition 1, red = Condition 2)')

