%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Calculate and compare empirical and multi-trial bootstrap VAR GC estimates
%
%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Calulate
%

% Simulation parameters - override on command line

if ~exist('nx',     'var'), nx      = 3;       end % number of target variables
if ~exist('ny',     'var'), ny      = 5;       end % number of source variables
if ~exist('nz',     'var'), nz      = 2;       end % number of conditioning variables
if ~exist('p',      'var'), p       = 6;       end % model order
if ~exist('m',      'var'), m       = 50;      end % number of observations per trial
if ~exist('N',      'var'), N       = 100;     end % number of trials
if ~exist('rho',    'var'), rho     = 0.95;    end % spectral radius
if ~exist('wvar',   'var'), wvar    = 1;       end % var coefficients decay weighting factor
if ~exist('rmi',    'var'), rmi     = 1;       end % residuals log-generalised correlation (multi-information):
if ~exist('regm',   'var'), regm    = 'OLS';   end % VAR model estimation regression mode ('OLS' or 'LWR')
if ~exist('S',      'var'), S       = 1000;    end % sample size
if ~exist('seed',   'var'), seed    = 0;       end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;       end % figure number

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = nx+ny+nz;
x = 1:nx;
y = nx+1:nx+ny;

rng_seed(seed);

% Random VAR model

AA = var_rand(n,p,rho,wvar);
VV = corr_rand(n,rmi);

% The data

XX = var_to_tsdata(AA,VV,m,N);

% Calculate actual GC y -> x | z

FF = var_to_mvgc(AA,VV,x,y);

% GC sampling distribution (can't do this with actual data!)

fprintf('\nCalculating sample distribution ');
[Fsam,et] = mvgc_var_sample(AA,VV,x,y,m,N,S,regm);
fprintf(' %.2f seconds\n',et);

% GC bootstrap distribution (CAN do this with actual data!)

fprintf('\nCalculating bootstrap distribution '); tic;
[Fbst,et] = mvgc_var_bootstrap(XX,x,y,p,S,regm);
fprintf(' %.2f seconds\n',et);

% Plot empirical sampling and bootstrap CDFs

Fsamm = mean(Fsam);
Fbstm = mean(Fbst);

[Psam,Fsam1] = ecdf(Fsam);
[Pbst,Fbst1] = ecdf(Fbst);

figure(fignum); clf
plot(Fsam1,Psam,Fbst1,Pbst);
title(sprintf('Sample and bootstrap empirical CDFs\n\n(NOTE: sample and bootstrap means are biased!)\n'));
xlabel('Granger causality');
ylabel('Probability');
legend('sample','bootstrap','location','southeast');
xline(FF,   '-','actual',   'color','k','HandleVisibility','off');
xline(Fsamm,'-','sample',   'color','b','HandleVisibility','off');
xline(Fbstm,'-','bootstrap','color','r','HandleVisibility','off');
