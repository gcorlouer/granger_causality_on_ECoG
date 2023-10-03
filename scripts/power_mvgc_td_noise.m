% Power analysis with given GC.

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

%% Estimate GC on ECoG data

subject = 'DiAs';
fname = [subject '_condition_visual_' signal '.mat'];
fpath = fullfile(datadir, fname);
time_series = load(fpath);
X = time_series.('Face');
[n, m, N] = size(X);

pf = 2*morder;
[A,C,K,V,Z,E] = tsdata_to_ss(X,pf,ssmo);
F = ss_to_pwcgc(A,C,K,V);

%% Compute average top down
R_idx = time_series.indices.('R');
F_idx = time_series.indices.('F');
F_td = mean(F(R_idx,F_idx), 'all');

%%

gctest = 'F';
nn = 11;
NN = 56;
noise_scale = 1:0.1:4;
knoise = length(noise_scale);
power = zeros(knoise,1);
parray = zeros(M,1);
fprintf('\nObservations per trial = %d\n',m);

nnn = length(nn);
nNN = length(NN);
spow = zeros(knoise,1);

for k=1:knoise
    
    for i = 1:nnn % loop through channel numbers
    
	    n = nn(i);
	    fprintf('\nno. of channels (%2d of %2d) = %2d\n',i,nnn,n);
    
	    for j = 1:nNN % loop through trial numbers
    
		    N = NN(j);
		    fprintf('\tno. of trials (%2d of %2d) = %2d\n',j,nNN,N);
    
		    tpos = 0; % true positives
		    for s = 1:M % loop through samples
    
			    % generate a random n-channel VAR(p)
    
			    A = var_rand(n,p,rho);
    
			    % generate an n-channel residuals correlation matrix
    
			    V = corr_rand(n,rmi);
    
			    % impose given F(1 -> 2)
    
			    A = var_adjust_mvgc(A,V,2,1,F_td);
    
			    % generate time-series data
    
			    X = var_to_tsdata(A,V,m,N);

                % Add observational noise
                obs_noise = randn(n,m,N);
                pf = 2*morder;
                [r, ~] = tsdata_to_ssmo(X,pf,[]);
                X = X + noise_scale(1,k) * obs_noise;
                [A,Css,K,V,Z,E] = tsdata_to_ss(X,pf,r);
                [X,Z,E,mtrunc] = ss_to_tsdata(A,Css,K,V,m,N);

			    % test statistic for H0:  F(1 -> 2) = 0 (we use actual model order p)
    
			    tstat = var_to_mvgc_tstat(X,[],2,1,p,regmode,gctest); % NOTE: pairwise conditional!
    
			    % calculate p-values for test statistics
    
			    pval = mvgc_pval(tstat,gctest,1,1,n-2,p,m,N);
    
			    % accumulate true positives
    
			    tpos = tpos + (pval < alpha);
    
		    end
		    spow(k,1) = tpos/M;
    
	    end
    
    end
end

%%

datadir = fullfile('~', 'projects', 'cifar', 'results');
fname = ['power_analysis_td_gc_noise_' signal '.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'spow')

%%

% figure(1); clf
% plot(NN(:),spow);
% title(sprintf('Statistical power vs number of trials (F = %g, observations per trial = %d)\n',F_td,m));
% xlabel('number of trials');
% ylabel('statistical power');
% legend(num2str(nn',' #channels = %2d '),'location','southeast');
% grid on