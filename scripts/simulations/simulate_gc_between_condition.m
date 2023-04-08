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
if ~exist('sfreqs',      'var'), sfreq       = 250;        end % Permutation sample sizes
if ~exist('seed',   'var'), seed    = 0;           end % random seed (0 for unseeded)
if ~exist('fignum', 'var'), fignum  = 1;           end % figure number

%%%%%%%%%%%%%%%%%%%%%%%%%############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Import some real data for parameters
input_parameters
subject = 'DiAs';
condition = 'Face';
gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                    'condition',condition);
indices = gc_input.indices;
fn = fieldnames(indices);
ng = length(fn);
group = cell(ng,1);
for k=1:length(fn)
    group{k} = double(indices.(fn{k}));
end
group = group';
%% Initialize 
n = 5;
GC = struct;
rng_seed(seed);
A = cell(2,1);
V = cell(2,1);
F = cell(2,1);
X = cell(2,1);


for i=1:2
    % Generate two distinct connectivity matrices
    A{i} = var_rand(n,p, wvar);
    V{i} = corr_rand(n,rmi);
    % Generate 2 conditions specific time series
    X{i} = var_to_tsdata(A{i},V{i},m,N);
    % Estimate SS model
    pf = 2 * p;
    [model.A,model.C,model.K,model.V,~,~] = tsdata_to_ss(X{i}, pf, ssmo);
    % Compute ground truth
    
    F{i} = ss_to_GC(model, 'group', group, 'connect', connect,...
            'dim', 3, 'nfreqs', nfreqs, 'band',[0 125]);
end


%% Compute statistics
obsStat = F{1} - F{2};
% Concatenate for permutation testing
X = cat(3, X{1},X{2});
% Compute permutation test statistic
tstat = permute_condition_GC(X, 'connect', connect ,'group', group,...
    'Ntrial',N,'Ns', Ns, 'morder', morder, 'ssmo',ssmo, ...
    'sfreq',sfreq, 'nfreqs', nfreqs,'dim',dim, 'band',band);
% Permutation testing
stat = permtest(tstat, obsStat, 'Ns', Ns, ...
    'alpha', alpha, 'mhtc','FDRD');
%% Save into GC structure for python plotting
GC.('F1') = F{1};
GC.('F2') = F{2};
GC.('T') = stat.T;
GC.('z') = stat.z;
GC.('sig') = stat.sig;
GC.('pval') = stat.pval;
GC.('zcrit') = stat.zcrit;
GC.('band') = band;
GC.('connectivity') = connect;
fprintf('\n')

bandstr = mat2str(band);
fname = ['compare_condition_simulated_GC_' connect '_' bandstr 'Hz.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'GC')