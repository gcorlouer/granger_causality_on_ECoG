%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise parameters for GC analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input data
if ~exist('subject', 'var'), subject = 'DiAs'; end
if ~exist('cohort','var'), cohort = {'AnRa', 'ArLa', 'DiAs'}; end
if ~exist('condition', 'var'), conditions = {'Rest', 'Face', 'Place'}; end
if ~exist('field', 'var'), field = {'time',  'condition', 'pair', 'subject','F'}; end
if ~exist('datadir', 'var'), datadir = fullfile('~', 'projects', 'cifar', 'results'); end
if ~exist('suffix', 'var'), suffix = '_condition_visual_ts.mat'; end


% Detrending
if ~exist('pdeg', 'var'), pdeg = 2; end % detrending degree
% vector of polynomial evaluation points (default: evenly spaced)
if ~exist('x', 'var'), x = []; end 
% normalise (temporal) variance of each variable to 1 (default: false)
if ~exist('normalise', 'var'), normalise = false; end 

% Mutual information
if ~exist('q', 'var'), q = 0; end % Covariance lag

% Modeling
if ~exist('regmode', 'var'), regmode = 'LWR'; end % OLS or LWR
if ~exist('morder', 'var'),    morder = 5; end % Model order. Pick 3 or 5.
if ~exist('ssmo', 'var'),    ssmo = 20; end % Model order for SS
if ~exist('momax', 'var'), momax = 20; end
if ~exist('pacf', 'var'), pacf = true; end
if ~exist('plotm', 'var'), plotm = []; end
if ~exist('verb', 'var'), verb = 0; end


% Spectral GC 
if ~exist('dim', 'var'), dim = 3; end % dimension of spectral integration
if ~exist('band', 'var'), band = [60 120]; end % band over which to integrate
if ~exist('sfreq', 'var'), sfreq = 250; end % sampling frequency
if ~exist('fres', 'var'), nfreqs = 1024; end % frequency bins 
if ~exist('conditional', 'var'), conditional = true; end % frequency bins 


% Rolling window
if ~exist('mw', 'var'), mw = 50; end % number of observations in window
if ~exist('shift', 'var'), shift = 10; end % window shift

% Connectivity 
if ~exist('connect', 'var'), connect = 'groupwise'; end


% Statistics
if ~exist('nsample', 'var'), nsample = 100; end
if ~exist('Ns', 'var'), Ns = 100; end % number of permutations
if ~exist('test', 'var'), test = 'F'; end
if ~exist('debias', 'var'), debias = true; end
if ~exist('alpha', 'var'), alpha = 0.05; end
if ~exist('mhtc', 'var'), mhtc = 'FDRD'; end % multiple testing correction
if ~exist('LR', 'var'), LR = true; end % If false F test

