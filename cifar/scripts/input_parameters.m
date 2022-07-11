%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise parameters for GC analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input data
if ~exist('subject', 'var'), subject = 'DiAs'; end
if ~exist('cohort','var'), cohort = {'AnRa', 'ArLa', 'DiAs'}; end
if ~exist('condition', 'var'), conditions = {'Rest', 'Face', 'Place', 'baseline'}; end
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

% VAR Modeling
if ~exist('regmode', 'var'), regmode = 'OLS'; end % OLS or LWR
if ~exist('morder', 'var'),    morder = 5; end % Model order. Pick 3 or 5.
if ~exist('momax', 'var'), momax = 20; end
if ~exist('pacf', 'var'), pacf = true; end
if ~exist('plotm', 'var'), plotm = []; end
if ~exist('verb', 'var'), verb = 0; end

% Spectral GC 
if ~exist('dim', 'var'), dim = 3; end % dimension of spectral integration
if ~exist('band', 'var'), band = [0 40]; end % band over which to integrate
if ~exist('sfreq', 'var'), sfreq = 250; end % sampling frequency
if ~exist('fres', 'var'), fres = 1024; end % frequency bins 


% Rolling window
if ~exist('mw', 'var'), mw = 80; end % number of observations in window
if ~exist('shift', 'var'), shift = 10; end % window shift


% Statistics
if ~exist('nsample', 'var'), nsample = 100; end
if ~exist('tstat', 'var'), tstat = 'LR'; end
if ~exist('debias', 'var'), debias = true; end
if ~exist('alpha', 'var'), alpha = 0.05; end
if ~exist('mhtc', 'var'), mhtc = 'FDRD'; end % multiple testing correction
if ~exist('LR', 'var'), LR = true; end % If false F test

