%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialise parameters for GC analysis one a single subject
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TODO figure out better fres
% Input data
if ~exist('sub_id', 'var') sub_id = 'DiAs'; end

% Detrending
if ~exist('pdeg', 'var')    pdeg = 1; end % detrending degree
% vector of polynomial evaluation points (default: evenly spaced)
if ~exist('x', 'var')       x = []; end 
% normalise (temporal) variance of each variable to 1 (default: false)
if ~exist('normalise', 'var')    normalise = false; end 

% Mutual information
if ~exist('q', 'var')    q = 0; end % Covariance lag

% Modeling
if ~exist('regmode', 'var') regmode = 'OLS'; end % OLS or LWR
if ~exist('morder', 'var')    morder = 5; end % Model order. Pick 3 or 5.
if ~exist('momax', 'var') momax = 10; end
if ~exist('pacf', 'var') pacf = true; end
if ~exist('plotm', 'var') plotm = []; end
if ~exist('verb', 'var') verb = 0; end

% Sliding window
if ~exist('mw', 'var') mw = 80; end % number of observations in window
if ~exist('shift', 'var') shift = 10; end % window shift


% Statistics
if ~exist('nsample', 'var') nsample = 100; end
if ~exist('tstat', 'var') tstat = 'LR'; end
if ~exist('debias', 'var') debias = true; end

if ~exist('alpha', 'var') alpha = 0.05; end
if ~exist('mhtc', 'var') mhtc = 'FDRD'; end % multiple testing correction
if ~exist('LR', 'var') LR = true; end % If false F test

% Spectral gc
if ~exist('fres', 'var') fres = 1024; end