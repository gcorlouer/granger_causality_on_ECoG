function VAR = ts_to_var_parameters(X, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function returns VAR parameters from time 
% series, model order estimate and regression
%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%
% - X :         n x m x N array 
%               Input time series 
% - morder:     int 
%               Estimated model order
% - regmode:    str
%               regression mode ('OLS' or 'LWR')
% Outputs
%%%%%%%%%%
% - VAR: Structure
%        The estimated VAR paremeters: coefficients,
%        variance and residual matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

defaultMorder = 5;
defaultRegmode = 'OLS';

p = inputParser;

addRequired(p,'X');
addParameter(p, 'regmode', defaultRegmode);  
addParameter(p, 'morder', defaultMorder);  

parse(p, X, varargin{:});

X = p.Results.X;
morder = p.Results.morder;
regmode = p.Results.regmode;

% VAR modeling
[VAR.A, VAR.V, VAR.E] = tsdata_to_var(X, ...
morder,regmode); 
% Spectral radius
VAR.info = var_info(VAR.A,VAR.V);

end