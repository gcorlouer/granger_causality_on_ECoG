function [VAR, moest] = VARmodeling(X, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate VAR model order and VAR parameters from input time series
%%%%%%% 
% Input
%%%%%%%
% - X: n x m x N time series
% - regmode: str
%            Regression mode 'OLS' or 'LWR'
% - mosel:   int
%            Model order selection  1 - AIC, 2 - BIC, 3 - HQC, 4 - LRT
% - momax:   int
%            maximum model order 
% - plotm:   int
%            plot model order esmation
%%%%%%%%
% Output
%%%%%%%%
% - VAR:   Struct; VAR parameters
% - moest: int; estimated model order
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
defaultMosel = 1;
defaultMomax = 15;
defaultregmode = 'OLS';
defaultPlotm = 1;

p = inputParser;

addRequired(p,'X');
addParameter(p, 'mosel', defaultMosel, @isscalar); % selected model order:
addParameter(p, 'momax', defaultMomax, @isscalar);
addParameter(p, 'regmode', defaultregmode);  
addParameter(p, 'plotm', defaultPlotm, @isscalar);  

parse(p, X, varargin{:});

% VAR model order estimation
[moest(1),moest(2), moest(3), moest(4)] = ... 
    tsdata_to_varmo(p.Results.X, p.Results.momax,p.Results.regmode, ...
    [], [], p.Results.plotm);
% VAR modeling
[VAR.A, VAR.V, VAR.E] = tsdata_to_var(p.Results.X, ...
    moest(p.Results.mosel),p.Results.regmode); 
% Spectral radius
VAR.info = var_info(VAR.A,VAR.V);

end