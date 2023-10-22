function infoCrit = varmo(X,pmax,regmode,alpha)
% Note would be useful to make a pull request on github to ask Lionel
% to modify its function. 
% Calculate AIC, BIC and HQC for model order estimation (useful to plot)

if nargin < 4 || isempty(alpha), alpha = [0.01 0.05]; end

if isscalar(alpha), alpha = [alpha alpha]; end

n = size(X,1);

% get log-likelihoods, pacfs at VAR model orders 1 .. p

[LL,Lk,Lm,R,RM] = tsdata_to_varll(X,pmax,regmode,true,[]); % log-likelihood, #free parameters. effective #observations

% calculate information criteria
% Pb missing some callculation, see original function  
[aic,bic,hqc] = infocrit(LL,Lk,Lm); % Akaike, Schwarz' Bayesian, Hannan-Quinn

ccalpha = alpha(2)/(n*n*pmax);             % Bonferroni correction on significance levels
Rcrit = norminv(1-ccalpha/2)./sqrt(RM); % 2-sided test
R(:,:,1) = nan(n);                      % lag-zero are trivial, don't display
Rlim = 1.1*nanmax(abs(R(:)));           % for display

infoCrit.aic = aic;
infoCrit.bic = bic;
infoCrit.hqc = hqc;
infoCrit.pacf = R;
infoCrit.pacf_crit = Rcrit;
infoCrit.pacf_lim = Rlim;          

end

	