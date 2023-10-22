function varmo = var_model_order(X,p,regmode,alpha,pacf,plotm,verb)

% plotm = []      - don't plot
% plotm = n       - Matlab plot to figure n (if zero, use next)
% plotm = string  - Gnuplot terminal (may be empty)

if nargin < 4 || isempty(alpha), alpha = [0.01 0.05]; end
if nargin < 5 || isempty(pacf),  pacf  = true;        end
if nargin < 6,                   plotm = [];          end
if nargin < 7 || isempty(verb),  verb  = 0;           end

if isempty(plotm), pacf = false; end % no point if we're not going to display

if isscalar(alpha), alpha = [alpha alpha]; end

n = size(X,1);

% get log-likelihoods, pacfs at VAR model orders 1 .. p

[LL,Lk,Lm,~,~] = tsdata_to_varll(X,p,regmode,pacf,verb>1); % log-likelihood, #free parameters. effective #observations

% calculate information criteria

[aic,bic,hqc] = infocrit(LL,Lk,Lm); % Akaike, Schwarz' Bayesian, Hannan-Quinn

% calculate optimal model orders according to information criteria (note: NaNs are ignored)

morder = (0:p)';
[~,idx] = min(aic); moaic = morder(idx);
[~,idx] = min(bic); mobic = morder(idx);
[~,idx] = min(hqc); mohqc = morder(idx);


if verb > 0
    fprintf('\nBest model orders\n');
    fprintf('-----------------\n\n');
    fprintf('AIC : %2d',moaic); if moaic == p, fprintf(' *'); end; fprintf('\n');
    fprintf('BIC : %2d',mobic); if mobic == p, fprintf(' *'); end; fprintf('\n');
    fprintf('HQC : %2d',mohqc); if mohqc == p, fprintf(' *'); end; fprintf('\n');
end

% Scale information criterion
gap = 0.05;
saic = gap+(1-gap)*(aic-min(aic))/(max(aic)-min(aic));
sbic = gap+(1-gap)*(bic-min(bic))/(max(bic)-min(bic));
shqc = gap+(1-gap)*(hqc-min(hqc))/(max(hqc)-min(hqc));

varmo.aic = moaic;
varmo.hqc = mohqc;
varmo.bic = mobic;

varmo.saic = saic;
varmo.shqc = shqc;
varmo.sbic = sbic;
lags = morder;
varmo.lags = lags;

end

