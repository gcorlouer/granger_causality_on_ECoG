%% Estimate VAR and SS model on multitrial data
input_parameters;
conditions = {'Rest' 'Face' 'Place'};
ncdt = 3;
%% Read data

% Initialise information criterion
moaic =  cell(ncdt,1);
mobic =  cell(ncdt,1);
mohqc =  cell(ncdt,1);
molrt =  cell(ncdt,1);
mosvc =  cell(ncdt,1);
% Initialise spectral radius
rho = cell(ncdt,1);

%% Multitrial VAR model estimation
for c=1:ncdt
    plotm = c;
    condition = conditions{c};
    gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                    'condition',condition, 'suffix', suffix);
    sfreq = gc_input.sfreq;
    X = gc_input.(condition);
    [nchan, nobs, ntrial] = size(X);
    subject = gc_input.subject;
    findices = gc_input.indices;
    fn = fieldnames(findices);
    time = gc_input.time;
    % Detrend
    %[X,~,~,~] = mvdetrend(X,pdeg,[]);
    % Estimate var model order with multiple information criterion
    [moaic{c},mobic{c},mohqc{c},molrt{c}] = tsdata_to_varmo(X, ... 
                    momax,regmode,alpha,pacf,plotm,verb);
    %% Estimate VAR model.
    morder = 5;
    VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
    rho{c} = VAR.info.rho;
    
    %% Estimate SS model
    pf = 2*morder;
    [mosvc,rmax] = tsdata_to_ssmo(X,pf,plotm);
    [A,C,K,V,Z,E] = tsdata_to_ss(X,pf,mosvc);
end