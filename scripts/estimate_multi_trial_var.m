%% Estimate VAR model on multitrial data
input_parameters;
conditions = {'Rest' 'Face' 'Place'};
ncdt = 3;
%% Read data
subject = 'DiAs';
datadir = fullfile('~', 'projects', 'cifar', 'results');
fname = [subject '_condition_visual_ts.mat'];
fpath = fullfile(datadir, fname);
time_series = load(fpath);


% Initialise information criterion
moaic = cell(ncdt,1);
mobic =  cell(ncdt,1);
mohqc =  cell(ncdt,1);
molrt =  cell(ncdt,1);

% Initialise spectral radius
rho = cell(ncdt,1);

%% Multitrial VAR model estimation
for c=1:ncdt
    plotm = c;
    condition = conditions{c};
    sfreq = time_series.sfreq;
    X = time_series.(condition);
    [nchan, nobs, ntrial] = size(X);
    subject = time_series.subject;
    findices = time_series.indices;
    fn = fieldnames(findices);
    time = time_series.time;
    % Detrend
    %[X,~,~,~] = mvdetrend(X,pdeg,[]);
    % Estimate var model order with multiple information criterion
    [moaic{c},mobic{c},mohqc{c},molrt{c}] = tsdata_to_varmo(X, ... 
                    momax,regmode,alpha,pacf,plotm,verb);
    %% Estimate VAR model.
    morder =5;
    VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
    rho{c} = VAR.info.rho;
end

%% Single trial VAR
condition = 'Face';
plotm = [];
X = time_series.(condition);
[n, m, N] = size(X);
subject = time_series.subject;
findices = time_series.indices;
fn = fieldnames(findices);
time = time_series.time;

moaic = cell(N,1);
mobic =  cell(N,1);
mohqc =  cell(N,1);
molrt =  cell(N,1);

% Initialise spectral radius
rho = cell(N,1);
for i=1:N
    [moaic{i},mobic{i},mohqc{i},molrt{i}] = tsdata_to_varmo(X(:,:,i), ... 
                momax,regmode,alpha,pacf,plotm,verb);
    % Estimate VAR model.
    morder = 2;
    VAR = ts_to_var_parameters(X(:,:,i), 'morder', morder, 'regmode', regmode);
    rho{i} = VAR.info.rho;
end