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
varModel = struct;

%% Multitrial VAR model estimation
for c=1:ncdt
    condition = conditions{c};
    X = time_series.(condition);
    [nchan, nobs, ntrial] = size(X);
    subject = time_series.subject;
    %% Estimate VAR model.
    varModel.(subject).(condition) = varmo(X,momax,regmode,alpha);
    morder = 5;
    VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
    varModel.(subject).condition.('rho') = VAR.info.rho;
end
