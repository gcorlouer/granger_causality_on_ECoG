input_parameters;
conditions = {'Rest' 'Face' 'Place'};
ncdt = 3;
dataset = struct;

%% Read data

datadir = fullfile('~', 'projects', 'cifar', 'results');
fname = [subject '_condition_bivariate_ts.mat'];
fpath = fullfile(datadir, fname);
time_series = load(fpath);

% Initialise information criterion
moaic = cell(ncdt,1);
mobic =  cell(ncdt,1);
mohqc =  cell(ncdt,1);
molrt =  cell(ncdt,1);

% Initialise spectral radius
rho = cell(ncdt,1);
%% 
for c=1:ncdt
    condition = conditions{c};
    sfreq = time_series.sfreq;
    X = time_series.(condition);
    [nchan, nobs, ntrial] = size(X);
    subject = time_series.subject;
    fn = fieldnames(findices);
    time = time_series.time;

    % Number of windows
    nwin = floor((nobs - mw)/shift +1);
    moaic{c} = zeros(nwin,1);
    mobic{c} = zeros(nwin,1);
    mohqc{c} = zeros(nwin,1);
    molrt{c} = zeros(nwin,1);
    %% VAR model order selection
    % detrend and demean data then estimate VAR model and estimate GC
    for w=1:nwin
        % window offset
        o = (w-1)*shift;   
        % the window
        W = X(:,o+1:o+mw,:); 
        % Detrend
        [W,~,~,~] = mvdetrend(W,pdeg,[]);
        % Estimate var model order with multiple information criterion
        [moaic{c}(w),mobic{c}(w),mohqc{c}(w),molrt{c}(w)] = tsdata_to_varmo(W, ... 
                    momax,regmode,alpha,pacf,[],verb);
    end

    %% Sample to time

    win_size = mw/sfreq;
    time_offset = shift/sfreq;
    win_time = zeros(nwin,mw);
    for w=1:nwin
        o = (w-1)*shift; 
        win_time(w,:) = time(o+1:o+mw);
    end

    %% Estimate VAR model.

    rho{c} = zeros(nwin,1);
    for w=1:nwin
        % window offset
        o = (w-1)*shift; 
        % the window
        W = X(:,o+1:o+mw,:);
        [W,~,~,~] = mvdetrend(W,pdeg,[]);
        VAR = ts_to_var_parameters(W, 'morder', morder, 'regmode', regmode);
        rho{c}(w) = VAR.info.rho;
    end
end

   
for c=1:ncdt
    condition = conditions{c};
    for w=1:nwin
        dataset(w,c).time = win_time(w,mw);
        dataset(w,c).condition = condition;
        dataset(w,c).subject = subject;
        dataset(w,c).aic = moaic{c}(w);
        dataset(w,c).bic = mobic{c}(w);
        dataset(w,c).hqc = mohqc{c}(w);
        dataset(w,c).lrt = molrt{c}(w);
        dataset(w,c).rho = rho{c}(w);
    end
end


lenData = numel(dataset);
dataset = reshape(dataset, lenData, 1);

%% Save dataset

df = struct2table(dataset);
fname = 'rolling_bivariate_var_estimation.csv';
fpath = fullfile(datadir, fname);
writetable(df, fpath)