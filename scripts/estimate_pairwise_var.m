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

X = time_series.('Face');
[nchan, nobs, ntrial] = size(X);

% Initialise information criterion
moaic = cell(ncdt,nchan, nchan);
mobic =  cell(ncdt,nchan, nchan);
mohqc =  cell(ncdt,nchan, nchan);
molrt =  cell(ncdt,nchan, nchan);

% Initialise spectral radius
rho = cell(ncdt,nchan, nchan, nchan);

%% Multitrial pairwise VAR model estimation

for c=1:ncdt
    condition = conditions{c};
    sfreq = time_series.sfreq;
    X = time_series.(condition);
    [nchan, nobs, ntrial] = size(X);
    subject = time_series.subject;
    findices = time_series.indices;
    fn = fieldnames(findices);
    time = time_series.time;
    for i=1:nchan
        for j=1:nchan
            if i==j
                continue
            else
                x = X([i j],:,:);
                % Detrend
                [x,~,~,~] = mvdetrend(x,pdeg,[]);
                % Estimate var model order with multiple information criterion
                [moaic{c,i,j},mobic{c,i,j},mohqc{c,i,j},molrt{c,i,j}] = tsdata_to_varmo(x, ... 
                                momax,regmode,alpha,pacf,[],verb);
                morder = [mobic{c,i,j}, mohqc{c,i,j}];
                morder = mean(morder);
                %% Estimate VAR model.    
                VAR = ts_to_var_parameters(x, 'morder', mohqc{c,i,j}, 'regmode', regmode);
                rho{c,i,j} = VAR.info.rho;
            end
        end
    end
end