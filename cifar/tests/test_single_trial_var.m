input_parameters;
datadir = fullfile('~', 'projects','CIFAR', 'data','results'); % For
%local machine
sub_id = 'DiAs';
ncdt = 3;
condition = {'rest', 'face', 'place'};
fname = [sub_id '_condition_ts_visual.mat']; 
fpath = fullfile(datadir, fname);
time_series = load(fpath);
for c=1:ncdt 
        X = time_series.(condition{c}); 
        % Functional group indices
        findices = time_series.functional_indices; fn = fieldnames(findices);
        [n, m, N] = size(X);
        nv = length(fn);
        for i=1:N
            [moaic(c,i),mobic(c,i),mohqc(c,i),molrt(c,i)] = tsdata_to_varmo(X(:,:,i), ... 
                    momax,regmode,alpha,pacf,[],verb);
        end
end
%% Estimate VAR model 
rho = zeros(ncdt, N);
for c=1:ncdt 
    X = time_series.(condition{c}); 
    for i=1:N
        VAR = ts_to_var_parameters(X(:,:,i), 'morder', 4, 'regmode', regmode);
        rho(c,i) = VAR.info.rho;
    end
end
%% Sliding window VAR model
shift = 20;
mw = 80;
nwin = floor((m - mw)/shift +1);
% detrend and demean data then estimate VAR model and estimate GC
moaic = zeros(ncdt, nwin, N);
mobic = zeros(ncdt, nwin, N);
mohqc = zeros(ncdt, nwin, N);
molrt = zeros(ncdt, nwin, N);

for c=1:ncdt
    X = time_series.(condition{c}); 
    for w=1:nwin
        o = (w-1)*shift;      % window offset
        W = X(:,o+1:o+mw,:);  % the window
        for i=1:N
        [moaic(c,w,i),mobic(c,w,i),mohqc(c,w,i),molrt(c,w,i)] = tsdata_to_varmo(W(:,:,i), ... 
                        momax,regmode,alpha,pacf,[],verb);
        end
    end
end

%%

rho = zeros(ncdt, nwin, N);
for c=1:ncdt
    X = time_series.(condition{c});
    for w=1:nwin
        o = (w-1)*shift;      % window offset
        W = X(:,o+1:o+mw,:,:);% the window
        for i=1:N
            VAR = ts_to_var_parameters(W(:,:,i), 'morder', morder, 'regmode', regmode);
            rho(c,w,i) = VAR.info.rho;
        end
    end
end

%% Count number of unstable trials
sfreq = time_series.sfreq;
sfreq = double(sfreq);
n_explode = sum(rho>1, 'all');
Ntrials = ncdt*nwin*N;
win_size = mw/sfreq;

fprintf('\n Explosive trials for %4.3fs window : %4.2f %% \n', ...
    win_size ,n_explode/Ntrials*100)

%% Estimate mvgc
regmode = 'LWR';
morder = 4;
fs = double(time_series.sfreq);
findices = time_series.functional_indices;
fn = fieldnames(findices);
time = time_series.time;
nv = length(fn);
nwin = floor((m - mw)/shift +1);
F = zeros(nv, nv, nwin,N);
for c=1:ncdt
    X = time_series.(condition{c});
    for i=1:N
        [F(:,:,:,i), wtime] = ts_to_sliding_mvgc(X(:,:,i),'time', time,...
            'gind', findices,'morder', morder,'regmode', regmode, 'mw', mw, 'shift', shift);
    end
end




            