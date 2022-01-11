input_parameters;
cdt = {'Rest' 'Face' 'Place'};
%% Read data
datadir = fullfile('~', 'projects', 'CIFAR', 'data', 'results');
fname = [sub_id '_condition_ts_visual.mat'];
fpath = fullfile(datadir, fname);

time_series = load(fpath);

X = time_series.data;
[nchan, nobs, ntrial, ncdt] = size(X);
sub_id = time_series.sub_id;
fs = double(time_series.sfreq);
findices = time_series.functional_indices;
fn = fieldnames(findices);
time = time_series.time;

% Number of window

nwin = floor((nobs - mw)/shift +1);
%% VAR model order selection
% detrend and demean data then estimate VAR model and estimate GC
moaic = zeros(ncdt, nwin);
mobic = zeros(ncdt, nwin);
mohqc = zeros(ncdt, nwin);
molrt = zeros(ncdt, nwin);

for w=1:nwin
    o = (w-1)*shift;      % window offset
	W = X(:,o+1:o+mw,:,:);% the window
    for c=1:ncdt
        [W(:,:,:,c),~,~,~] = mvdetrend(W(:,:,:,c),pdeg,[]);
        W(:,:,:,c) = demean(W(:,:,:,c),normalise);
        [moaic(c,w),mobic(c,w),mohqc(c,w),molrt(c,w)] = tsdata_to_varmo(W(:,:,:,c), ... 
                    momax,regmode,alpha,pacf,[],verb);
                %VAR(c,w) = ts_to_var_parameters(W(:,:,:,c), 'morder', molrt(c,w), 'regmode', regmode);
                %disp(VAR.info);
    end
end

%% Sample to time

win_size = mw/fs;
time_offset = shift/fs;
win_time = zeros(nwin,mw);
for w=1:nwin
    o = (w-1)*shift; 
    win_time(w,:) = time(o+1:o+mw);
end

%% Estimate VAR model.

rho = zeros(ncdt, nwin);
for w=1:nwin
    o = (w-1)*shift;      % window offset
	W = X(:,o+1:o+mw,:,:);% the window
    for c=1:ncdt
        [W(:,:,:,c),~,~,~] = mvdetrend(W(:,:,:,c),pdeg,[]);
        %W(:,:,:,c) = demean(W(:,:,:,c),normalise);
        VAR = ts_to_var_parameters(W(:,:,:,c), 'morder', morder, 'regmode', regmode);
        rho(c,w) = VAR.info.rho;
    end
end

% Plot spectral radius along sliding window

for c=1:ncdt
    subplot(ncdt,1,c)
    plot(win_time(:,mw), rho(c,:));
    ylim([0.8 1])
    xlabel(['Time(s) ' cdt{c}])
    ylabel('Spectral radius')
end

clear VAR;

%% Deprecated 

%% Plot model orders

% for c=1:ncdt
%     subplot(ncdt,1,c)
%     plot(win_time(:,mw), moaic(c,:), 'DisplayName', 'aic');
%     ylim([0 10])
%     hold on
%     plot(win_time(:,mw), mobic(c,:), 'DisplayName', 'bic');
%     ylim([0 10])
%     hold on
%     plot(win_time(:,mw), mohqc(c,:), 'DisplayName', 'mohqc')
%     ylim([0 10])
%     hold on
%     plot(win_time(:,mw), molrt(c,:), 'DisplayName', 'molrt')
%     legend
%     ylim([0 10])
%     xlabel(['Time(s) ' cdt{c}])
%     ylabel('model order')
% end
