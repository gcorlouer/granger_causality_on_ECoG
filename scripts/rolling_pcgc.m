input_parameters;
%% Load data
datadir = fullfile('~', 'projects', 'CIFAR', 'data', 'results');
fname = [sub_id '_condition_ts_visual.mat'];
fpath = fullfile(datadir, fname);

time_series = load(fpath);

X = time_series.data;
[nchan, nobs, ntrial, ncdt] = size(X);
sub_id = time_series.sub_id;
fs = double(time_series.sfreq);
findices = time_series.functional_indices;
ts_type = time_series.ts_type;
fn = fieldnames(findices);
time = time_series.time;

%% Detrend and demean
nwobs = 80;
nsobs = 5;
n_cdt = 3;
nwin = floor((nobs - nwobs)/nsobs +1);
F = zeros(nchan, nchan, nwin, ncdt);
nv = 2;
Fb = zeros(nv, nv, nwin, ncdt, nsample);
morder = 3;
iR = 3;
iF = 4;
vpop = [iR iF];
for w = 1:nwin
    o = (w-1)*nsobs;      % window offset
	W = X(:,o+1:o+nwobs,:,:);% the window
    for c=1:ncdt
        [W(:,:,:,c),~,~,~] = mvdetrend(W(:,:,:,c),pdeg,[]);
        W(:,:,:,c) = demean(W(:,:,:,c),normalise);
        % Compute VAR model order
        %[moaic(c,w),mobic(c,w),mohqc(c,w),molrt(c,w)] = tsdata_to_varmo(W(:,:,:,c), ... 
        %        momax,regmode,alpha,pacf,plotm,verb);
        % Compute GC
        F(:,:, w, c) = ts_to_var_pcgc(W(:,:,:,c), 'regmode',regmode, 'morder', morder);    
    end
end

%% Sample to time

win_size = nwobs/fs;
time_offset = nsobs/fs;
win_time = zeros(nwin,nwobs);
for w=1:nwin
    o = (w-1)*nsobs; 
    win_time(w,:) = time(o+1:o+nwobs);
end

%% Plot GC along sliding window
cdt = {'Rest', 'Face', 'Place'};
ymax = max(F, [], 'all');
for c=1:n_cdt
    subplot(ncdt,1,c)
    plot(win_time(:,nwobs), squeeze(F(iF, iR, :, c)), 'linewidth', 1.3)
    hold on
    plot(win_time(:,nwobs), squeeze(F(iR, iF, :, c)), 'linewidth', 1.3)
    ylim([0 ymax])
    xline(0, 'LineWidth',1, 'FontSize', 15)
    legend('F -> F', 'F -> F', cdt{c})
    ylabel(['GC'])
end
xlabel('Time (s)')
sgtitle(['Pairwise conditional GC between R and F over ' num2str(win_size) 's sliding window'], 'FontSize',18)
