%input_parameters;
cdt = {'Rest' 'Face' 'Place'};
%% Read data
datadir = fullfile('~', 'projects', 'CIFAR', 'data', 'results');
fname = [sub_id '_condition_ts_visual.mat'];
fpath = fullfile(datadir, fname);

time_series = load(fpath);

X = time_series.data;
[n, m, N, ncdt] = size(X);
sub_id = time_series.sub_id;
fs = double(time_series.sfreq);
findices = time_series.functional_indices;
fn = fieldnames(findices);
time = time_series.time;
% Number of window

nwin = floor((m - mw)/shift +1);

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
        W(:,:,:,c) = demean(W(:,:,:,c),normalise);
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

%% Estimate Sliding mvgc
% gind = findices;
% gi = fieldnames(findices);
% ng = length(gi); % Number of groups
% nwin = floor((m - mw)/shift +1);
% F = zeros(ng, ng, nwin, ncdt);
% for c=1:ncdt
%     for w=1:nwin
%         o = (w-1)*shift;      % window offset
%         W = X(:,o+1:o+mw,:,c); % the window
%         [W,~,~,~] = mvdetrend(W,pdeg,[]);
%         VAR = ts_to_var_parameters(W, 'morder', morder, 'regmode', regmode);
%         F(:,:,w) = ts_to_mvgc(W, 'gind', findices,'morder', morder,...
%         'regmode', regmode);
%         for i=1:ng
%             for j=1:ng
%                 if i==j
%                     % Return causal density for diagonal elements
%                     % Compute pairwise conditional GC
%                     pF = var_to_pwcgc(VAR.A,VAR.V);
%                     % Return 0 when group of channel is singleton
%                     pF(isnan(pF))=0;
%                     x = gind.(gi{i});
%                     y = gind.(gi{j});
%                     F(i,j) = mean(pF(x,y),'all');
%                 else 
%                     % Return mvgc between group of populations
%                     x = gind.(gi{i});
%                     y = gind.(gi{j});
%                     nx = length(x);
%                     ny = length(y);
%                     nz = n - nx - ny;
%                     % Compute F(y->x|z) effect size 
%                     F(i,j) = var_to_mvgc(VAR.A,VAR.V,x,y);
%                 end
%             end
%         end
%     end
%     %[F(:,:,:,c), wtime] = ts_to_sliding_mvgc(X(:,:,:,c),'time', time,...
%     %   'gind', findices,'morder', morder,'regmode', regmode, 'mw', mw, 'shift', shift);
%     
% end

