cohort = {'AnRa',  'ArLa',  'BeFe',  'DiAs',  'JuRo'};
condition = {'rest', 'face', 'place'};
conditionb = {'rest_baseline', 'face_baseline', 'place_baseline'};
field = {'time',  'condition', 'pair', 'subject','F'};
dataset = struct;
nb= 10; % Number of window sampled within baseline
input_parameters;
nsub = length(cohort);
ncdt = length(condition);
%% Read data

for s=1:nsub
    % Read data
    datadir = fullfile('~', 'projects', 'CIFAR', 'data', 'results');
    sub_id = cohort{s};
    fname = [sub_id '_condition_ts_visual.mat']; 
    fpath = fullfile(datadir, fname);
    time_series = load(fpath);
    
    % The time series
    for c=1:ncdt 
        X = time_series.(condition{c}); sub_id = time_series.sub_id; 
        time = time_series.time; fs = double(time_series.sfreq);

        % Functional group indices
        findices = time_series.functional_indices; fn = fieldnames(findices);
        [n, m, N] = size(X);

        % Read baseline data
        Xb = time_series.(conditionb{c});
        timeb = time_series.timeb;

        %% Window sample to time

        nwin = floor((m - mw)/shift +1);
        win_size = mw/fs;
        time_offset = shift/fs;
        win_time = zeros(nwin,mw);
        for w=1:nwin
            o = (w-1)*shift; 
            win_time(w,:) = time(o+1:o+mw);
        end
        %% Estimate mvgc
        [F, wtime] = ts_to_sliding_mvgc(X,'time', time, 'shift', shift, ... 
        'gind', findices,'mw',mw, 'morder', morder, 'regmode', regmode);

        % Estimate baselibe mvgc
        [Fb, wtimeb] = ts_to_sliding_mvgc(Xb,'time', time, 'shift', shift, ... 
        'gind', findices,'mw',mw, 'morder', morder, 'regmode', regmode);
        % Return average of baseline mvgc
        Fb = mean(Fb,3);
        %% Create dataset
        ng = length(fn);
        for w=1:nwin
            for i=1:ng
                for j=1:ng   
                    dataset(w,i,j,c,s).time = win_time(w,mw);
                    dataset(w,i,j,c,s).pair =  [fn{j} '->' fn{i}];
                    dataset(w,i,j,c,s).condition = condition{c};
                    dataset(w,i,j,c,s).subject = sub_id;
                    dataset(w,i,j,c,s).F = F(i,j,w);
                    dataset(w,i,j,c,s).Fb = Fb(i,j);
                end
            end
        end
    end
end
lenData = numel(dataset);
dataset = reshape(dataset, lenData, 1);

%% Save dataset

df = struct2table(dataset);
fname = 'cross_sliding_mvgc_test.csv';
fpath = fullfile(datadir, fname);
writetable(df, fpath)