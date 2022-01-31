cohort = {'AnRa',  'ArLa', 'DiAs'};
condition = {'Rest', 'Face', 'Place', 'baseline'};
%conditionb = {'baseline'};
field = {'time',  'condition', 'pair', 'subject','F'};
dataset = struct;
mw = 40;
nb= 10; % Number of window sampled within baseline
input_parameters;
nsub = length(cohort);
ncdt = length(condition);
%% Read data

for s=1:nsub
    % Read data
    datadir = fullfile('~', 'projects', 'cifar', 'results');
    subject = cohort{s};
    fname = [subject '_condition_ts.mat']; 
    fpath = fullfile(datadir, fname);
    time_series = load(fpath);
    
    % The time series
    for c=1:ncdt 
        X = time_series.(condition{c}); subject = time_series.subject; 
        time = time_series.time; fs = double(time_series.sfreq);

        % Functional group indices
        indices = time_series.indices; fn = fieldnames(indices);
        [n, m, N] = size(X);

        % Read baseline data
        %Xb = time_series.(conditionb);
        %timeb = time_series.timeb;

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
        'gind', indices,'mw',mw, 'morder', morder, 'regmode', regmode);

        % Estimate baselibe mvgc
        %[Fb, wtimeb] = ts_to_sliding_mvgc(Xb,'time', time, 'shift', shift, ... 
        %'gind', indices,'mw',mw, 'morder', morder, 'regmode', regmode);
        % Return average of baseline mvgc
        %Fb = mean(Fb,3);
        %% Create dataset
        ng = length(fn);
        for w=1:nwin
            for i=1:ng
                for j=1:ng   
                    dataset(w,i,j,c,s).time = win_time(w,mw);
                    dataset(w,i,j,c,s).pair =  [fn{j} '->' fn{i}];
                    dataset(w,i,j,c,s).condition = condition{c};
                    dataset(w,i,j,c,s).subject = subject;
                    dataset(w,i,j,c,s).F = F(i,j,w);
                    %dataset(w,i,j,c,s).Fb = Fb(i,j);
                end
            end
        end
    end
end
lenData = numel(dataset);
dataset = reshape(dataset, lenData, 1);

%% Save dataset

df = struct2table(dataset);
fname = 'rolling_mvgc.csv';
fpath = fullfile(datadir, fname);
writetable(df, fpath)