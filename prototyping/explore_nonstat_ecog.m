%% Parameters
subject_id = 'DiAs';
datadir = fullfile('~', 'projects', 'cifar', 'results');
suffix = '_condition_visual_ts.mat';
fname = [subject_id suffix];
fpath = fullfile(datadir, fname);
time_series = load(fpath);
condition = 'Face';
pdeg = 2;
%% Read condition specific time series
gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
    'condition',condition, 'suffix', suffix);

% Read conditions specific time series
X = gc_input.X;
indices = gc_input.indices;
time = gc_input.time;

%% Detrending

[X,~,~,~] = mvdetrend(X,pdeg,[]);
%% Plot 9 trials picked at random for individual face channels
ichan = indices.F(1);
rand_trials = randi(56, 9,1)';
trials = X(:,:,rand_trials);
ntrials = size(trials, 3);

for i=1:ntrials
    subplot(3,3,i)
    plot(time, trials(ichan,:,i))
end