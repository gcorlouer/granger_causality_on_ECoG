function gc_input = read_cdt_time_series(args)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read subject and returns input time series for gc analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
arguments
   args.datadir char = fullfile('~', 'projects', 'cifar', 'results'); % Result path
   args.subject char = 'DiAs'; % Subject to read
   args.suffix char = '_condition_visual_ts.mat'; % suffix filename
   args.condition char = 'Face'; % suffix filename
   args.pdeg double = 2; % detrending degree
end

datadir = args.datadir; subject = args.subject; suffix = args.suffix;
condition = args.condition; pdeg = args.pdeg;

fname = [subject suffix];
fpath = fullfile(datadir, fname);
% Meta data about time series
gc_input = load(fpath);
% Read conditions specific time series
X = gc_input.(condition);
% return gc input
gc_input.X = X;
end


