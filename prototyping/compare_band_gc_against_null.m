% Input parameters
input_parameters;
ncdt = length(conditions);
nsub = length(cohort);
subject_id = 'DiAs';
% EEG bands
bands = struct;
bands.omega = [0 120];
bands.delta = [0 4];
bands.theta = [4 7];
bands.alpha = [8 12];
bands.beta = [13 30];
bands.gamma = [32 60];
bands.hgamma = [60 120];
%%

fn = fieldnames(bands);
nbands = length(fn);
for c = 1:ncdt
    condition = conditions{c};
    for i = 1:nbands
        band = bands.(fn{i});
        bandName = fn{i};
        % Read condition specific time series
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
            'condition',conditions{c}, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        indices = gc_input.indices;
        % Compute pair band GC
        pGC = ts_to_spgc(X, 'morder',morder, 'regmode', regmode, ...
                    'dim',dim,'band',band ,'sfreq',sfreq,'nfreqs',nfreqs,...
                    'tstat',tstat,'mhtc', mhtc, 'alpha', alpha,...
                    'conditional',conditional);
        % Compute group band GC
        gGC = ts_to_smvgc(X, 'gind', indices, 'morder',morder, 'regmode', regmode, ...
                    'dim',dim,'band',band ,'sfreq',sfreq,...
                    'tstat',tstat,'mhtc', mhtc, 'alpha', alpha);
        bandGC.(condition).(bandName).pairwise = pGC;
        bandGC.(condition).(bandName).groupwise = gGC;
    end
end
bandGC.indices = indices;

%% Plot results in a specific condition

condition = 'Face';
ptitle = cell(nbands,1);
F = cell(nbands,1);
cm = [];
Fmax = 0.001;
plotm = 0;
psize = 0;

%Plot pair GC
for b = 1:nbands
    bandName = fn{b};
    pGC = bandGC.(condition).(bandName).pairwise;
    ptitle{b} = {bandName condition};
    F{b} = pGC.F;
end

plot_gc(F',ptitle',cm,Fmax,plotm,psize)
        
%% Plot group GC
condition = 'Face';

Fmax = 0.005;
for b = 1:nbands
    bandName = fn{b};
    gGC = bandGC.(condition).(bandName).groupwise;
    ptitle{b} = {bandName condition};
    F{b} = gGC.F;
end

plot_gc(F',ptitle',cm,Fmax,plotm,psize)


