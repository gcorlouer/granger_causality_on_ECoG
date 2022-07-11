% Testing null of no pairwise unconditional gc for all channels
%% Input parameters
input_parameters;
ncdt = length(conditions);
nsub = length(cohort);
dataset = struct;
subject_id = 'DiAs';

for c=1:ncdt
        condition = conditions{c};
        % Read condition specific time series
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
            'condition',condition, 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        [n,m,N] = size(X);
        F = zeros(n,n);
        sig = zeros(n,n);
        % Pairwise unconditional GC, loop over pairs of channels [i j]
        for i=1:n
            F(i,i) = 0;
            sig(i,i) = 0;
            for j=i+1:n
                % Compute pairwise GC with double regression
                pF = ts_to_dual_pgc(X([i j],:,:), 'morder',morder, 'regmode', regmode, ...
                    'tstat',tstat,'mhtc', mhtc, 'alpha', alpha);
                F(i,j) = pF.F(1,2);
                F(j,i) = pF.F(2,1);
                sig(i,j) = pF.sig(1,2);
                sig(j,i) = pF.sig(2,1);
            end
        end
        PwGC.(condition).F = F;
        PwGC.(condition).sig = sig;
end
%%

fname = 'null_gc_all_chan.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'PwGC')