%% Input parameters
input_parameters;
nsub = length(cohort);
ncdt = length(conditions);
%%
for s=1:nsub
       subject = cohort{s};
    for c=1:ncdt
                condition = conditions{c};
                gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                    'condition',condition, 'suffix', suffix);
                X = gc_input.X;
                % Delete other visual channels
                indices = gc_input.indices;
                [n,m,N] = size(X);
                sfreq = gc_input.sfreq;
                % Estimate SS model
                pf = 2 * morder;
                [model.A,model.C,model.K,model.V,~,~] = tsdata_to_ss(X,pf,ssmo);
                % Compute observed GC
                fn = fieldnames(indices);
                ng = length(fn);
                group = cell(ng,1);
                for k=1:length(fn)
                    group{k} = double(indices.(fn{k}));
                end
                group = group';
                F = ss_to_GC(model, 'connect', connect ,'group', group,...
                    'dim', dim, 'sfreq', sfreq, 'nfreqs', nfreqs, 'band',band);
                GC.(subject).(condition).('F') = F;
                GC.('band') = band;
                GC.('connectivity') = connect;
                GC.(subject).indices = indices;
    end
end
%% Save dataset for plotting in python

fname = 'GC_effect_size.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'GC')
