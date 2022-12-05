input_parameters
subject = 'DiAs';
conditions = {'Rest','Face', 'Place'};
ncdt = length(conditions);
suffix = ['_condition_two_chans_' signal '.mat'];
connect = 'pairwise';
morder = 10;  % suggested model order
nfreq = 1024;
ssmo = 39; % suggested state space model order
dim = 3;
GC = struct;
comparisons = {{'Face' 'Rest'}, {'Place' 'Rest'}, {'Face' 'Place'}};
compare = {'FR' 'PR' 'FP'};
nComp = size(comparisons,2);
%%
for c=1:nComp
            comparison = comparisons{c};
            comparison_name = [comparison{1}(1) 'vs' comparison{2}(1)];
            fprintf('Comparison %s subject %s \n', comparison_name, subject)
            for i=1:2
                condition = comparison{i};
                gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                    'condition',condition, 'suffix', suffix);
                X = gc_input.X;
                [n,m,N] = size(X);
                % Take same number of trials as Face (faster computation)
                if strcmp(condition, 'Rest')
                    trial_idx = 1:N;
                    Nt = 56;
                    trials = datasample(trial_idx, Nt,'Replace',false);
                    X = X(:,:, trials);
                end
                indices = gc_input.indices;
                Xc{i} = X;
                [n,m,N] = size(Xc{i});
                sfreq = gc_input.sfreq;
                % Estimate SS model
                pf = 2 * morder;
                [model.A,model.C,model.K,model.V,~,~] = tsdata_to_ss(Xc{i},pf,ssmo);
                % Compute observed GC
                fn = fieldnames(indices);
                ng = length(fn);
                group = cell(ng,1);
                for k=1:length(fn)
                    group{k} = double(indices.(fn{k}));
                end
                group = group';
                F{i} = ss_to_GC(model, 'connect', connect ,'group', group,...
        'dim', dim, 'sfreq', sfreq, 'nfreqs', nfreqs, 'band',band);
            end
            % Observed statistic
            obsStat = F{1} - F{2};
            % Concatenate for permutation testing
            X = cat(3, Xc{1},Xc{2});
            % Compute permutation test statistic
            tstat = permute_condition_GC(X, 'connect', connect ,'group', group,...
                'Ntrial', N, 'Ns', Ns, 'morder', morder, 'ssmo',ssmo, ...
                'sfreq',sfreq, 'nfreqs', nfreqs,'dim',dim, 'band',band);
            % Permutation testing
            stat = permtest(tstat, 'obsStat', obsStat, 'Ns', Ns, ...
                'alpha', alpha, 'mhtc','FDRD');
            % Save into GC structure for python plotting
            GC.(subject).(comparison_name).('T') = stat.T;
            GC.(subject).(comparison_name).('z') = stat.z;
            GC.(subject).(comparison_name).('sig') = stat.sig;
            GC.(subject).(comparison_name).('pval') = stat.pval;
            GC.(subject).(comparison_name).('zcrit') = stat.zcrit;
            GC.(subject).indices = indices;
            GC.('band') = band;
            GC.('connectivity') = connect;
            fprintf('\n')
end

%% Save dataset for plotting in python
bandstr = mat2str(band);
fname = ['two_chans_compare_gc_' bandstr 'Hz.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'GC')