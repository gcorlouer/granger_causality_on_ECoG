%% Estimate VAR and SS model on multitrial data
input_parameters;
suffix = ['_condition_visual_' signal '.mat'];
conditions = {'Rest','Face','Place'};
nsub = length(cohort);
ncdt = length(conditions);
plotm = [];
verb = [];
momax = 30;
ModelOrder = struct;
%Multitrial VAR model estimation
for s=1:nsub
    subject = cohort{s};
    for c=1:ncdt
        condition = conditions{c};
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                        'condition',condition, 'suffix', suffix);
        sfreq = gc_input.sfreq;
        X = gc_input.(condition);
        [n, m,N] = size(X);
        subject = gc_input.subject;
        findices = gc_input.indices;
        fn = fieldnames(findices);
        time = gc_input.time;
        % Take same number of trials as Face (no bias)
        if strcmp(condition, 'Rest')
                trial_idx = 1:N;
                Ntrial = 56; 
                trials = datasample(trial_idx, Ntrial,'Replace',false);
                X = X(:,:, trials);
        end
        % Estimate VAR model.
        varmo = var_model_order(X, ... 
                        momax,regmode,alpha,pacf,plotm,verb);
        morder = varmo.bic;
        VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
        rho = VAR.info.rho;
        % Estimate SS model
        pf = 2*morder;
        ssmo = ssmodel_order(X,pf,plotm);
        [A,C,K,V,Z,E] = tsdata_to_ss(X,pf,ssmo.mosvc);
        ModelOrder.(subject).(condition).('varmo') = varmo;
        ModelOrder.(subject).(condition).('ssmo') = ssmo;
        ModelOrder.(subject).(condition).('rho') = rho;
    end
end
%% Save dataset
fname = [signal '_model_order_estimation.m'];
fpath = fullfile(datadir, fname);
save(fpath, 'ModelOrder')