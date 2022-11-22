%% Estimate rolling VAR and SS model on multitrial data
input_parameters;
suffix = ['_condition_visual_' signal '.mat'];
conditions = {'Rest','Face','Place'};
nsub = length(cohort);
ncdt = length(conditions);
plotm = [];
verb = [];
momax = 30;
ModelOrder = struct;
for s=1:nsub
    subject = cohort{s};
    for c=1:ncdt
        condition = conditions{c};
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                        'condition',condition, 'suffix', suffix);
        sfreq = gc_input.sfreq;
        X = gc_input.(condition);
        [n, m,N] = size(X);
        time = gc_input.time;
        % Number of windows
        nwin = floor((m - mw)/shift +1);
        moaic = zeros(nwin,1);
        mobic = zeros(nwin,1);
        mohqc = zeros(nwin,1);
        mosvc = zeros(nwin,1);
        rho = zeros(nwin,1);
        % Take same number of trials as Face (no bias)
        if strcmp(condition, 'Rest')
                trial_idx = 1:N;
                Ntrial = 56; 
                trials = datasample(trial_idx, Ntrial,'Replace',false);
                X = X(:,:, trials);
        end
        win_size = mw/sfreq;
        fprintf('Rolling time window of size %4.2f \n', win_size)
        for w=1:nwin
            fprintf('Time window number %d over %d \n', w,nwin)
            % window offset
            o = (w-1)*shift;   
            % the window
            W = X(:,o+1:o+mw,:); 
            % Estimate var model order with multiple information criterion
            [moaic(w),mobic(w),mohqc(w),~] = tsdata_to_varmo(W, ... 
                        momax,regmode,alpha,pacf,[],verb);
        end
        morder = mean(mohqc(w));
        % Sample to time
        time_offset = shift/sfreq;
        win_time = zeros(nwin,mw);
        for w=1:nwin
            o = (w-1)*shift; 
            win_time(w,:) = time(o+1:o+mw);
        end
        % Estimate rolling VAR model order
        fprintf('Rolling time window of size %4.2f \n', win_size)
        for w=1:nwin
            fprintf('Time window number %d over %d \n', w,nwin)
            % window offset
            o = (w-1)*shift; 
            % the window
            W = X(:,o+1:o+mw,:);
            VAR = ts_to_var_parameters(W, 'morder', morder, 'regmode', regmode);
            % Spectral radius
            rho(w) = VAR.info.rho;
        end
        % Estimate rolling SS model order
        pf = 2*morder;
        for w=1:nwin
            fprintf('Time window number %d over %d \n', w,nwin)
            % window offset
            o = (w-1)*shift; 
            % the window
            W = X(:,o+1:o+mw,:);
            ssmo = ssmodel_order(W,pf,[]);
            mosvc(w) = ssmo.mosvc;
        end
        ModelOrder.(subject).(condition).('aic') = moaic;
        ModelOrder.(subject).(condition).('hqc') = mohqc;
        ModelOrder.(subject).(condition).('bic') = mobic;
        ModelOrder.(subject).(condition).('ssmo') = mosvc;
        ModelOrder.(subject).(condition).('rho') = rho;
        ModelOrder.(subject).(condition).('time') = win_time;
    end
end
%% Save dataset
fname = [signal '_rolling_model_order_estimation.m'];
fpath = fullfile(datadir, fname);
save(fpath, 'ModelOrder')