%% In this script we explore SS modelling on pair of channels
%% Input parameters 
input_parameters;
suffix = ['_condition_visual_' signal '.mat'];
conditions = {'Rest','Face','Place'};
nsub = length(cohort);
ncdt = length(conditions);
plotm = [];
verb = [];
momax = 30;
morder = zeros(nsub, ncdt); % morder from VAR modelling
med_mosvc = zeros(nsub, ncdt);% morder from ss modelling
%% Estimate pairwise VAR model order
for s=1:nsub
    subject = cohort{s};
    fprintf('Estimate pairwise morder for subject %s \n', subject)
    for c=1:ncdt
        condition = conditions{c};
        fprintf('Estimate pairwise morder for condition %s \n', condition)
        % Read data
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        indices = gc_input.indices;
        % Get retonotopic and Face channels indices
        R_idx = indices.('R');
        F_idx = indices.('F');
        % Sizes
        nR = length(R_idx);
        nF = length(F_idx);
        [n,m,N] = size(X);
        % Initialise model orders
        moaic =  zeros(nR,nF);
        mobic =  zeros(nR,nF);
        mohqc =  zeros(nR,nF);
        mosvc =  zeros(nR,nF);
        % Equalise trials number
        if strcmp(condition, 'Rest')
                trial_idx = 1:N;
                Ntrial = 56; 
                trials = datasample(trial_idx, Ntrial,'Replace',false);
                X = X(:,:, trials);
        end
        % Estmate varmorder
        for i=1:nR
            for j=1:nF
                iR = R_idx(i);
                iF = F_idx(j);
                x = X([iR,iF],:,:);
                % Estimate VAR model order
                [moaic(i,j),mobic(i,j),mohqc(i,j),~] = tsdata_to_varmo(x, ... 
                    momax,regmode,alpha,pacf,plotm,verb);
            end
        end
        % Flatten and return non zeros entries
        moaic = nonzeros(moaic);
        mobic = nonzeros(mobic);
        mohqc = nonzeros(mohqc);
        % Compute model order as median of other pairwise morder
        if strcmp(signal,'hfa')
            morder(c,s) = floor(median(mohqc)); 
        elseif strcmp(signal,'lfp')
            morder(c,s) = floor(median(mobic));
        end
        % Estimate pairwise SS model order
        pf = 2*morder(c,s);
        % Compute pairwise mosvc 
        for i=1:nR
            for j=i:nF
                iR = R_idx(i);
                iF = F_idx(j);
                x = X([iR,iF],:,:);
                [mosvc(i,j),rmax] = tsdata_to_ssmo(x,pf,plotm);
                [A,C,K,V,Z,E] = tsdata_to_ss(x,pf,mosvc(i,j));
            end
        end
        % Compute model order as median of other pairwise morder
        mosvc = nonzeros(mosvc);
        med_mosvc(s,c) = floor(median(mosvc));
    end
end
var_morder = max(morder,[],'all');
ss_morder = max(med_mosvc,[],'all');
fprintf('var model order for %s is %d \n', signal, var_morder)
fprintf('ss model order for %s is %d \n', signal, ss_morder)