%% Comparing GC between direction
% This script computes the difference between top down and bottom up GC
% using permutation testing. 
% For each subjects and conditions, take condition specific time series X
% For each pairs of channels (i,j) running in the list RF of ret and face
% indices compute observed stat pairwise GC between channels X(i,:) and X(j,:)
% Concatenate X(i,:) and X(j,:) into a single time series Xc 
% Randomly allocate trials to ret and face channels without replacement
% Compute pairwise BU and TD GC between Xi and Xj and return test statistic Ti
% Repeat 1000 times, compute pvalues, significance and zscore for each pair
% Plot heatmap with ret chan and F chan showing BU vs TD for each sub and
% cdt (plot in python). 
%% Input parameters

input_parameters;
ncdt = length(conditions);
nsub = length(cohort);

%%

subject = 'DiAs';
condition = 'Rest';
Ns = 500;

gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition, 'suffix', suffix);
X = gc_input.X;
[n,m,N] = size(X);
indices = gc_input.indices;

stat = compare_TD_BU_pgc(X, indices, 'morder', morder, 'ssmo', ssmo,...
    'Ns',Ns,'alpha',alpha, 'mhtc',mhtc);

%%

for s=1:nsub
    subject_id = cohort{s};
    for c=1:ncdt
        condition = conditions{c};
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject_id,...
            'condition',condition, 'suffix', suffix);
        X = gc_input.X;
        [n,m,N] = size(X);
        indices = gc_input.indices;
        R_idx = indices.('R');
        F_idx = indices.('F');
        nR = length(R_idx);
        nF = length(F_idx);
        % Test statistic
        T = zeros(nR, nF,N);
        for i=1:nR
            for j=1:nF
                x = X(i,:,:);
                y = X(j,:,:);
                % Compute observed top down and bottom up GC
                [A,C,K,V,Z,E] = tsdata_to_ss(X([i,j],:,:), pf, ssmo);
                F = ss_to_pwcgc(A,C,K,V);
                Ta(i,j) = F(1,2) - F(2,1);
                % Concatenate in one multitrial time series
                xRF = cat(1,x,y);
                [n,m,Nt] = size(xRF);
                for s=1:Ns
                    fprintf('MVGC: permutation sample %d of %d',s,Ns);
                    % Permute trial index
                    trials = randperm(Nt);
                    trialsR = trials(1:N);
                    trialsF = trials(N+1:Nt);
                    xR = xRF(:,:,trials1);
                    xF = xRF(:,:,trials2);
                    % Concatenate pairwise time series
                    xRFp = cat(1,x1,x2);
                    % Estimate permutation GC
                    [A,C,K,V,Z,E] = tsdata_to_ss(xRFp, pf, ssmo);
                    Fp =  ss_to_pwcgc(A,C,K,V);
                    % Compute permutation statistic
                    T(i,j,s) = Fp(1,2)-Fp(2,1);
                    fprintf('\n');
                end           
            end
        end
        
    end
end

