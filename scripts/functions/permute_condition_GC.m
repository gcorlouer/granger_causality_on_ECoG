function tstat = permute_condition_GC(X, args)
% compare pair or group GC between conditions
% permutation testing

arguments
    X double % Concatenated condition time series (along trials dimensions)
    args.group cell % group indices 
    args.connect char % Observed difference in MI between conditions
    args.morder double = 5 % VAR model order
    args.ssmo double = 20 % SS model order
    args.Ntrial double = [56 56] % Number of trials per condition
    args.Ns double = 500; % Number of permutations
    args.dim double = 3; % Integration dimension
    args.sfreq double = 250; % Sampling frequency
    args.nfreqs double = 1024; % Number of frequency bins
    args.band double = [0 40]; % frequency band
end

group = args.group; connect = args.connect;
Ntrial = args.Ntrial; Ns = args.Ns;
morder = args.morder; ssmo = args.ssmo;
sfreq = args.sfreq; dim = args.dim; band = args.band; nfreqs = args.nfreqs;

[~,~,Nt] = size(X); % Number of concatenated trials
trial_idx = 1:Nt; % Indices of trials

if strcmp(connect, 'groupwise')
    ng = length(group);
    F = zeros(ng,ng,Ns);
else
    F = zeros(n,n,Ns);
end
permF = cell(2,1);
% Future and past regression for state space modeling
pf = 2 * morder;

% Compute permutation GC
for i=1:2
    for s=1:Ns
        fprintf('MVGC: permutation sample %d of %d',s,Ns);
        trials = datasample(trial_idx, Ntrial(i),'Replace',false);
        Xp = X(:,:,trials);
        % Estimate SS model
        [A,C,K,V,~,~] = tsdata_to_ss(Xp,pf,ssmo);
        model.A = A; model.C = C; model.K = K; model.V =V;
        % Compute GC 
        F(:,:,s) = ss_to_GC(model, 'connect', connect ,'group', group,...
        'dim', dim, 'sfreq', sfreq, 'nfreqs', nfreqs, 'band',band);
        fprintf('\n');
    end
    permF{i} = F;
end
tstat = permF{1} - permF{2};
end