function tstat = permute_condition_GC(X, args)
% compare pair or group GC between conditions
% permutation testing

arguments
    X double % Concatenated condition time series (along trials dimensions)
    args.group cell % group indices 
    args.connect char % Observed difference in MI between conditions
    args.morder double = 5 % VAR model order
    args.ssmo double = 20 % SS model order
    args.Ntrial double = 56 % Number of trials per condition
    args.Ns double = 500; % Number of permutations
    args.dim double = 3; % Integration dimension
    args.sfreq double = 250; % Sampling frequency
    args.nfreqs double = 1024; % Number of frequency bins
    args.band double = [0 40]; % frequency band
end

group = args.group; connect = args.connect;
N = args.Ntrial; Ns = args.Ns;
morder = args.morder; ssmo = args.ssmo;
sfreq = args.sfreq; dim = args.dim; band = args.band; nfreqs = args.nfreqs;

bandstr = mat2str(band);

[n,m,Nt] = size(X); % Number of concatenated trials

if strcmp(connect, 'groupwise')
    ng = length(group);
    F1 = zeros(ng,ng,Ns); % F for permuted condition 1
    F2 = zeros(ng,ng,Ns); % F for permuted condition 2
else
    F1 = zeros(n,n,Ns); % F for permuted condition 1
    F2 = zeros(n,n,Ns); % F for permuted condition 2
end

% Future and past regression for state space modeling
pf = 2 * morder;

% Compute permutation GC
for s=1:Ns
    if mod(s, Ns/10) == 0
        fprintf('Compare condition %s %s GC sample %d of %d \n',connect,bandstr, s,Ns);
    end
    % Sample trials without replacement
    trials = randperm(Nt);
    % Estimate SS model for each conditions
    % condition 1
    trial1 = trials(1:N);
    X1 = X(:,:,trial1);
    [A1,C1,K1,V1,~,~] = tsdata_to_ss(X1,pf,ssmo);
    model1.A = A1; model1.C = C1; model1.K = K1; model1.V = V1;
    % condition 2
    trial2 = trials(N+1:Nt);
    X2 = X(:,:,trial2);
    [A2,C2,K2,V2,~,~] = tsdata_to_ss(X2,pf,ssmo);
    model2.A = A2; model2.C = C2; model2.K = K2; model2.V = V2;
    % Compute band specific GC 
    % condition 1
    F1(:,:,s) = ss_to_GC(model1, 'connect', connect ,'group', group,...
    'dim', dim, 'sfreq', sfreq, 'nfreqs', nfreqs, 'band',band);
    % condition 2
    F2(:,:,s) = ss_to_GC(model2, 'connect', connect ,'group', group,...
    'dim', dim, 'sfreq', sfreq, 'nfreqs', nfreqs, 'band',band);
end
fprintf('\n');
tstat = F1 - F2;
end