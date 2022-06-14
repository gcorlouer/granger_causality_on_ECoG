function F = ts_to_single_pair_unconditional_gc(X, args)
% This function compute pairwise unconditional single trial GC distribution 
%from time series X

arguments
    X double;
    args.morder double = 3;
    args.regmode = 'OLS';
    args.tstat char = 'LR' % GC statistics
    args.mhtc char = 'FDRD';
    args.alpha double = 0.05;
end

morder = args.morder; regmode = args.regmode;

% Initialise variables
[n,~,N] = size(X);
F = zeros(n,n,N);
% Loop over trials
for i=1:N
    trial = X(:,:,i);
    % Pairwise unconditional GC estimation from single regression
    for ic=1:n
        for jc=ic:n
            if ic==jc
                 F(ic,jc,i) = 0;
            else
            % Extract trial from pair of channels
            trial_pair = trial([ic jc],:,:);
            % VAR model pair of channels
            VAR = ts_to_var_parameters(trial_pair, 'morder', morder, 'regmode', regmode);
            V = VAR.V;
            A = VAR.A;
            disp(VAR.info)
            % Estimate pairwise unconditional GC
            f = var_to_pwcgc(A,V);
            F(ic,jc,i) = f(1,2);
            F(jc,ic,i) = f(2,1);
            end
        end
    end
F(isnan(F))=0;
end
