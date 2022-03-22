function F = ts_to_single_pGC(X, args)

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
for i=1:N
    trial = X(:,:,i);
    VAR = ts_to_var_parameters(trial, 'morder', morder, 'regmode', regmode);
    V = VAR.V;
    A = VAR.A;
    disp(VAR.info)
    % Pairwise conditional GC estimation
    % Single regression
    F(:,:,i) = var_to_pwcgc(A,V);
    F(isnan(F))=0;
end
