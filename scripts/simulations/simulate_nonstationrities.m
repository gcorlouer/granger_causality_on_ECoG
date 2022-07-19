input_parameters
nobs = 500;
tsdim = 4;
N = 50;
rhoX = 0.9;
rhoY=0.8;
morderX = 10;
morderY = 1;
plotm = [];
[X,~, ~, Cx] = var_simulation(tsdim, ...
    'nobs',nobs, 'ntrials', N, 'specrad', rhoX, 'morder', morderX);
[Y,~, ~, Cy] = var_simulation(tsdim, ...
    'nobs',nobs, 'ntrials', N, 'specrad', rhoY,  'morder', morderY);

Z = cat(2,X,Y);
Z = X; 
%% VAR modeling

% Estimate var model order with multiple information criterion
[moaic,mobic,mohqc,molrt] = tsdata_to_varmo(Z([1 2],:,:), ... 
                    momax,regmode,alpha,pacf,plotm,verb);

morder = moaic;
%% Estimate GC

% Pairwise conditional GC
pGC = ts_to_spgc(X, 'morder',morder, 'regmode', regmode, ...
                    'tstat',tstat,'mhtc', mhtc, 'alpha', alpha, ...
                    'conditional', true, 'band', [], 'dim',3);
F = pGC.F;
sig = pGC.sig;
%% Plot results

plotF = {F sig Cx};
ptitle = {'Estimated' 'Significance' 'True'};
plot_gc(plotF,ptitle,[],[],0)

%% Plot spectral GC
lam = pGC.freqs;
logsx = false;
f = pGC.f;
ptitle = 'Spectral GC';
plot_sgc(f,lam,ptitle,logsx,0,[],[])