%% Input parameters
input_parameters;
nsub = length(cohort);
ncdt = length(conditions);
suffix = '_condition_ts.mat';
subject = 'DiAs';
condition = 'Rest';
connect = 'pairwise';
morder = 12;
ssmo = 20;
maic = cell(ncdt,1);
mbic =  cell(ncdt,1);
mhqc =  cell(ncdt,1);
msvc =  cell(ncdt,1);
%% Estimate GC

gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
    'condition',condition, 'suffix', suffix);
X = gc_input.X;
% Delete other visual channels
indices = gc_input.indices;
[n,m,N] = size(X);
sfreq = gc_input.sfreq;
F = zeros(n,n);
for i=1:n
    fprintf('Source channel %d of %d',i,n);
    for j=i+1:n
        % Estimate SS model
        x = X([i j],:,:);
        pf = 2 * morder;
        [model.A,model.C,model.K,model.V,~,~] = tsdata_to_ss(x,pf,ssmo);
        % Compute pairwise unconditional GC
        group = {};
        pF = ss_to_GC(model, 'connect', connect ,'group', group,...
            'dim', dim, 'sfreq', sfreq, 'nfreqs', nfreqs, 'band',band);
        F(i,j) = pF(1,2);
        F(j,i) = pF(2,1);
    end
    fprintf('\n');
end
GC.(subject).(condition).('F') = F;
GC.('band') = band;
GC.('connectivity') = connect;
GC.(subject).indices = indices;
%% Save dataset for plotting in python

fname = 'unconditional_GC.mat';
fpath = fullfile(datadir, fname);
save(fpath, 'GC')

%%

% %% Estimate pairwise unconditional model order
% condition = 'Face';
% fprintf('Condition %s \n',condition);
% gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
%     'condition',condition, 'suffix', suffix);
% X = gc_input.X;
% % Delete other visual channels
% indices = gc_input.indices;
% [n,m,N] = size(X);
% sfreq = gc_input.sfreq;
% moaic = zeros(n,n);
% mobic = zeros(n,n);
% mohqc = zeros(n,n);
% mosvc = zeros(n,n);
% for i=1:n
%     fprintf('Source channel %d of % \n',i,n);
%     for j=1:n
%         fprintf('Target channel %d of %d',j,n);
%         if i==j
%             continue
%         else
%             x = X([i j],:,:);
%             [moaic(i,j),mobic(i,j),mohqc(i,j), ~] = tsdata_to_varmo(x, ... 
%                     momax,regmode,alpha,pacf,plotm,verb);
%             % Estimate VAR model.
%             morder = moaic(i,j);
%             % Estimate SS model
%             pf = 2*morder;
%             [mosvc(i,j),rmax] = tsdata_to_ssmo(x,pf,plotm);
%         end
%         fprintf('\n')
%     end
%     fprintf('\n');
% end
% maic = mean(moaic, 'all');
% mbic = mean(mobic, 'all');
% mhqc = mean(mohqc, 'all');
% msvc = mean(mosvc, 'all');

