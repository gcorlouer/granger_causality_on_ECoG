function F = ts_to_single_smvgc(X, args)
% Compute single trial band specific group conditional GC
arguments
    X double % Time series
    args.gind struct % group indices 
    args.morder double = 3 % model order
    args.regmode char = 'OLS' % model regression (OLS or LWR)
    args.dim double = 3;
    args.band double = [0 40];
    args.sfreq double = 250;
    args.nfreqs double = 1024
end

gind = args.gind; morder = args.morder; regmode = args.regmode; dim = args.dim;
gi = fieldnames(gind); band = args.band; sfreq = args.sfreq; nfreqs = args.nfreqs;

[n,m,N]=size(X);

ng = length(gi); % Number of groups
groups = cell(ng,1);

% Write functional groups into cell array
for g=1:ng
    groups{g} = gind.(gi{g});
end

% Compute spectral global and group conditional GC
fglobal = zeros(ng, nfreqs+1, N) ;
f =  zeros(ng, ng, nfreqs+1, N) ;

for k=1:N
    VAR = ts_to_var_parameters(X(:,:,k), 'morder', morder, 'regmode', regmode);
    fglobal(:,:,k) = var_to_sgwcggc(VAR.A,VAR.V,groups,nfreqs);
    f(:,:,:,k) = var_to_sgwcgc(VAR.A,VAR.V,groups,nfreqs);
end
% Diagonal entries are global GC
for i=1:ng
    f(i,i,:,:) = fglobal(i,:,:);
end

% Integrate over frequency band
F = bandlimit(f,dim, sfreq, band);
end

%%


% for k=1:N
%     VAR = ts_to_var_parameters(X(:,:,k), 'morder', morder, 'regmode', regmode);
%     for i=1:ng
%         for j=1:ng
%             x = gind.(gi{i});
%             y = gind.(gi{j});
%             nx = length(x); ny = length(y); nz = n -nx - ny;
%             if i==j
%                 % Return multi-information for diagonal elements
%                 fk = zeros(nx,nfreqs+1);
%                 for l =1:nx
%                     xk = x;
%                     xk(l) = [];
%                     fk(l,:) = var_to_smvgc(VAR.A,VAR.V,x(l), xk,nfreqs);
%                 end
%                 f(i,j,:,k) = sum(fk,1);
%              % Compute spectral group GC
%             else 
%                 f(i,j,:,k) = var_to_smvgc(VAR.A,VAR.V,x,y,nfreqs);
%             end            
%         end
%     end
% end
% 
% % Integrate over frequency band
% F = bandlimit(f,dim, sfreq, band);
% end