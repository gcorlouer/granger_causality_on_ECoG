%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, we test the use of wilcoxon sum rank test to test
% difference in GC accross conditions on simulated VAR models.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TODO resimulate stuff
%% Initialise parameters

% Time series parameters
morder = 3; specrad = 0.7; nx=3; ny=5 ; nz=4; n = nx+ny+nz;
x = 1:nx;
y = nx+1:nx+ny;

% VAR residuals multi-information (generalised correlation)
mii= 2;

% VAR coefficients decay weighting factor
cdw= 1;
regmode = 'OLS';

% Observations and trials
m = 100; N = {56, 120};

% Generate connectivity matrix
cmatrix = randi([0 1], n);

% Number of conditions
ncdt = 2;
ts = cell(ncdt,1); F = cell(ncdt,1); Fa = cell(ncdt,1);

% Histogram 
cell(ncdt,1);

%% Simulate VAR in both conditions and compute F

for c=1:ncdt
    % Generate VAR model
    A = var_rand(cmatrix,morder, specrad, cdw);
    V = corr_rand(n,mii); 
    % Generate time series from different model for each conditions
    ts{c} = var_to_tsdata(A,V,m,N{c});
    % Calculate actual GC 
    Fa{c} = var_to_mvgc(A,V,x,y);

    % Estimate single trial pcgc
    for i=1:N{c}
        % Take single trial
        X = ts{c}(:,:,i);
        % Estimate gc on each trials
        [A,V] = tsdata_to_var(X,morder,regmode);
        f = var_to_mvgc(A,V,x,y); 
        F{c}(i) = f;
    end
end

%% Compute statistics to compare F


Fc1 = F{1};
Fc2 = F{2};
n1 = length(Fc1);
% Compute mann whitney 
[z_mann, U_mann] = mann_whitney(Fc1, Fc2);
pval = 2*(1-normcdf(abs(z_mann)));
% Return wether significant
sig = pval < 0.05;
% Compute Wilcoxon ranksum
[p,h,stats] = ranksum(Fc2, Fc1);
zSum = stats.zval;
w_stat = stats.ranksum;
U_will = w_stat - n1*(n1+1)/2;

%% Compare GC and estimated

fprintf('\n--------------------------------------------------\n');
fprintf('GC                      Condition 1    Condition 2\n');
fprintf('--------------------------------------------------\n');
fprintf('Actual                :   %6.4f         %6.4f    !\n',Fa{1},Fa{2});
fprintf('Median Estimated      :   %6.4f         %6.4f\n',median(F{1}), median(F{2}) );
fprintf('--------------------------------------------------\n');

%% Report inference

% Statistical dominance with Mann whitney
fprintf('\n--------------------------------------------------\n');
fprintf('Stochastic dominance\n');
fprintf('----------------------------------------\n');
fprintf('Dominance with Mann Whitney:\n')
if sig
	if z_mann > 0, sigstr = 'YES (Condition 2 > Condition 1) - WRONG!';
	else,     sigstr = 'YES (Condition 1 > Condition 2) - WRONG!';
	end
else
    sigstr = 'NO - CORRECT!';
end
fprintf('U Mann    :   %6.4f\n',U_mann     );
fprintf('z-score Sum    :   %6.4f\n',z_mann     );
fprintf('p-value   Sum  :   %g\n',   pval  );
fprintf('Significant :   %s\n',   sigstr);
fprintf('----------------------------------------\n');

% Statistical dominance with Wilcoxon
fprintf('Dominance with Wilcoxon summ rank:\n')
if h
	if zSum > 0, sigstr = 'YES (Condition 2 > Condition 1) - WRONG!';
	else,     sigstr = 'YES (Condition 1 > Condition 2) - WRONG!';
	end
else
    sigstr = 'NO - CORRECT!';
end
fprintf('U Sum    :   %6.4f\n',U_will     );
fprintf('z-score Sum    :   %6.4f\n',zSum     );
fprintf('p-value   Sum  :   %g\n',   p  );
fprintf('Significant :   %s\n',   sigstr);
fprintf('----------------------------------------\n');



%% Plot 

% diff = cmatrix2 - cmatrix1;
% subplot(3,2,1)
% imagesc(diff)
% title('True difference')
% colormap(gray)
% colorbar
% subplot(3,2,2)
% imagesc(pval)
% title('pval Mann')
% colormap(gray)
% colorbar
% subplot(3,2,3)
% imagesc(p)
% title('pval Wilcoxon')
% colormap(gray)
% colorbar
% subplot(3,2,4)
% imagesc(z_mann)
% title('Z Mann')
% colormap(gray)
% colorbar
% subplot(3,2,5)
% imagesc(zval)
% title('Z Wilcoxon')
% colormap(gray)
% colorbar

%% Plot histogram
hbins = 30;
histogram(F{1},hbins,'facecolor','g');
hold on
histogram(F{2},hbins,'facecolor','r');
hold off
title(sprintf('\nGC single-trial empirical distributions\n'));
xlabel('GC (green = Condition 1, red = Condition 2)')


%%


function [F,et] = single_trial_distribution(X,x,y,p,regm)

	% Single-trial GC empirical distribution
	%
	% X         multi-trial time-series data
	% x         target variable indices
	% y         source variable indices
	% p         VAR model order
	% regm      regression mode ('OLS' or 'LWR')
	%
	% F         single-trial GC estimates
	% et        elapsed time

	tic;
	N = size(X,3);
	F = zeros(N,1);
	N10 = round(N/10);
	for i = 1:N
		if rem(i,N10) == 0, fprintf('.'); end      % progress indicator
		[As,Vs] = tsdata_to_var(X(:,:,i),p,regm);  % estimated model based on single trial
		F(i)    = var_to_mvgc(As,Vs,x,y);          % single-trial GC estimate
	end
	et = toc; % elapsed time

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%