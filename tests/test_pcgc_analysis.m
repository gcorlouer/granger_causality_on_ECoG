%% Testing validity of single trial gc analysis
%% Plot single trial distribution
% Pick chan:
iR = 3;
iF = 4;
c1 = 4;
c2 = 2;
hbins = 20;
% Subject id
s =3;
% Get single trial distribution
bias1 = dataset(c1,s).bias; bias2 = dataset(c2,s).bias;
sF1 = dataset(c1,s).single_F; sF2 = dataset(c2,s).single_F; 
sF1 = sF1(iF,iR,:); sF2 = sF2(iF,iR,:);
% Plot histogram of single trial distribution

histogram(sF1,hbins,'facecolor','g', 'Normalization', 'probability');
hold on
histogram(sF2,hbins,'facecolor','r', 'Normalization', 'probability');
hold off
title(sprintf('\nGC single-trial empirical distributions\n'));
xlabel('GC (green = Condition 1, red = Condition 2)')

%% Test detrending

% pdeg = 2;
% mX = mean(X,1);
% [dX,~,~,~] = mvdetrend(X,pdeg,[]);
% mdX = mean(dX,1);
% k = 5;
% subplot(2,1,1)
% plot(time, mX(1,:,k))
% title('No detrend')
% subplot(2,1,2)
% plot(time, mdX(1,:,k))
% title(['Detrend of deg ' num2str(pdeg)])