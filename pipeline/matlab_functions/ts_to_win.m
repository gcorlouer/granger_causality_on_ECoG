function W = ts_to_win(X, mw, nw)
%% Take time series X and return a window a random origin
% lw: integer length of window
% nw: number of windows

[n,m,N] = size(X);

W = zeros(n,mw,N,nw);
for i=1:nw
    o = randsample(m-mw,nw);
    W(:,:,:,i) = X(:,o+1:o+mw,:,:);
end
end