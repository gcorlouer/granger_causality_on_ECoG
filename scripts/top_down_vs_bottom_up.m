%% Compare top down vs bottom up GC
% In this script we compare top down (F -> R) vs bottom-up (R -> F) GC 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  -Subject and condition-specific mutltirial HFA #X=(n x m x N)
% Output: -3x1 Z-score testing stochastic dominance of top down vs
%         bottom up GC in each condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm: 
%  for c in conditions, for s in subjects
%     R_idx = gc_input.indices('R')
%     F_idx = gc_input.indices('F')  
%     F = ts_to_single_pcgc(X, args);
%     F_td{c,s} = F(R_idx, F_idx);
%     F_bu{c,s} = F(F_idx, R_idx);
%  F_td = flatten(F_td{c,s});
%  F_bu = flatten(F_bu{c,s});
%  z = wilcox(F_td, F_bu); #z = ncdt x 1
