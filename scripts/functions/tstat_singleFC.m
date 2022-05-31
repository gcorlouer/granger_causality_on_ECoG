function tstat = tstat_singleFC(Subject, args)
% Compute z scores and z crit for each subjects between
% single trial MI and single trial GC for a given pair of condition.
arguments
    Subject struct
    args.subject_id string = 'DiAs'
    args.comparison cell = {'Face', 'Place'};
    args.FC string = 'single_MI';
    args.alpha double = 0.05
    args.mhtc string = 'Sidak';
end

subject_id = args.subject_id; comparison = args.comparison; FC = args.FC; 
alpha = args.alpha; mhtc = args.mhtc; 

fc1 = Subject.(subject_id).(comparison{1}).(FC);
fc2 = Subject.(subject_id).(comparison{2}).(FC);
indices = Subject.(subject_id).indices;
% Test whether fc1 > fc2 (in the sense of stochastic dominance)
tstat = mann_whitney_array(fc1,fc2, indices, alpha, mhtc);

end

function tstat = mann_whitney_array(fc1,fc2, indices, alpha, mhtc)
% Return Z score and statistics from comparing two functional connectivity
% arrays
populations = fieldnames(indices);
ng = length(populations);
z = zeros(ng,ng);
for i=1:ng
    for j=1:ng
        z(i,j) = mann_whitney(fc2(i,j,:),fc1(i,j,:));
    end
end
pvals = erfc(abs(z)/sqrt(2));
% Number of hypotheses
nsub = 3;
nhyp = numel(pvals)*nsub;
% No pvalues
nopv = true;
% Compute pcrit and Zcrit
pcrit = significance(nhyp,alpha,mhtc,nopv);
zcrit = sqrt(2)*erfcinv(pcrit);
tstat.z = z;
tstat.zcrit = zcrit;
end

