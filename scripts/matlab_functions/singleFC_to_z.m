function z = singleFC_to_z(Struct, args)
% Compute z scores and z crit for each subjects between
% single trial MI and single trial GC for a given pair of condition.
arguments
    Struct struct
    args.group = false;
    args.subject_id string = 'DiAs'
    args.comparison cell = {'Face', 'Place'};
    args.FC string = 'single_MI';
end

group = args.group; subject_id = args.subject_id; comparison = args.comparison; 
FC = args.FC; 

if group == false
    % compute individual subjects z scores
    fc1 = Struct.(subject_id).(comparison{1}).(FC);
    fc2 = Struct.(subject_id).(comparison{2}).(FC);
    indices = Struct.(subject_id).indices;
    % Test whether fc1 > fc2 (in the sense of stochastic dominance)
    z = mann_whitney_array(fc1,fc2, indices, group);
else
    % Compute group subjects z scores
    fc1 = Struct.(comparison{1}).(FC);
    fc2 = Struct.(comparison{2}).(FC);
    indices = Struct.indices;
    % Test whether fc1 > fc2 (in the sense of stochastic dominance)
    z = mann_whitney_array(fc1,fc2, indices, group);
end
end

function  z = mann_whitney_array(fc1,fc2, indices, group)
% Return Z score and statistics from comparing two functional connectivity
% arrays
nsub = size(fc1,1);
populations = fieldnames(indices);
ng = length(populations);
z = zeros(ng,ng);
if group == false
    for i=1:ng
        for j=1:ng
            z(i,j) = mann_whitney(fc2(i,j,:),fc1(i,j,:));
        end
    end
else
    for i=1:ng
        for j=1:ng
            f1 = cell(nsub,1);
            f2 = cell(nsub,1);
            for s=1:nsub
                f2{s} = fc2{s}(i,j,:);
                f1{s} = fc1{s}(i,j,:);
            end
            z(i,j) = mann_whitney_group(f2,f1);
        end
    end
end
end
