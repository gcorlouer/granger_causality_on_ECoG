function F = ss_to_gGC(ss, args)


arguments
    ss struct % ss model
    args.gind struct % group indices 
end

gind = args.gind;
gi = fieldnames(gind); 

ng = length(gi); % Number of groups
F = zeros(ng, ng);

for i=1:ng
    for j=1:ng
        x = gind.(gi{i});
        y = gind.(gi{j});
        if i==j
            % Return gobal GC for diagonal elements
            f(i,j,:) = ss_to_scggc(ss.A,ss.C,ss.K,ss.V,x, fres);
        else 
            % Return mvgc between group of populations
            f(i,j,:) = ss_to_mvgc(ss.A,ss.C,ss.K,ss.V,x,y);
        end            
    end
end
end