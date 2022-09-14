function I = cov_to_gMI(V, args)


arguments
    V double % ss model
    args.gind struct % group indices 
end

gind = args.gind;
gi = fieldnames(gind); 

ng = length(gi); % Number of groups
I = zeros(ng, ng);

for i=1:ng
    for j=1:ng
        x = gind.(gi{i});
        y = gind.(gi{j});
        if i==j
            % Return gobal GC for diagonal elements
            I(i,j) = cov_to_cmii(V,x);
        else 
            % Return mvgc between group of populations
            I(i,j) = cov_to_mvmi(V,x,y);
        end            
    end
end
end