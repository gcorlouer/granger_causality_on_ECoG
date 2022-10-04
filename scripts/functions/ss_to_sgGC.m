function F = ss_to_sgGC(ss, args)

arguments
    ss struct % ss model
    args.gind struct % group indices
    args.sfreq double = 250;
    args.nfreqs double = 1024;
end

gind = args.gind; gi = fieldnames(gind); sfreq = args.sfreq; 
nfreqs = args.nfreqs;

ng = length(gi); % Number of groups
F = zeros(ng, ng, nfreqs+1);

for i=1:ng
    for j=1:ng
        x = gind.(gi{i});
        y = gind.(gi{j});
        if i==j
            % Return gobal GC for diagonal elements
            F(i,j,:) = ss_to_scggc(ss.A,ss.C,ss.K,ss.V,x, nfreqs);
        else 
            % Return mvgc between group of populations
            F(i,j,:) = ss_to_smvgc(ss.A,ss.C,ss.K,ss.V,x,y, nfreqs);
        end            
    end
end
end
