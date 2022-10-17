function F = ss_to_GC(model, args)

% Return band-specific between and within groupwise conditional GC
% from state space model

arguments
    model struct % ss model
    args.connect char % connectivity: 'groupwise' or 'pairwise'
    args.group cell % group indices
    args.dim double = 3; % dimension along which to integrate
    args.band double = [0 40]; % frequency band
    args.sfreq double = 250; 
    args.nfreqs double = 1024
end

group = args.group; connect = args.connect;
dim = args.dim; band = args.band; sfreq = args.sfreq; nfreqs = args.nfreqs;

g = length(group);

if strcmp(connect, 'groupwise')
    % Compute between group spectral GC
    f = ss_to_sgwcgc(model.A,model.C,model.K,model.V,group,nfreqs);
    % Compute within group spectral GC
    for ig=1:g
        f(ig,ig,:) = ss_to_scggc(model.A,model.C,model.K,model.V,group{ig},nfreqs);
    end
else
    % Compute pairwise conditional spectral GC
    f = ss_to_spwcgc(model.A,model.C,model.K,model.V,nfreqs);
end

% Integrate over frequency band
F = bandlimit(f,dim, sfreq, band);

end