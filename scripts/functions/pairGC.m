function F = pairGC(model, args)

arguments
    model struct % ss model
    args.dim double = 3; % dimension along which to integrate
    args.band double = [0 40]; % frequency band
    args.sfreq double = 250; 
    args.nfreqs double = 1024
end

dim = args.dim; band = args.band; sfreq = args.sfreq; nfreqs = args.nfreqs;

% Compute pairwise conditional spectral GC
f = ss_to_spwcgc(model.A,model.C,model.K,model.V,nfreqs);

% Integrate over frequency band
F = bandlimit(f,dim, sfreq, band);

end