%% In this script we run the pipeline on 2 chans to validate it

bands = struct;
bands.theta = [4 7];
bands.alpha = [8 12];
bands.beta = [13 30];
bands.gamma = [32 60];
bands.hgamma = [60 120];
band_names = fieldnames(bands);
nband = length(band_names);

input_parameters
conditions = {'Rest', 'Face', 'Place'};
ncdt = length(conditions);
signal = 'lfp';
suffix = '_condition_pick_chans_lfp';
p = 10;  % suggested model order
nfreq = 1024;
mosvc = 39; % suggested state space model order
dim = 3;
GC = struct;

for ib=1:nband
band = bands.(band_names{ib});
    for c=1:ncdt
            condition = conditions{c};
            % Read condition specific time series
            gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                'condition',condition, 'suffix', suffix);
            X = gc_input.X;
            sfreq = gc_input.sfreq;
    %         % VAR modeling
    %         [moaic,mobic,mohqc,molrt] = tsdata_to_varmo(X,momax,regmode,alpha,0,[],[]);
    %         % SS modeling
            pf = 2*p;
            % [mosvc,rmax] = tsdata_to_ssmo(X,pf,[]);
            [A,C,K,V,Z,E] = tsdata_to_ss(X,pf,mosvc);
            f = ss_to_spwcgc(A,C,K,V,nfreq);
            GC.(condition) = bandlimit(f,dim,sfreq,band);
    end
    % Save dataset
    bandstr = mat2str(band);
    fname = ['validate_GC_' bandstr 'Hz.mat'];
    fpath = fullfile(datadir, fname);
    save(fpath, 'GC')
end