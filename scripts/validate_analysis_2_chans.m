%% In this script we run the pipeline on 2 chans to validate it
input_parameters
conditions = {'Rest', 'Face', 'Place'};
ncdt = length(conditions);
suffix = ['_condition_two_chans_' signal '.mat'];
p = 10;  % suggested model order
nfreq = 1024;
mosvc = 39; % suggested state space model order
dim = 3;
GC = struct;

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
fname = ['two_chans_magnitude_GC_' bandstr 'Hz.mat'];
fpath = fullfile(datadir, fname);
save(fpath, 'GC')
