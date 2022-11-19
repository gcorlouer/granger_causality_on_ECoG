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
fname = 'rolling_GC_two_chan.mat';
p = 10;  % suggested model order
nfreq = 1024;
mosvc = 39; % suggested state space model order
dim = 3;
mw = 80;
shift = 10;
GC = struct;

for ib=1:nband
band = bands.(band_names{ib});
bandstr = mat2str(band);
    for c=1:ncdt
            condition = conditions{c};
            % Read condition specific time series
            gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
                'condition',condition, 'suffix', suffix);
            X = gc_input.X;
            time = gc_input.time;
            [n,nobs,N] = size(X);
            nwin = floor((nobs - mw)/shift +1);
            F = zeros(n,n,nwin);
            sfreq = gc_input.sfreq;
            pf = 2*p;
            for w=1:nwin
                fprintf('Time window number %d over %d \n', w,nwin)
                % window offset
                o = (w-1)*shift;   
                % the window
                W = X(:,o+1:o+mw,:);
                [A,C,K,V,Z,E] = tsdata_to_ss(W,pf,mosvc);
                f = ss_to_spwcgc(A,C,K,V,nfreq);
                F(:,:,w) = bandlimit(f,dim,sfreq,band);
            end
            %% Sample to time
            time_offset = shift/sfreq;
            win_time = zeros(nwin,mw);
            for w=1:nwin
                o = (w-1)*shift; 
                win_time(w,:) = time(o+1:o+mw);
            end
            GC.(band_names{ib}).(condition).('F') = F;
            GC.(band_names{ib}).(condition).('time') = win_time;
    end
end
fpath = fullfile(datadir, fname);
save(fpath, 'GC')