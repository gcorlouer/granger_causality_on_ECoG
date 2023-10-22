%% Input parameters
input_parameters;
ncdt = length(conditions);
nsub = length(cohort);

%% 
mosvc = cell(nsub, ncdt);
moaic = cell(nsub, ncdt);
mobic = cell(nsub, ncdt);
mohqc = cell(nsub, ncdt);
molrt = cell(nsub, ncdt);

%% Rolling VAR

for s=1:nsub
    for c=1:ncdt 
        % Read condition specific time series
        gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition', conditions{c} , 'suffix', suffix);
        % Read conditions specific time series
        X = gc_input.X;
        fs  = gc_input.sfreq;
        [n,m,N] = size(X);
        nwin = floor((m - mw)/shift +1);
        tw = mw/fs; % Duration of time window
        aic = zeros(nwin, 1);
        bic = zeros(nwin, 1);
        hqc = zeros(nwin, 1);
        lrt = zeros(nwin, 1);
        fprintf('Rolling time window of size %4.2f s \n', tw)
        for w=1:nwin
            fprintf('Time window number %d over %d \n', w,nwin)
            o = (w-1)*shift;      % window offset
            W = X(:,o+1:o+mw,:,:);% the window
            [aic(w),bic(w),hqc(w),lrt(w)] = tsdata_to_varmo(W,momax,...
                regmode,alpha,[],[],[]);
        end

        moaic{s,c}= floor(mean(aic));
        mobic{s,c}= floor(mean(bic));
        mohqc{s,c}= floor(mean(hqc));
        molrt{s,c}= floor(mean(lrt));
        

        %% Rolling svc
        
        pf = 2 * 5;
        svc = zeros(nwin, 1);
        fprintf('Rolling time window of size %4.2f \n', tw)
        for w=1:nwin
            fprintf('Time window number %d over %d \n', w,nwin)
            o = (w-1)*shift;      % window offset
            W = X(:,o+1:o+mw,:,:);% the window
            svc(w) = tsdata_to_ssmo(W, pf , []);
        end

        mosvc{s,c} = floor(mean(svc));
    end
end