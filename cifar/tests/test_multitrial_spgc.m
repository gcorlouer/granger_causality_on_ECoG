% Multitrial functional connectivity analysis
%% Input parameters
input_parameters;
ncdt = length(condition);
nsub = length(cohort);
dataset = struct;
subject = 'DiAs';
suffix = '_condition_visual_ts.mat';
fres = 1024;
sfreq = 250;
F = cell(ncdt,1);
ncdt = 3;
%% Loop multitrial functional connectivity analysis over each subjects
for c=1:ncdt
    gc_input = read_cdt_time_series('datadir', datadir, 'subject', subject,...
            'condition',condition{c}, 'suffix', suffix);
    % Read conditions specific time series
    X = gc_input.X;
    gind = gc_input.indices;
    gi = fieldnames(gind);
    ng = length(gi);
    [n,m,N] = size(X);
    % Compute VAR model
    VAR = ts_to_var_parameters(X, 'morder', morder, 'regmode', regmode);
    V = VAR.V;
    A = VAR.A;
    disp(VAR.info)
    % Compute spectral group GC
    f = zeros(ng,ng,fres+1);
    % Estimate mvgc stat
    for i=1:ng
        for j=1:ng
            % Get indices of specific groups
            x = gind.(gi{i});
            y = gind.(gi{j});
            nx = length(x); ny = length(y); nz = n -nx - ny;
            if i==j
                continue
            else
                f(i,j,:) = var_to_smvgc(A,V,x,y,fres);
            end
        end
    end
    F{c} = f;
end

%% Plot spectral GC
lwidth = 2;
freqs = sfreqs(fres,sfreq);
iF = find(contains(fieldnames(gind), 'F'));
iR = find(contains(fieldnames(gind), 'R'));
bu = squeeze(f(iF,iR, :));
td = squeeze(f(iR, iF,:));
for c=1:ncdt
    f = F{c};
    bu = squeeze(f(iF,iR, :));
    td = squeeze(f(iR, iF,:));
    subplot(ncdt, 1,c)
    plot(freqs, bu, 'DisplayName', 'R to F', 'LineWidth',lwidth)
    hold on
    plot(freqs, td, 'DisplayName', 'F to R', 'LineWidth',lwidth);
    hold off
    ylim([0 0.2])
    xlim([0 50])
    ylabel(['pGC ' condition{c}])
    xlabel('Frequencies (Hz)')
    legend
end