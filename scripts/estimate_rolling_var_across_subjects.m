cohort = {'AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi'};
conditions = {'Rest', 'Face', 'Place'};
field = {'subject', 'condition', 'time', 'aic', 'bic', 'hqc', 'lrt', 'rho'};
value = {};
nsub = length(cohort);
dataset = struct;

for i=1:nsub
    subject = cohort{i};
    disp(['VAR estimation subject ' subject])
    rolling_var
    for c=1:ncdt
        condition = conditions{c};
        for w=1:nwin
            dataset(w,c,i).time = win_time(w,mw);
            dataset(w,c,i).condition = condition;
            dataset(w,c,i).subject = subject;
            dataset(w,c,i).aic = moaic{c}(w);
            dataset(w,c,i).bic = mobic{c}(w);
            dataset(w,c,i).hqc = mohqc{c}(w);
            dataset(w,c,i).lrt = molrt{c}(w);
            dataset(w,c,i).rho = rho{c}(w);
        end
    end
end

lenData = numel(dataset);
dataset = reshape(dataset, lenData, 1);

%% Save dataset

df = struct2table(dataset);
fname = 'rolling_var_estimation.csv';
fpath = fullfile(datadir, fname);
writetable(df, fpath)