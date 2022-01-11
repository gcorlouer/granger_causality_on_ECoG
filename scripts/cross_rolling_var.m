cohort = {'AnRa',  'ArLa',  'BeFe',  'DiAs',  'JuRo'};
condition = {'rest', 'face', 'place'};
field = {'subject', 'condition', 'time', 'aic', 'bic', 'hqc', 'lrt', 'rho'};
value = {};
nsub = length(cohort);
dataset = struct;
mw = 80;

for i=1:nsub
    sub_id = cohort{i};
    disp(['VAR estimation subject ' sub_id])
    sliding_var
    for c=1:ncdt
        for w=1:nwin
            dataset(w,c,i).time = win_time(w,mw);
            dataset(w,c,i).condition = condition{c};
            dataset(w,c,i).subject = sub_id;
            dataset(w,c,i).aic = moaic(c,w);
            dataset(w,c,i).bic = mobic(c,w);
            dataset(w,c,i).hqc = mohqc(c,w);
            dataset(w,c,i).lrt = molrt(c,w);
            dataset(w,c,i).rho = rho(c,w);
        end
    end
end

lenData = numel(dataset);
dataset = reshape(dataset, lenData, 1);

%% Save dataset

df = struct2table(dataset);
fname = 'sliding_var_estimation.csv';
fpath = fullfile(datadir, fname);
writetable(df, fpath)