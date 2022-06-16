sublist ={'AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo',  'NeLa',  'SoGi'};
for i = 1: size(sublist,2)
    subject = sublist{i};
    anatpath = fullfile(cfsubdir, subject, 'brain');
    elecfile = 'elecinfo.mat';
    elecpath = fullfile(anatpath, elecfile);
    sm = get_SUMA_map(subject);

    coord = sm.coord.afniXYZ;
    electrode_name = sm.elecNames;
    isdepth = sm.isDepthElectrode;
    ROI_DK = sm.aparcaseg.bestLabel.labels;
    hemisphere = sm.hemisphere;
    Brodman = sm.Brodmann.nearestROIname;

    fprintf('saving elecinfo file ... ');
    save(elecpath, 'coord','electrode_name','isdepth','ROI_DK','hemisphere','Brodman')
    fprintf('done\n\n');
end
