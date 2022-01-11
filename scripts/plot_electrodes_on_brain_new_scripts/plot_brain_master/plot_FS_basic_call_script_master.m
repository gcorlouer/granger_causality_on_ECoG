
ieeg_dir = fullfile('~', 'projects','CIFAR', 'CIFAR_data', 'iEEG_10');
subject_dir = fullfile(ieeg_dir, 'subjects');
%% PLOT FS barin:
close all

S_brain=struct;
S_brain.plotsurf='pial';
S_brain.layout='full';
S_brain.surfacealpha=1;
S_brain.ch_list= array2table(nan(1,9),'VariableNames',{'subjid','ch_label','ch_sig','ch_hemi','ch_eCrd','ch_nodeIDX','ch_handle','dist_to_srf','aparcTag'});
S_brain.ch_list(1,:)= [];
disp_brain='DiAs'; % alternative - use an average template: 'fsaverage'
S_brain.meshdir=fullfile(subject_dir, disp_brain, 'brain');
[S_brain,H,SUMAsrf] = plot_FS_brain_master(disp_brain,S_brain);


%% PLOT electrodes
sub_id='DiAs';
brain_dir=fullfile(subject_dir,sub_id, 'brain');
load(fullfile(brain_dir,'SUMAprojectedElectrodes.mat'))
     ch_labels=[SUMAprojectedElectrodes.elecNames];
     ePlot=ch_labels;
     ePlot1=ch_labels(multiStrFind(ch_labels,'Grid')|multiStrFind(ch_labels,'Grd'));
     ePlot2=setdiff(ch_labels,ePlot1);
     eSize=2; % radius in mm
     eColor1=[0 0 0];
     eColor2=[1 0 0];
     textFlag=1;

     S_brain=plot_Fs_electrode_master(sub_id,H,S_brain,SUMAsrf,brain_dir,ePlot1,eColor1,eSize,textFlag,'gouraud','top');
     S_brain=plot_Fs_electrode_master(sub_id,H,S_brain,SUMAsrf,brain_dir,ePlot2,eColor1,eSize,textFlag,'gouraud','top');
     S_brain=plot_Fs_electrode_master(sub_id,H,S_brain,SUMAsrf,brain_dir,{'Grid16'},eColor2,2.5,textFlag,'gouraud','top');
  %% Save figure:

     figname=[sub_id '_on_blank_surface'];
     outdir=['/home/guime/projects/CIFAR/figures/brains'];
     if ~exist(outdir,'dir')
         mkdir(outdir);
         disp('Creating Output Directory...')
     end
     %saveas(gcf,fullfile(outdir,[figname '.fig']))
     export_fig(fullfile(outdir,figname),'-jpg','-r100')
