
%% PLOT FS brain:
close all

S_brain=struct;
S_brain.plotsurf='pial';
S_brain.layout='compact';
S_brain.surfacealpha=1;
S_brain.meshdir='~/projects/cifar/data/source_data/iEEG_10';
S_brain.ch_list= array2table(nan(1,9),'VariableNames',{'subjid','ch_label','ch_sig','ch_hemi','ch_eCrd','ch_nodeIDX','ch_handle','dist_to_srf','aparcTag'});
S_brain.ch_list(1,:)= [];
disp_brain='fsaverage'; % alternative - use an average template: 'fsaverage'
[S_brain,H,SUMAsrf] = plot_FS_brain_master(disp_brain,S_brain);
%% PLOT electrodes
initials='DiAs';
elocDir=fullfile(S_brain.meshdir,'subjects',initials,'brain');
electrodes = load(fullfile(elocDir,'SUMAprojectedElectrodes.mat'));
ch_labels=[SUMAprojectedElectrodes.elecNames];

%ePlot=stats_table.Properties.RowNames(ismember(stats_table.Channel,visual_responsive_ROI));
ePlot=ch_labels;
ePlot1=ch_labels(multiStrFind(ch_labels,'Grid')|multiStrFind(ch_labels,'Grd'));
ePlot2=setdiff(ch_labels,ePlot1); 
eSize=2; % radius in mm
eColor1=[0 0 0];
eColor2=[1 0 0];
textFlag=1;
%was functino plot_Fs_electrode_example_code() before 
S_brain=plot_Fs_electrode_master(initials,H,S_brain,SUMAsrf,elocDir,ePlot1,eColor1,eSize,textFlag,'gouraud','top');
S_brain=plot_Fs_electrode_master(initials,H,S_brain,SUMAsrf,elocDir,ePlot2,eColor1,eSize,textFlag,'gouraud','top'); 
S_brain=plot_Fs_electrode_master(initials,H,S_brain,SUMAsrf,elocDir,{'Grid16'},eColor2,2.5,textFlag,'gouraud','top'); 
%% Save figure: 

   

  
     
