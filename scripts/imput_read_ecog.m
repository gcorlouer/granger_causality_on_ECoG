%% Read cifar data into EEG structure. 
function EEG = cf_read_data(path, options)
    arguments
        path   string 
        options.sub_id  string  = "DiAs"
        options.proc    string  = "bipolar_montage" % or bipolar_montage or preproc
        options.processing_stage   string  = "bad_channels_removed.set"
        options.condition string = "rest_baseline" % or stimuli
        options.run int32 = 1
    end

sub_id = options.sub_id;
proc = options.proc;
processing_stage = options.processing_stage;
condition = options.condition;
run = options.run;
% For original data needs to write function for each run and condition
if proc=="raw_signal" || proc=="bipolar_montage"
    fname = cf_write_fname("proc", proc, "condition", condition, "run", run);
elseif proc=="preproc"
    % filename is name of subject + processing stage
    fname = [sub_id,  processing_stage];
    fname = strjoin(fname, '_');
end
path = fullfile(path, sub_id, "EEGLAB_datasets", proc);
% Convert to char otherwise EEGLAB pop_load dataset yields error
fname = convertStringsToChars(fname);
path = convertStringsToChars(path);
EEG = pop_loadset(fname, path);
end

function fname = cf_write_fname(options)
    arguments 
        options.sub_id  string  = "DiAs"
        options.proc    string  = "raw_signal" % or bipolar_montage
        options.condition string = "rest_baseline" % or stimuli
        options.run int32 = 1
    end
    
sub_id = options.sub_id ;
proc = options.proc;
condition = options.condition;
run = options.run;

if proc=="bipolar_montage"
    fname = [sub_id, "freerecall", condition, int2str(run),"preprocessed_BP_montage.set"];
else
    fname = [sub_id, "freerecall", condition, int2str(run), "preprocessed.set"];
end
fname = strjoin(fname, '_');
% Convert to char otherwise EEGLAB pop_load dataset yields error
fname = convertStringsToChars(fname);
end
