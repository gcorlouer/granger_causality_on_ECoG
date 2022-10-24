# This script prepares a condition specific time series
# for each subject and compare GC between conditions.
################################################################################
# Prepare condition ts
python scripts/prepare_input_ts.py;
# Compare GC between conditions
matlab -nodisplay -nosplash -nodesktop -r \
"run('scripts/startup_mat.m'); run('scripts/compare_condition_GC.m'); exit;";
