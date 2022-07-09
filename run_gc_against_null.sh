# This script prepares a condition specific time series
# for each subject and compare GC against the null.
################################################################################
# Input parameters
python src/input_config.py;
# Prepare condition ts
python scripts/prepare_condition_specific_ts.py;
# Compare GC between conditions
matlab -nodisplay -nosplash -nodesktop -r \
"run('scripts/startup_mat.m'); run('scripts/compare_null_against_gc.m'); exit;";