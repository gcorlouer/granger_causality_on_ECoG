# This script prepares a condition specific time series
# for each subject and compare top-down relative to bottom up gc.
################################################################################
# Prepare condition ts
python scripts/prepare_input_ts.py;
# Compare GC between conditions
matlab -nodisplay -nosplash -nodesktop -r \
"run('scripts/startup_mat.m'); run('scripts/compare_bu_td_gc_permtest.m'); exit;";