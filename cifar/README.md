### MVGC analysis
This project runs MVGC analysis on ECoG time series. The pipelines runs as follows:
* Remove bad channels and concatenate runs
* Classify visually responsive channels
* Classify face and retinotopic channels
* Prepare condition specific time series
* Run pairwise GC analysis using MVGC 2.0 toolbox
* Run rolling groupwise GC analysis using MVGC 2.0 toolbox
* HFB_process scripts contains the core functions to preprocess the ecog data
* prepare_condition_* scripts Prepare input time series for MVGC analysis
* ts_data_to_* : function scripts used in main analysis script to estimate mutual information, VAR model order, VAR parameters and pairwise conditional GC.
* cross_sliding_* scripts run rolling window mvgc accross subjects