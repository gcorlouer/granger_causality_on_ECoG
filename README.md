### MVGC analysis
The "analysis" folder contains script to run mvgc analysis on preprocessed HFA and iEEG multitrial time series in distinct conditions. The main scripts are:
* pcgc_analysis.m is a main analysis script which run pairwise conditional Granger causality analysis on preprocessed time series and save results into a .m file to plot results in python
* spectral_pcgc_analysis.m  is a main analysis script which estimates spectral pairwise conditional GC following Geweke formulation (for more detail see mvgc paper)
* ts_data_to_* : function scripts used in main analysis script to estimate mutual information, VAR model order, VAR parameters and pairwise conditional GC.