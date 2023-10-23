# MVGC analysis
MVGC pipeline on ECoG times series during visual perception and recall (data is same as in [paper](https://pubmed.ncbi.nlm.nih.gov/29101322/))

## Pipeline

In folder pipeline:
* preprocessing: visualize channels, extract high-frequency broadband amplitude, epoching, prepare input time series
* modeling: State-space (SS) and VAR modeling (model order and parameters estimation)
* gc_analyses: Estimate MVGC, compare MVGC between experimental conditions and directions of information flow
* plot_scripts: Plot results from VAR/SS modeling and GC analyses

### Dependencies:

[MNE](https://mne.tools/stable/index.html) python for preprocessing (latest stable release), [MVGC2.0](https://github.com/lcbarnett/MVGC2) toolbox, python, scipy, statsmodel.




