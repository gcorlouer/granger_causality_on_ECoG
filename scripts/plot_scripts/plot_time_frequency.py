from src.time_frequency import plot_tf
from pathlib import Path

#%%
cifar_path = Path('~','projects','cifar').expanduser()
fpath = cifar_path.joinpath('data_transfer')
fpath = fpath.joinpath("tf_power_dataframe.pkl")

#%%

plot_tf(fpath, subject='DiAs', vmax=25)