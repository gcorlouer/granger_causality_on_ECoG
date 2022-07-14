import argparse 

from src.time_frequency import plot_tf
from pathlib import Path

fpath = Path('..','data_transfer')

plot_tf(fpath, subject='DiAs', vmax=25)