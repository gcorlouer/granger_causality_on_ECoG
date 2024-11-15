from src.preprocessing_lib import prepare_condition_ts
from pathlib import Path
from scipy.io import savemat

import argparse

# %% Parameters
conditions = ["Rest", "Face", "Place", "baseline"]
cohort = ["AnRa", "ArLa", "DiAs"]
signal = "hfa"

# Paths (Change before running. Run from root.)
cifar_path = Path("~", "projects", "cifar").expanduser()
data_path = cifar_path.joinpath("data")
derivatives_path = data_path.joinpath("derivatives")
result_path = cifar_path.joinpath("results")

# Signal dic
hfa_dic = {
    "suffix": "_hfb_continuous_raw.fif",
    "log_transf": True,
    "l_freq": 0.1,
    "decim": 4,
}
lfp_dic = {
    "suffix": "_bad_chans_removed_raw.fif",
    "log_transf": False,
    "l_freq": 1,
    "decim": 2,
}
if signal == "hfa":
    signal_dic = hfa_dic
elif signal == "lfp":
    signal_dic = lfp_dic

# Parser arguments
parser = argparse.ArgumentParser()
# Dataset parameters
parser.add_argument("--subject", type=str, default="DiAs")
parser.add_argument("--sfeq", type=float, default=500.0)
parser.add_argument("--stage", type=str, default="preprocessed")
parser.add_argument("--preprocessed_suffix", type=str, default=signal_dic["suffix"])
parser.add_argument("--epoch", type=bool, default=False)
parser.add_argument("--channels", type=str, default="visual_channels.csv")

# Epoching parameters
parser.add_argument("--condition", type=str, default="Stim")
parser.add_argument("--t_prestim", type=float, default=-0.5)
parser.add_argument("--t_postim", type=float, default=1.5)
parser.add_argument("--baseline", default=None)  # No baseline from MNE
parser.add_argument("--preload", default=True)
parser.add_argument("--tmin_baseline", type=float, default=-0.5)
parser.add_argument("--tmax_baseline", type=float, default=0)

# Wether to log transform the data
parser.add_argument("--log_transf", type=bool, default=signal_dic["log_transf"])
# Mode to rescale data (mean, logratio, zratio)
parser.add_argument("--mode", type=str, default="logratio")
# Pick visual chan
parser.add_argument("--pick_visual", type=bool, default=True)
# Create category specific time series
parser.add_argument("--l_freq", type=float, default=signal_dic["l_freq"])
parser.add_argument("--decim", type=float, default=signal_dic["decim"])
parser.add_argument("--tmin_crop", type=float, default=0.2)
parser.add_argument("--tmax_crop", type=float, default=1.5)
parser.add_argument("--matlab", type=bool, default=True)

args = parser.parse_args()

# Suffix
#'_bad_chans_removed_raw.fif'
# '_hfb_continuous_raw.fif'
# signal hfa/lfp
# %%


def input_ts_fname(subject, visual=True, signal="hfa"):
    if visual == True:
        fname = subject + "_condition_" + "visual_" + signal + ".mat"
    else:
        fname = subject + "_condition_" + signal + ".mat"
    return fname


# %% Prepare condition ts for each subjects
for subject in cohort:
    ts = prepare_condition_ts(
        data_path,
        subject=subject,
        stage=args.stage,
        matlab=args.matlab,
        preprocessed_suffix=args.preprocessed_suffix,
        decim=args.decim,
        epoch=args.epoch,
        t_prestim=args.t_prestim,
        t_postim=args.t_postim,
        tmin_baseline=args.tmin_baseline,
        tmax_baseline=args.tmax_baseline,
        tmin_crop=args.tmin_crop,
        tmax_crop=args.tmax_crop,
        mode=args.mode,
        log_transf=args.log_transf,
        pick_visual=args.pick_visual,
        channels=args.channels,
    )

    # %% Save condition ts as mat file
    # Save file as  _condition_visual_ts.mat or _condition_ts.mat
    fname = input_ts_fname(subject, visual=args.pick_visual, signal=signal)
    fpath = result_path.joinpath(fname)
    print(f"\n Saving in {fpath}")
    savemat(fpath, ts)
    print(f"\n Sampling rate is {500/args.decim}Hz")
    print(f"\n Stimulus is during {args.tmin_crop} and {args.tmax_crop}s")
