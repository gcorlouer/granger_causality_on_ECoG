from libs.preprocessing_lib import Ecog
from libs.input_config import args

# %% Some unit tests

path = args.path
stage = args.stage
cohort = args.cohort


def test_chan_number():
    """
    Just pring number of chans in each channels, should expect different
    channels in each subjects
    """
    for subject in cohort:
        ecog = Ecog(
            path,
            stage=stage,
            subject=subject,
            preprocessed_suffix="_BP_montage_HFB_raw.fif",
            preload=True,
            epoch=False,
        )
        raw = ecog.read_ecog(run=1, condition="stimuli")
        print(f"Subject {subject} has {len(raw.info['ch_names'])} chans")


def test_condition():
    """
    Check if the number of channels is the same accross condition
    """
    for subject in cohort:
        ecog = Ecog(
            path,
            stage=stage,
            subject=subject,
            preprocessed_suffix="_BP_montage_HFB_raw.fif",
            preload=True,
            epoch=False,
        )
        stimuli = ecog.read_ecog(run=1, condition="stimuli")
        rest = ecog.read_ecog(run=1, condition="rest_baseline")
        nchan_stim = len(stimuli.info["ch_names"])
        nchan_rest = len(rest.info["ch_names"])
        assert nchan_stim == nchan_rest


# %% Test concatenation


def test_concatenate():
    ecog = Ecog(path, stage=stage, subject="DiAs", preload=True, epoch=False)
    stimuli = ecog.concatenate_run(condition="stimuli")
    rest = ecog.concatenate_run(condition="rest_baseline")
    raw = ecog.concatenate_condition()
    # Test if durations are compatible
    assert len(raw) == len(rest) + len(stimuli)


# %% Plot concatenation
# %matplotlib auto

ecog = Ecog(path, stage=stage, subject="DiAs", preload=True, epoch=False)
raw = ecog.concatenate_condition()

raw.plot(scalings=1e-4)
