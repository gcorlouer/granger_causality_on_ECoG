from libs.preprocessing_lib import EcogReader, HfbExtractor
from libs.input_config import args

import matplotlib.pyplot as plt

# %%

# Read bad chan removed ecog
ecog = EcogReader(
    args.data_path,
    subject=args.subject,
    stage=args.stage,
    preprocessed_suffix=args.preprocessed_suffix,
)
raw = ecog.read_ecog()

hfb = HfbExtractor(
    l_freq=args.l_freq,
    nband=args.nband,
    band_size=args.band_size,
    l_trans_bandwidth=args.l_trans_bandwidth,
    h_trans_bandwidth=args.h_trans_bandwidth,
    filter_length=args.filter_length,
    phase=args.phase,
    fir_window=args.fir_window,
)

envelope = hfb.extract_envelope(raw)

# %% Test narrow band envelope extraction


def test_extract_envelope(channel):
    # Take one channel to check extraction
    ecog = EcogReader(
        args.data_path,
        subject=args.subject,
        stage=args.stage,
        preprocessed_suffix=args.preprocessed_suffix,
    )
    raw = ecog.read_ecog()
    raw = raw.copy().pick_channels(channel).crop(tmin=10, tmax=15)
    raw_band = raw.copy().filter(
        l_freq=args.l_freq,
        h_freq=args.l_freq + args.band_size,
        phase=args.phase,
        filter_length=args.filter_length,
        l_trans_bandwidth=args.l_trans_bandwidth,
        h_trans_bandwidth=args.h_trans_bandwidth,
        fir_window=args.fir_window,
    )
    time = raw_band.times
    raw_band = raw_band.get_data()
    hfb = HfbExtractor(
        l_freq=args.l_freq,
        nband=args.nband,
        band_size=args.band_size,
        l_trans_bandwidth=args.l_trans_bandwidth,
        h_trans_bandwidth=args.h_trans_bandwidth,
        filter_length=args.filter_length,
        phase=args.phase,
        fir_window=args.fir_window,
    )

    envelope = hfb.extract_envelope(raw)
    plt.plot(time, raw_band.T)
    plt.plot(time, envelope.T)


# %matplotlib qt
channel = ["LTo1-LTo2"]
test_extract_envelope(channel)

# %% Test Frequency bands partition


def test_freq_bands():
    hfb = HfbExtractor(
        l_freq=args.l_freq,
        nband=args.nband,
        band_size=args.band_size,
        l_trans_bandwidth=args.l_trans_bandwidth,
        h_trans_bandwidth=args.h_trans_bandwidth,
        filter_length=args.filter_length,
        phase=args.phase,
        fir_window=args.fir_window,
    )
    bands = hfb.freq_bands()
    print(f"Bands span {bands}")


test_freq_bands()
# %% Test Mean normalisation


def test_mean_normalise():
    reader = EcogReader(
        args.data_path,
        subject=args.subject,
        stage=args.stage,
        preprocessed_suffix=args.preprocessed_suffix,
    )
    raw = reader.read_ecog()
    extractor = HfbExtractor(
        l_freq=args.l_freq,
        nband=args.nband,
        band_size=args.band_size,
        l_trans_bandwidth=args.l_trans_bandwidth,
        h_trans_bandwidth=args.h_trans_bandwidth,
        filter_length=args.filter_length,
        phase=args.phase,
        fir_window=args.fir_window,
    )
    envelope = extractor.extract_envelope(raw)
    envelope_norm = extractor.mean_normalise(envelope)
    print(f"Shape of normalised narrow envelope is {envelope_norm.shape}")


test_mean_normalise()

# %% Test hfb extraction
# Note these channels only work for subject DiAs
chans = ["LTo1-LTo2", "LTo5-LTo6", "LGRD58-LGRD59", "LGRD60-LGRD61"]


def test_extract_hfb(chans):
    reader = EcogReader(
        args.data_path,
        subject=args.subject,
        stage=args.stage,
        preprocessed_suffix=args.preprocessed_suffix,
    )
    raw = reader.read_ecog()
    extractor = HfbExtractor(
        l_freq=args.l_freq,
        nband=args.nband,
        band_size=args.band_size,
        l_trans_bandwidth=args.l_trans_bandwidth,
        h_trans_bandwidth=args.h_trans_bandwidth,
        filter_length=args.filter_length,
        phase=args.phase,
        fir_window=args.fir_window,
    )
    hfb = extractor.extract_hfb(raw)
    hfb = hfb.pick_channels(chans)
    hfb.plot()


test_extract_hfb(chans)
