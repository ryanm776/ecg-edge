import numpy as np
import scipy.signal as signal

from ecg.features import compute_rr_timing_features, compute_spectrogram


def test_compute_spectrogram_shape_matches_scipy_and_is_normalized():
    fs = 360
    window = np.sin(2 * np.pi * 5 * np.arange(250) / fs).astype(np.float32)

    spec = compute_spectrogram(window, fs, nperseg=64, noverlap=48)

    f, t, _ = signal.spectrogram(window, fs=fs, nperseg=64, noverlap=48)
    assert spec.shape == (len(f), len(t))
    assert np.min(spec) >= 0.0
    assert np.max(spec) <= 1.0


def test_compute_spectrogram_handles_flat_window_without_nan():
    flat_window = np.zeros(250, dtype=np.float32)

    spec = compute_spectrogram(flat_window, fs=360)

    assert not np.any(np.isnan(spec))


def test_compute_rr_timing_features_values():
    rr_intervals = np.array([0.8, 0.8, 0.8, 0.8, 1.6])

    pre_rr, post_rr, local_mean_rr, rr_ratio, pre_post_ratio = compute_rr_timing_features(
        rr_intervals, i=4, lookback=10
    )

    assert pre_rr == rr_intervals[3]
    assert post_rr == rr_intervals[4]
    assert local_mean_rr == np.mean(rr_intervals[0:4])
    assert rr_ratio == pre_rr / local_mean_rr
    assert pre_post_ratio == pre_rr / post_rr


def test_compute_rr_timing_features_respects_lookback_window():
    rr_intervals = np.array([10.0, 10.0, 0.5, 0.5, 0.5, 0.8])

    # lookback=2 should only average the two most recent intervals before i=5
    _, _, local_mean_rr, _, _ = compute_rr_timing_features(rr_intervals, i=5, lookback=2)

    assert local_mean_rr == np.mean(rr_intervals[3:5])
