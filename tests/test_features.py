import numpy as np
import scipy.signal as signal

from ecg.features import (
    apply_rr_normalizer,
    compute_rr_timing_features,
    compute_spectrogram,
    extract_beats_from_record,
    fit_rr_normalizer,
)


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


def test_rr_normalizer_standardizes_train_and_reuses_train_stats():
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0.2, 2.0, size=(200, 5))
    X_other = rng.uniform(0.2, 2.0, size=(50, 5))

    mean, std = fit_rr_normalizer(X_train)
    Z_train = apply_rr_normalizer(X_train, mean, std)

    assert Z_train.dtype == np.float32
    np.testing.assert_allclose(Z_train.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(Z_train.std(axis=0), 1.0, atol=1e-5)
    # Other splits use the train stats, and the transform inverts exactly.
    np.testing.assert_allclose(apply_rr_normalizer(X_other, mean, std) * std + mean, X_other, rtol=1e-5)


def test_rr_normalizer_constant_feature_does_not_divide_by_zero():
    X = np.column_stack([np.full(10, 0.8), np.linspace(0.5, 1.5, 10)])

    mean, std = fit_rr_normalizer(X)
    Z = apply_rr_normalizer(X, mean, std)

    assert std[0] == 1.0
    assert np.all(np.isfinite(Z))
    np.testing.assert_allclose(Z[:, 0], 0.0, atol=1e-6)


def test_extract_beats_returns_symbols_aligned_with_labels_and_augmentation():
    fs = 360
    signal = np.sin(2 * np.pi * 1.2 * np.arange(5 * fs) / fs)
    samples = np.array([10, 300, 600, 900, 1200, 1700])
    symbols = ["N", "V", "N", "A", "+", "N"]  # "+" isn't a beat; first/last are skipped

    _, _, y, record_ids, syms = extract_beats_from_record(
        signal, fs, samples, symbols, "r", augment=True, rng=np.random.default_rng(0)
    )

    assert syms == ["V", "V", "N", "N", "A", "A"]  # each beat plus its jitter duplicate
    assert y == [1, 1, 0, 0, 1, 1]
    assert len(record_ids) == len(syms)
