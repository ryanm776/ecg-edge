import numpy as np
import scipy.signal as signal

import pytest

from ecg.features import (
    BEAT_SYMBOLS,
    SYMBOL_MAP,
    aami_class,
    aami_class_metrics,
    apply_rr_normalizer,
    beat_annotation_mask,
    compute_rr_timing_features,
    compute_spectrogram,
    extract_beats_from_record,
    fit_rr_normalizer,
    rr_quantile_ratio,
    type_recall,
)


def test_aami_classes_cover_the_binary_symbol_map():
    symbols = np.array(list(SYMBOL_MAP) + ["F", "Q"], dtype=object)
    classes = aami_class(symbols)
    for sym, cls in zip(symbols, classes):
        if sym in SYMBOL_MAP:
            assert (cls == "N") == (SYMBOL_MAP[sym] == 0)
        else:
            assert cls == ""


def test_aami_binary_se_equals_type_recall_and_pp_is_undefined():
    rng = np.random.default_rng(0)
    symbols = rng.choice(np.array(["N", "L", "V", "E", "A", "a", "J"], dtype=object), 200)
    y = np.array([SYMBOL_MAP[s] for s in symbols])
    probs = rng.uniform(size=200)

    m = aami_class_metrics(symbols, abnormal_pred=probs > 0.5)
    by_type = type_recall(y, symbols, probs, 0.5)

    assert m["V"][:2] == (by_type["ventricular"][0], by_type["ventricular"][2])
    assert m["S"][:2] == (by_type["supraventricular"][0], by_type["supraventricular"][2])
    assert np.isnan(m["V"][2]) and np.isnan(m["S"][2])


def test_aami_multiclass_se_and_pp_from_confusion_counts():
    #            true: N  N  N  S  S  S  V  V
    symbols = np.array(["N", "R", "N", "A", "J", "A", "V", "E"], dtype=object)
    pred = np.array(["N", "S", "V", "S", "S", "V", "V", "N"], dtype=object)

    m = aami_class_metrics(symbols, pred_classes=pred)

    # S: 2 of 3 true S predicted S; 3 predicted S, 2 correct.
    assert m["S"] == (3, 2 / 3, 2 / 3)
    # V: 1 of 2 true V predicted V; 3 predicted V, 1 correct.
    assert m["V"] == (2, 1 / 2, 1 / 3)


def test_aami_class_metrics_needs_exactly_one_prediction_kind():
    with pytest.raises(ValueError):
        aami_class_metrics(["N"])
    with pytest.raises(ValueError):
        aami_class_metrics(["N"], abnormal_pred=[False], pred_classes=["N"])


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


def test_rr_quantile_ratio_uses_the_mean_window():
    rr_intervals = np.array([10.0, 0.8, 0.4, 0.8, 0.4, 0.8, 0.5])
    i, lookback = 6, 5
    window = rr_intervals[1:6]  # the same slice compute_rr_timing_features averages

    assert rr_quantile_ratio(rr_intervals, i, lookback, q=0.5) == rr_intervals[5] / np.quantile(window, 0.5)
    assert rr_quantile_ratio(rr_intervals, i, lookback, q=1.0) == rr_intervals[5] / window.max()
    assert rr_quantile_ratio(rr_intervals, i, lookback, q=0.9) == rr_intervals[5] / np.quantile(window, 0.9)


def test_rr_quantile_ratio_median_is_masked_when_most_beats_are_premature():
    # 7 premature (0.5 s) and 3 sinus (1.0 s) intervals before a premature beat.
    rr_intervals = np.array([0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5])

    assert rr_quantile_ratio(rr_intervals, 10, lookback=10, q=0.5) == 1.0  # looks on-time
    assert rr_quantile_ratio(rr_intervals, 10, lookback=10, q=0.9) == 0.5
    assert rr_quantile_ratio(rr_intervals, 10, lookback=10, q=1.0) == 0.5


def test_beat_annotation_mask_covers_every_dataset_symbol_and_no_rhythm_codes():
    assert set(SYMBOL_MAP) | {"F", "Q"} <= BEAT_SYMBOLS
    mask = beat_annotation_mask(["N", "+", "A", "~", "|", "V", "x", "!", '"', "F"])
    assert mask.tolist() == [True, False, True, False, False, True, False, False, False, True]


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
