"""Pure signal-processing functions for ECG beat feature extraction.

No I/O here by design -- everything takes arrays in and returns arrays/lists out,
so it can be unit tested with synthetic signals instead of MIT-BIH data.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal

# Binary mapping: 0 = Normal, 1 = Abnormal
SYMBOL_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "V": 1, "E": 1,
    "A": 1, "a": 1, "J": 1, "S": 1,
}
# The two kinds of Abnormal beat, for per-type reporting (the model itself is binary).
VENTRICULAR = ("V", "E")
SUPRAVENTRICULAR = ("A", "a", "J", "S")
ABNORMAL_TYPES = {"ventricular": VENTRICULAR, "supraventricular": SUPRAVENTRICULAR}

WINDOW_SIZE = 250


def type_recall(y, symbols, probs, threshold):
    """Recall of each abnormal type at a threshold: {type: (n, detected, recall)}."""
    y, symbols, pred = np.asarray(y), np.asarray(symbols), np.asarray(probs) > threshold
    out = {}
    for name, syms in ABNORMAL_TYPES.items():
        mask = (y == 1) & np.isin(symbols, syms)
        n, detected = int(mask.sum()), int(pred[mask].sum())
        out[name] = (n, detected, detected / n if n else float("nan"))
    return out


# AAMI EC57 classes for the beats this dataset keeps (F and Q beats are dropped by extract_beats_from_record).
AAMI_CLASSES = {
    "N": tuple(s for s, label in SYMBOL_MAP.items() if label == 0),
    "S": SUPRAVENTRICULAR,
    "V": VENTRICULAR,
}


def aami_class(symbols):
    """AAMI class ("N", "S", "V") of each beat's annotation symbol; "" for symbols outside them."""
    symbols = np.asarray(symbols)
    out = np.full(symbols.shape, "", dtype=object)
    for cls, syms in AAMI_CLASSES.items():
        out[np.isin(symbols, syms)] = cls
    return out


def aami_class_metrics(symbols, abnormal_pred=None, pred_classes=None):
    """AAMI sensitivity (Se) and positive predictivity (+P) for S and V: {cls: (n, se, pp)}.

    Pass exactly one of:
    - `abnormal_pred` (binary model, bool per beat): Se is the fraction of the class's beats predicted
      abnormal. +P is NaN: an abnormal prediction doesn't name S or V, so there is no predicted-S/V set.
    - `pred_classes` (multiclass model, "N"/"S"/"V" per beat): Se = TP / n and +P = TP / predicted as the class.
    """
    if (abnormal_pred is None) == (pred_classes is None):
        raise ValueError("Pass exactly one of abnormal_pred or pred_classes")
    true = aami_class(symbols)
    out = {}
    for cls in ("S", "V"):
        is_cls = true == cls
        n = int(is_cls.sum())
        if abnormal_pred is not None:
            tp, n_pred = int(np.asarray(abnormal_pred, dtype=bool)[is_cls].sum()), None
        else:
            predicted = np.asarray(pred_classes) == cls
            tp, n_pred = int((predicted & is_cls).sum()), int(predicted.sum())
        se = tp / n if n else float("nan")
        pp = tp / n_pred if n_pred else float("nan")
        out[cls] = (n, se, pp)
    return out


def bandpass_filter(raw_signal, fs, low=0.5, high=40.0):
    nyq = 0.5 * fs
    b, a = signal.butter(2, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, raw_signal)


def compute_spectrogram(window, fs, nperseg=64, noverlap=48):
    _, _, sxx = signal.spectrogram(window, fs=fs, nperseg=nperseg, noverlap=noverlap)
    sxx_log = np.log1p(sxx * 1000)
    s_min, s_max = np.min(sxx_log), np.max(sxx_log)
    if s_max - s_min > 0:
        return (sxx_log - s_min) / (s_max - s_min)
    return sxx_log


def compute_rr_timing_features(rr_intervals, i, lookback=10):
    pre_rr = rr_intervals[i - 1]
    post_rr = rr_intervals[i]
    start_k = max(0, i - lookback)
    local_mean_rr = np.mean(rr_intervals[start_k:i])
    rr_ratio = pre_rr / local_mean_rr if local_mean_rr > 0 else 1.0
    pre_post_ratio = pre_rr / post_rr if post_rr > 0 else 1.0
    return [pre_rr, post_rr, local_mean_rr, rr_ratio, pre_post_ratio]


def fit_rr_normalizer(X_rr):
    """Per-feature mean/std of the R-R features. Fit on the training set only."""
    X_rr = np.asarray(X_rr, dtype=np.float64)
    mean = X_rr.mean(axis=0)
    std = X_rr.std(axis=0)
    std[std < 1e-12] = 1.0  # a constant feature passes through centered, not divided by ~zero
    return mean, std


def apply_rr_normalizer(X_rr, mean, std):
    return ((np.asarray(X_rr) - mean) / std).astype(np.float32)


def extract_beats_from_record(
    filtered_signal,
    fs,
    samples,
    symbols,
    record_id,
    max_normal=None,
    augment=False,
    window_size=WINDOW_SIZE,
    rng=None,
):
    """Extract (spectrogram, R-R features, label) triples for every usable beat in a record.

    `max_normal=None` means no cap on normal beats. `augment=True` adds an R-peak-jitter
    duplicate of each beat (simulating R-peak detection error) in addition to the original.
    Also returns each beat's original annotation symbol (e.g. to split recall by V vs. S).
    """
    half_window = window_size // 2
    rng = rng if rng is not None else np.random.default_rng()

    rr_intervals = np.diff(samples) / fs
    normal_count = 0

    spec_list, rr_list, y_list, record_id_list, symbol_list = [], [], [], [], []

    for i in range(1, len(samples) - 1):
        sample_idx = samples[i]
        sym = symbols[i]

        if sym not in SYMBOL_MAP:
            continue
        if sample_idx - half_window < 0 or sample_idx + half_window >= len(filtered_signal):
            continue

        target_class = SYMBOL_MAP[sym]

        if target_class == 0:
            if max_normal is not None and normal_count >= max_normal:
                continue
            normal_count += 1

        jitter_offsets = [0, int(rng.integers(-8, 9))] if augment else [0]

        for jitter in jitter_offsets:
            adj_idx = sample_idx + jitter
            if adj_idx - half_window < 0 or adj_idx + half_window >= len(filtered_signal):
                adj_idx = sample_idx

            window = filtered_signal[adj_idx - half_window: adj_idx + half_window]
            norm_spec = compute_spectrogram(window, fs)
            rr_features = compute_rr_timing_features(rr_intervals, i)

            spec_list.append(norm_spec)
            rr_list.append(rr_features)
            y_list.append(target_class)
            record_id_list.append(record_id)
            symbol_list.append(sym)

    return spec_list, rr_list, y_list, record_id_list, symbol_list
