"""Threshold selection and float/INT8 evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .export import dequantize_value, quantize_value

THRESHOLD_SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _sweep_thresholds(y_true, probs, thresholds):
    rows = []
    for t in thresholds:
        y_pred = (probs > t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], zero_division=0
        )
        rows.append({"threshold": float(t), "precision": precision[0], "recall": recall[0], "f1": f1[0]})
    return pd.DataFrame(rows)


def select_threshold(y_val, probs_val, strategy="f1", min_precision=0.90):
    """Pick a decision threshold from VALIDATION predictions only.

    strategy="f1" (default): maximize Abnormal-class F1.
    strategy="recall_at_precision": the lowest threshold with Abnormal precision >= min_precision
    (falls back to the highest-precision threshold if none clears the floor).
    """
    sweep_df = _sweep_thresholds(y_val, probs_val, THRESHOLD_SWEEP)

    if strategy == "f1":
        best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
    elif strategy == "recall_at_precision":
        eligible = sweep_df[sweep_df["precision"] >= min_precision]
        if eligible.empty:
            best_row = sweep_df.loc[sweep_df["precision"].idxmax()]
        else:
            best_row = eligible.loc[eligible["threshold"].idxmin()]
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy!r}")

    return float(best_row["threshold"]), sweep_df


def threshold_at_recall(y_val, probs_val, target_recall=0.95):
    """The highest threshold whose VALIDATION Abnormal recall (under `probs > t`) is still
    >= target_recall. Exact, not snapped to THRESHOLD_SWEEP: it sits just below the
    ceil(target_recall * n_abnormal)-th highest abnormal probability.
    """
    pos = np.sort(np.asarray(probs_val, dtype=np.float64)[np.asarray(y_val) == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("no abnormal beats in the validation labels")
    m = int(np.ceil(target_recall * len(pos)))
    return float(np.nextafter(pos[m - 1], -np.inf))


def _tensor_detail_by_name(details, name):
    for d in details:
        if name in d["name"]:
            return d
    raise KeyError(f"No tensor with name containing {name!r} found among {[d['name'] for d in details]}")


def _bind_inputs(input_details, n_rr_features=5):
    """Return (spec_detail, rr_detail). Binds by name substring when the names survived
    conversion (PTQ models); otherwise falls back to element count, the same rule the
    firmware uses (QAT models' inputs are named after tfmot's quantize_layer, not the
    Keras inputs).
    """
    try:
        return (
            _tensor_detail_by_name(input_details, "spectrogram_input"),
            _tensor_detail_by_name(input_details, "rr_input"),
        )
    except KeyError:
        pass
    rr = [d for d in input_details if int(np.prod(d["shape"])) == n_rr_features]
    spec = [d for d in input_details if int(np.prod(d["shape"])) != n_rr_features]
    if len(input_details) != 2 or len(rr) != 1 or len(spec) != 1:
        raise KeyError(f"Can't bind spectrogram/R-R inputs among {[(d['name'], d['shape']) for d in input_details]}")
    return spec[0], rr[0]


def predict_tflite(tflite_path, X_spec, X_rr):
    """Run the INT8 .tflite model over a dataset, returning dequantized probabilities."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    spec_detail, rr_detail = _bind_inputs(input_details)
    out_detail = output_details[0]

    spec_scale, spec_zp = spec_detail["quantization"]
    rr_scale, rr_zp = rr_detail["quantization"]
    out_scale, out_zp = out_detail["quantization"]

    probs = np.empty(len(X_spec), dtype=np.float32)

    for i in range(len(X_spec)):
        spec_q = quantize_value(X_spec[i:i + 1], spec_scale, spec_zp)
        rr_q = quantize_value(X_rr[i:i + 1], rr_scale, rr_zp)

        interpreter.set_tensor(spec_detail["index"], spec_q)
        interpreter.set_tensor(rr_detail["index"], rr_q)
        interpreter.invoke()

        raw_output = interpreter.get_tensor(out_detail["index"])[0, 0]
        probs[i] = dequantize_value(raw_output, out_scale, out_zp)

    return probs


def min_int8_levels(tflite_path, X_spec, X_rr):
    """Smallest number of INT8 levels any op-output tensor actually spans over the given
    beats (observed max - min of its int8 values), and that tensor's name. This is the
    diagnostic from artifacts/diagnosis.md: the baseline's pre-BN tensors span ~4-5.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    spec_detail, rr_detail = _bind_inputs(interpreter.get_input_details())
    spec_scale, spec_zp = spec_detail["quantization"]
    rr_scale, rr_zp = rr_detail["quantization"]

    tensor_names = {d["index"]: d["name"] for d in interpreter.get_tensor_details()}
    tensor_dtypes = {d["index"]: d["dtype"] for d in interpreter.get_tensor_details()}
    op_outputs = sorted({
        i for op in interpreter._get_ops_details() for i in op["outputs"]
        if i >= 0 and tensor_dtypes[i] == np.int8
    })

    lo = {i: 127 for i in op_outputs}
    hi = {i: -128 for i in op_outputs}
    for n in range(len(X_spec)):
        interpreter.set_tensor(spec_detail["index"], quantize_value(X_spec[n:n + 1], spec_scale, spec_zp))
        interpreter.set_tensor(rr_detail["index"], quantize_value(X_rr[n:n + 1], rr_scale, rr_zp))
        interpreter.invoke()
        for i in op_outputs:
            t = interpreter.get_tensor(i)
            lo[i] = min(lo[i], int(t.min()))
            hi[i] = max(hi[i], int(t.max()))

    worst = min(op_outputs, key=lambda i: hi[i] - lo[i])
    return hi[worst] - lo[worst], tensor_names[worst]


def ranking_metrics(y_true, probs):
    """Threshold-independent metrics for the Abnormal class. PR-AUC is average precision."""
    return {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
    }


def evaluate_predictions(y_true, probs, record_ids, threshold):
    """Overall + per-record metrics for the Abnormal class at a fixed threshold.

    Per-record precision/recall/F1 use zero_division=0; n_abnormal is included so a
    genuine 0 can be told apart from a record where the metric is undefined.
    """
    y_true = np.asarray(y_true)
    record_ids = np.asarray(record_ids)
    y_pred = (probs > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    overall = {
        "threshold": float(threshold),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "confusion_matrix": cm.tolist(),
        "n_beats": int(len(y_true)),
    }

    rows = []
    for rec in sorted(set(record_ids.tolist())):
        mask = record_ids == rec
        yt, yp = y_true[mask], y_pred[mask]
        rec_precision, rec_recall, rec_f1, _ = precision_recall_fscore_support(
            yt, yp, labels=[1], zero_division=0
        )
        rec_cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = rec_cm.ravel()
        n_beats = int(mask.sum())
        rows.append({
            "record_id": rec,
            "n_beats": n_beats,
            "n_abnormal": int((yt == 1).sum()),
            "n_normal": int((yt == 0).sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "accuracy": float((tp + tn) / n_beats) if n_beats else 0.0,
            "precision": float(rec_precision[0]),
            "recall": float(rec_recall[0]),
            "f1": float(rec_f1[0]),
        })

    return overall, pd.DataFrame(rows)
