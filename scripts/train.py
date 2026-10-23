#!/usr/bin/env python
"""Train, evaluate, and export the ECG heartbeat classifier.

Entry point that replaces the Colab notebook: downloads/caches MIT-BIH locally,
trains the dual-input model, converts it to INT8, selects a decision threshold
from the INT8 model's validation predictions, evaluates both the float and INT8
models on the full test set (overall + per-record), and writes all artifacts.

`--variant` picks the architecture/quantization path (see artifacts/diagnosis.md):
  baseline   build_multiscale_ptq_model + PTQ (the current deployed model)
  bn_folded  build_multiscale_bn_folded_model + PTQ
  qat        build_multiscale_bn_folded_model, trained float, then QAT fine-tuned.
             Runs under tf_keras (TF_USE_LEGACY_KERAS=1, set below before TF is imported).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# de Chazal et al., IEEE TBME 2004 inter-patient split: DS1 = train + val, DS2 = test.
# The paced records are excluded from both.
PACED_RECORDS = ["102", "104", "107", "217"]
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]
# Records 201 and 202 are the same patient, but the published split puts them in DS1 and DS2.
# Kept for comparability; 201 stays in train (never val) and test is also reported without 202.
SAME_PATIENT_EXCEPTIONS = [("201", "202")]

# Validation, carved patient-wise from DS1 (rationale in README.md, "Data split"):
# 207 is the one LBBB record (109 stays in train), 220/223 carry S and V beats,
# 115/122 are normal-only and give realistic normal volume.
VAL_RECORDS = ["115", "122", "207", "220", "223"]
TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in VAL_RECORDS]
TEST_RECORDS = list(DS2_RECORDS)

# The record that dominated the baseline's INT8 false positives on the old split (artifacts/diagnosis.md).
FOCUS_RECORD = "214"
# Same patient as train record 201; test metrics are also reported without it.
OVERLAP_RECORD = "202"
# Test beats sampled for the min_int8_levels diagnostic.
LEVELS_SAMPLE_SIZE = 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--deployment-dir", default="deployment_model")
    parser.add_argument("--variant", choices=["baseline", "bn_folded", "qat"], default="baseline")
    parser.add_argument("--normalize-rr", action="store_true",
                        help="Standardize the R-R features with train-set mean/std (saved to rr_norm.json).")
    parser.add_argument("--no-firmware-export", action="store_true",
                        help="Don't write deployment_model/*.h (for comparison runs).")
    parser.add_argument("--load-model", default=None,
                        help="Skip training and evaluate this saved float .keras model (PTQ variants only).")
    parser.add_argument("--calib-samples", type=int, default=100,
                        help="PTQ calibration-set size (ignored for qat, which needs no calibration set).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument("--max-normal-per-record-train", type=int, default=600)
    parser.add_argument("--threshold-strategy", choices=["f1", "recall_at_precision"], default="f1")
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--target-recall", type=float, default=0.95,
                        help="Val recall for the fixed-recall operating point reported in comparison_rows.csv.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.variant == "qat" and args.load_model:
        parser.error("--load-model is only supported for PTQ variants (baseline, bn_folded)")
    if args.normalize_rr and not args.no_firmware_export:
        parser.error("--normalize-rr needs --no-firmware-export: the firmware doesn't apply R-R normalization yet")
    return args


def comparison_rows(label, args, y_val, y_test, record_ids_test, preds, threshold, float_threshold,
                    tflite_kb, levels, evaluate):
    """One row per (model, threshold source): float @ INT8-val threshold, float @ its own
    val threshold, and INT8 @ INT8-val threshold. Every column is computed from the row's
    own model: `val_f1` from that model's val sweep, PR/ROC-AUC from its test probabilities,
    and the fixed-recall operating point from its own val predictions.

    `preds` maps "float"/"int8" to (val_probs, val_sweep, test_probs).
    """
    def row(model_name, threshold_from, t):
        val_probs, val_sweep, test_probs = preds[model_name]
        overall, per_record = evaluate.evaluate_predictions(y_test, test_probs, record_ids_test, t)
        (_, fp), (fn, tp) = overall["confusion_matrix"]
        fp_focus = int(per_record.loc[per_record["record_id"] == FOCUS_RECORD, "fp"].sum())

        t_recall = evaluate.threshold_at_recall(y_val, val_probs, args.target_recall)
        at_recall, _ = evaluate.evaluate_predictions(y_test, test_probs, record_ids_test, t_recall)
        is_int8 = model_name == "int8"
        return {
            "variant": label,
            "calib_samples": None if args.variant == "qat" else args.calib_samples,
            "model": model_name,
            "threshold_from": threshold_from,
            "threshold": t,
            "val_f1": float(val_sweep.loc[val_sweep["threshold"] == t, "f1"].iloc[0]),
            "test_precision": overall["precision"],
            "test_recall": overall["recall"],
            "test_f1": overall["f1"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            f"fp_rec{FOCUS_RECORD}": fp_focus,
            f"fp_excl_{FOCUS_RECORD}": fp - fp_focus,
            **evaluate.ranking_metrics(y_test, test_probs),
            "r95_threshold": t_recall,
            "r95_test_precision": at_recall["precision"],
            "r95_test_recall": at_recall["recall"],
            "r95_fp": at_recall["confusion_matrix"][0][1],
            "tflite_kb": round(tflite_kb, 2) if is_int8 else None,
            "min_int8_levels": levels if is_int8 else None,
        }

    return pd.DataFrame([
        row("float", "int8_val", threshold),
        row("float", "float_val", float_threshold),
        row("int8", "int8_val", threshold),
    ])


def main():
    args = parse_args()
    if args.variant == "qat":
        # tfmot only supports Keras 2; this must happen before TensorFlow is imported.
        os.environ["TF_USE_LEGACY_KERAS"] = "1"

    import tensorflow as tf

    from ecg import data, evaluate, export, features
    from ecg import model as ecg_model

    tf.keras.utils.set_random_seed(args.seed)
    label = args.variant + ("+rr_norm" if args.normalize_rr else "")

    print(f"Loading datasets (caches MIT-BIH records under {args.data_dir}/mitdb)...")
    X_train_spec, X_train_rr, y_train, _ = data.build_dataset(
        TRAIN_RECORDS, args.data_dir,
        max_normal_per_record=args.max_normal_per_record_train,
        augment=True, seed=args.seed,
    )
    X_val_spec, X_val_rr, y_val, record_ids_val = data.build_dataset(
        VAL_RECORDS, args.data_dir, max_normal_per_record=None, augment=False, seed=args.seed,
    )
    X_test_spec, X_test_rr, y_test, record_ids_test = data.build_dataset(
        TEST_RECORDS, args.data_dir, max_normal_per_record=None, augment=False, seed=args.seed,
    )
    print(f"Train: {len(X_train_spec)} | Val: {len(X_val_spec)} | Test: {len(X_test_spec)}")

    os.makedirs(args.artifacts_dir, exist_ok=True)

    if args.normalize_rr:
        rr_mean, rr_std = features.fit_rr_normalizer(X_train_rr)
        X_train_rr, X_val_rr, X_test_rr = (
            features.apply_rr_normalizer(X, rr_mean, rr_std) for X in (X_train_rr, X_val_rr, X_test_rr)
        )
        with open(os.path.join(args.artifacts_dir, "rr_norm.json"), "w") as f:
            json.dump({"mean": rr_mean.tolist(), "std": rr_std.tolist()}, f, indent=2)
        print(f"R-R normalizer (train): mean={rr_mean.round(4).tolist()} std={rr_std.round(4).tolist()}")

    loss = ecg_model.focal_loss(gamma=2.0, alpha=0.25)

    def fit(m, lr, epochs):
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=["accuracy"])
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
        m.fit(
            x=[X_train_spec, X_train_rr],
            y=y_train,
            validation_data=([X_val_spec, X_val_rr], y_val),
            epochs=epochs,
            batch_size=args.batch_size,
            callbacks=[early_stop_cb],
        )

    spec_shape = X_train_spec.shape[1:]
    if args.load_model:
        print(f"\nLoading saved float model {args.load_model} (no training)...")
        model = tf.keras.models.load_model(args.load_model, compile=False)
    else:
        if args.variant == "baseline":
            model = ecg_model.build_multiscale_ptq_model(spec_shape)
        else:
            model = ecg_model.build_multiscale_bn_folded_model(spec_shape, keras_mod=tf.keras)
        model.summary()
        print("\nTraining (float)...")
        fit(model, 0.001, args.epochs)
        model.save(os.path.join(args.artifacts_dir, "ecg_multiscale_model.keras"))

    if args.variant == "qat":
        from ecg import qat

        print("\nQAT fine-tuning from the trained float weights...")
        qat_model = qat.quantize_aware(model)
        fit(qat_model, args.qat_lr, args.qat_epochs)
        # .h5, not .keras: tf_keras's .keras loader drops the quantize_layer's variables.
        qat_model.save(os.path.join(args.artifacts_dir, "ecg_qat_model.h5"))
        print("\nConverting QAT model to INT8 TFLite (no calibration set needed)...")
        tflite_bytes = qat.convert_qat_to_int8_tflite(qat_model)
    else:
        print(f"\nConverting to INT8 TFLite (shuffled calibration set, {args.calib_samples} samples)...")
        tflite_bytes = export.convert_to_int8_tflite(
            model, X_train_spec, X_train_rr, num_samples=args.calib_samples, seed=args.seed
        )
    tflite_path = os.path.join(args.artifacts_dir, "model_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    print("\nSelecting threshold from the INT8 model's VALIDATION predictions...")
    val_probs_int8 = evaluate.predict_tflite(tflite_path, X_val_spec, X_val_rr)
    threshold, sweep_df = evaluate.select_threshold(
        y_val, val_probs_int8, strategy=args.threshold_strategy, min_precision=args.min_precision,
    )
    print(sweep_df.to_string(index=False))
    print(f"\nChosen threshold: {threshold:.2f} (strategy={args.threshold_strategy})")
    val_metrics_at_threshold = sweep_df.loc[sweep_df["threshold"] == threshold].iloc[0].to_dict()

    # The float model's own val-optimal threshold, so float isn't only judged at an INT8-chosen one.
    val_probs_float = model.predict([X_val_spec, X_val_rr]).flatten()
    float_threshold, float_sweep_df = evaluate.select_threshold(
        y_val, val_probs_float, strategy=args.threshold_strategy, min_precision=args.min_precision,
    )
    print(f"Float model's own val threshold: {float_threshold:.2f}")

    print("\n--- Test set evaluation: FLOAT model ---")
    test_probs_float = model.predict([X_test_spec, X_test_rr]).flatten()
    float_overall, float_per_record = evaluate.evaluate_predictions(
        y_test, test_probs_float, record_ids_test, threshold
    )
    print(json.dumps(float_overall, indent=2))
    print(float_per_record.to_string(index=False))
    float_own_overall, _ = evaluate.evaluate_predictions(
        y_test, test_probs_float, record_ids_test, float_threshold
    )

    print("\n--- Test set evaluation: INT8 model (actually deployed) ---")
    test_probs_int8 = evaluate.predict_tflite(tflite_path, X_test_spec, X_test_rr)
    int8_overall, int8_per_record = evaluate.evaluate_predictions(
        y_test, test_probs_int8, record_ids_test, threshold
    )
    print(json.dumps(int8_overall, indent=2))
    print(int8_per_record.to_string(index=False))

    model_size_kb = len(tflite_bytes) / 1024
    print(f"\nModel size (INT8 .tflite): {model_size_kb:.2f} KB")

    levels_idx = np.random.default_rng(args.seed).permutation(len(X_test_spec))[:LEVELS_SAMPLE_SIZE]
    levels, levels_tensor = evaluate.min_int8_levels(tflite_path, X_test_spec[levels_idx], X_test_rr[levels_idx])
    print(f"Min INT8 levels spanned by an intermediate tensor: {levels} ({levels_tensor})")

    with open(os.path.join(args.artifacts_dir, "threshold.json"), "w") as f:
        json.dump({
            "threshold": threshold,
            "strategy": args.threshold_strategy,
            "min_precision": args.min_precision if args.threshold_strategy == "recall_at_precision" else None,
            "validation_metrics_at_threshold": val_metrics_at_threshold,
        }, f, indent=2)

    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
        json.dump({
            "variant": label,
            "calib_samples": None if args.variant == "qat" else args.calib_samples,
            "model_size_kb": model_size_kb,
            "min_int8_levels": levels,
            "min_int8_levels_tensor": levels_tensor,
            "float": float_overall,
            "float_own_threshold": float_own_overall,
            "int8": int8_overall,
        }, f, indent=2)

    per_record_combined = pd.concat([
        float_per_record.assign(model="float"),
        int8_per_record.assign(model="int8"),
    ], ignore_index=True)
    per_record_combined.to_csv(os.path.join(args.artifacts_dir, "per_record_metrics.csv"), index=False)

    comparison_rows(
        label, args, y_val, y_test, record_ids_test,
        {"float": (val_probs_float, float_sweep_df, test_probs_float),
         "int8": (val_probs_int8, sweep_df, test_probs_int8)},
        threshold, float_threshold, model_size_kb, levels, evaluate,
    ).to_csv(os.path.join(args.artifacts_dir, "comparison_rows.csv"), index=False)

    if args.no_firmware_export:
        print(f"\nWrote artifacts to {args.artifacts_dir}/ (firmware headers not written).")
    else:
        os.makedirs(args.deployment_dir, exist_ok=True)
        export.write_c_header(tflite_bytes, os.path.join(args.deployment_dir, "model_data.h"))
        export.write_threshold_header(threshold, os.path.join(args.deployment_dir, "threshold.h"))
        print(f"\nWrote artifacts to {args.artifacts_dir}/ and {args.deployment_dir}/.")


if __name__ == "__main__":
    main()
