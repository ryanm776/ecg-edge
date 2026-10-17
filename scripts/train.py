#!/usr/bin/env python
"""Train, evaluate, and export the ECG heartbeat classifier.

Entry point that replaces the Colab notebook: downloads/caches MIT-BIH locally,
trains the dual-input model, converts it to INT8, selects a decision threshold
from the INT8 model's validation predictions, evaluates both the float and INT8
models on the full test set (overall + per-record), and writes all artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import keras
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecg import data, evaluate, export
from ecg import model as ecg_model

TRAIN_RECORDS = [
    "101", "106", "108", "112", "114", "115", "118", "119", "200", "201",
    "203", "205", "208", "209", "210", "212", "213", "215", "219", "221",
    "222", "223",
]
VAL_RECORDS = ["103", "109", "111", "124", "220", "230", "231", "232"]
TEST_RECORDS = ["100", "105", "113", "121", "123", "202", "214", "233"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--deployment-dir", default="deployment_model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-normal-per-record-train", type=int, default=600)
    parser.add_argument("--threshold-strategy", choices=["f1", "recall_at_precision"], default="f1")
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

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

    spec_shape = X_train_spec.shape[1:]
    model = ecg_model.build_multiscale_ptq_model(spec_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=ecg_model.focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )
    model.summary()

    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

    print("\nTraining...")
    model.fit(
        x=[X_train_spec, X_train_rr],
        y=y_train,
        validation_data=([X_val_spec, X_val_rr], y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop_cb],
    )

    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs(args.deployment_dir, exist_ok=True)

    model.save(os.path.join(args.artifacts_dir, "ecg_multiscale_model.keras"))

    print("\nConverting to INT8 TFLite (shuffled calibration set)...")
    tflite_bytes = export.convert_to_int8_tflite(
        model, X_train_spec, X_train_rr, num_samples=100, seed=args.seed
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

    print("\n--- Test set evaluation: FLOAT model ---")
    test_probs_float = model.predict([X_test_spec, X_test_rr]).flatten()
    float_overall, float_per_record = evaluate.evaluate_predictions(
        y_test, test_probs_float, record_ids_test, threshold
    )
    print(json.dumps(float_overall, indent=2))
    print(float_per_record.to_string(index=False))

    print("\n--- Test set evaluation: INT8 model (actually deployed) ---")
    test_probs_int8 = evaluate.predict_tflite(tflite_path, X_test_spec, X_test_rr)
    int8_overall, int8_per_record = evaluate.evaluate_predictions(
        y_test, test_probs_int8, record_ids_test, threshold
    )
    print(json.dumps(int8_overall, indent=2))
    print(int8_per_record.to_string(index=False))

    model_size_kb = len(tflite_bytes) / 1024
    print(f"\nModel size (INT8 .tflite): {model_size_kb:.2f} KB")

    with open(os.path.join(args.artifacts_dir, "threshold.json"), "w") as f:
        json.dump({
            "threshold": threshold,
            "strategy": args.threshold_strategy,
            "min_precision": args.min_precision if args.threshold_strategy == "recall_at_precision" else None,
            "validation_metrics_at_threshold": val_metrics_at_threshold,
        }, f, indent=2)

    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
        json.dump({
            "model_size_kb": model_size_kb,
            "float": float_overall,
            "int8": int8_overall,
        }, f, indent=2)

    per_record_combined = pd.concat([
        float_per_record.assign(model="float"),
        int8_per_record.assign(model="int8"),
    ], ignore_index=True)
    per_record_combined.to_csv(os.path.join(args.artifacts_dir, "per_record_metrics.csv"), index=False)

    export.write_c_header(tflite_bytes, os.path.join(args.deployment_dir, "model_data.h"))
    export.write_threshold_header(threshold, os.path.join(args.deployment_dir, "threshold.h"))

    print(f"\nWrote artifacts to {args.artifacts_dir}/ and {args.deployment_dir}/.")


if __name__ == "__main__":
    main()
