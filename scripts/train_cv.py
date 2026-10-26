#!/usr/bin/env python
"""Patient-grouped cross-validation over DS1, then one final model evaluated once on DS2.

Per seed (bn_folded only; the baseline is retired from new experiments):
  1. For each of the 5 folds in train.CV_FOLDS: train on DS1 minus the fold (201, in no fold, always trains),
     early-stopping on the held-out fold; convert to INT8; score the held-out fold (float and INT8).
     Each fold model is also scored on DS2 as a diagnostic (folds.json) -- never used for any choice.
  2. Pool the out-of-fold scores over the 21 folded records. The threshold is selected on the pooled
     INT8 scores (the deployed convention); the float model's own pooled-OOF threshold is kept too.
  3. Train the final model on all 22 DS1 records for the median of the folds' best epochs (no early
     stopping), convert to INT8, and evaluate on DS2 at the pooled-OOF threshold.

All thresholds run on evaluate.abnormal_score (P(abnormal) for binary output, 1 - P(N) for multiclass).
Writes <artifacts-dir>/cv/<variant>_seed<S>/ per seed and <artifacts-dir>/cv_comparison.{csv,md} plus
cv_comparison_seeds.csv over all runs found there. In the tables, `int8_val`/`float_val` and `val_f1` refer
to the pooled out-of-fold predictions, which play the role of validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import compare_variants  # noqa: E402  (pandas only)
import train  # noqa: E402  (numpy/pandas only)
from ecg.features import type_recall  # noqa: E402

VARIANTS = ["bn_folded"]
LEARNING_RATE = 0.001
CV_COLUMNS = [
    "run", "variant", "calib_samples", "seed", "final_epochs", "model", "threshold_from", "threshold", "val_f1",
    "test_precision", "test_recall", "test_f1", "tp", "fp", "fn", "se_V", "pp_V", "se_S", "pp_S",
    "f1_excl_202", "fp_excl_202", "fn_excl_202", "pr_auc_excl_202",
    "pr_auc", "roc_auc", "r95_threshold", "r95_test_precision", "r95_test_recall", "r95_fp",
    "tflite_kb", "min_int8_levels",
]
CV_TABLE = compare_variants.Table(
    subdir="cv", name="cv_comparison", title="Grouped CV over DS1: final model on DS2",
    columns=CV_COLUMNS,
    summary_columns=["pr_auc", "test_f1", "fp", "fn", "se_V", "se_S", "f1_excl_202", "fp_excl_202",
                     "fn_excl_202", "pr_auc_excl_202", "threshold", "final_epochs"],
    notes=[
        "DS2, one row per run (seed in `seed`), final model trained on all 22 DS1 records for `final_epochs`",
        "(median of the 5 folds' best epochs). `threshold_from`: `int8_val` is the threshold chosen on the",
        "pooled out-of-fold INT8 scores (the deployed convention); `float_val` is the float model's own pooled-OOF",
        "threshold. `val_f1` is the pooled-OOF F1 at the row's threshold. `r95_*`: the highest threshold reaching",
        ">= 0.95 pooled-OOF recall, then DS2 precision / recall / FP there. Record 201 is never held out, so it",
        "is not in the OOF pool.",
        "`se_V` / `se_S`: AAMI sensitivity for V (V, E) and S (A, a, J, S). `pp_V` / `pp_S` (+P) are undefined",
        "for a binary model (an abnormal prediction names neither class) and show `—`.",
        "`*_excl_202`: sensitivity check, DS2 without record 202 (same patient as DS1 record 201);",
        "reported alongside the full-DS2 columns, not instead of them.",
        "Per-fold OOF results and a diagnostic DS2 evaluation of each fold model are in each run's folds.json.",
    ],
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, default="bn_folded")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Run the full CV + final model once per seed (e.g. 42 43 44).")
    parser.add_argument("--artifacts-dir", default=os.path.join("artifacts", "dechazal"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per fold (early stopping on the fold).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calib-samples", type=int, default=100)
    parser.add_argument("--max-normal-per-record-train", type=int, default=600)
    parser.add_argument("--threshold-strategy", choices=["f1", "recall_at_precision"], default="f1")
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--aggregate-only", action="store_true", help="Skip the runs; just rebuild the tables.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing runs for these seeds (refused by default; earlier runs may be inputs "
                             "to other analyses, e.g. diagnose_s_beats.py reads artifacts/dechazal/cv/).")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds has duplicates")
    existing = [run_dir(args, s) for s in args.seeds if os.path.exists(os.path.join(run_dir(args, s), "folds.json"))]
    if existing and not args.aggregate_only and not args.overwrite:
        parser.error(f"runs already exist: {existing}; pass a new --artifacts-dir or --overwrite")
    return args


def run_dir(args, seed):
    return os.path.join(args.artifacts_dir, "cv", f"{args.variant}_seed{seed}")


def fold_report(k, records, best_epoch, y_ho, symbols_ho, oof_int8, y_test, test_int8, threshold, evaluate):
    """folds.json entry: the fold's own OOF metrics at the pooled threshold, and the fold model's DS2
    metrics (diagnostic only: shows fold-to-fold spread and whether the final model is in line)."""
    oof, _ = evaluate.evaluate_predictions(y_ho, oof_int8, np.zeros(len(y_ho)), threshold)
    by_type = type_recall(y_ho, symbols_ho, oof_int8, threshold)
    ds2, _ = evaluate.evaluate_predictions(y_test, test_int8, np.zeros(len(y_test)), threshold)
    (_, ds2_fp), (ds2_fn, _) = ds2["confusion_matrix"]
    return {
        "fold": k,
        "records": list(records),
        "best_epoch": best_epoch,
        "oof_int8": {
            "n_beats": len(y_ho),
            "n_abnormal": int(np.sum(y_ho)),
            "f1": oof["f1"],
            "pr_auc": evaluate.ranking_metrics(y_ho, oof_int8)["pr_auc"],
            "recall_ventricular": by_type["ventricular"][2],
            "recall_supraventricular": by_type["supraventricular"][2],
        },
        "ds2_int8_diagnostic": {
            "pr_auc": evaluate.ranking_metrics(y_test, test_int8)["pr_auc"],
            "f1": ds2["f1"],
            "fp": ds2_fp,
            "fn": ds2_fn,
        },
    }


def run_seed(args, seed, ds1_eval, ds2, tf, ecg_model, data, evaluate, export, cv):
    X_ev_spec, X_ev_rr, y_ev, rec_ev, sym_ev = ds1_eval
    X_te_spec, X_te_rr, y_te, rec_te, sym_te = ds2
    out_dir = run_dir(args, seed)
    os.makedirs(os.path.join(out_dir, "folds"), exist_ok=True)

    # Training view of DS1: normal-beat cap and augmentation, like train.py's train set.
    X_tr_spec, X_tr_rr, y_tr, rec_tr = data.build_dataset(
        train.DS1_RECORDS, args.data_dir, max_normal_per_record=args.max_normal_per_record_train,
        augment=True, seed=seed,
    )
    spec_shape = X_tr_spec.shape[1:]
    train_masks = [m[0] for m in cv.fold_masks(rec_tr, train.CV_FOLDS)]
    held_out_masks = [m[1] for m in cv.fold_masks(rec_ev, train.CV_FOLDS)]

    best_epochs, oof_float, oof_int8, test_int8_by_fold = [], [], [], []
    for k, (tr, ho) in enumerate(zip(train_masks, held_out_masks)):
        print(f"\n=== seed {seed}, fold {k}: hold out {train.CV_FOLDS[k]} ===", flush=True)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        m = train.build_model(args.variant, spec_shape, tf, ecg_model)
        best_epochs.append(train.fit_model(
            m, tf, ecg_model, [X_tr_spec[tr], X_tr_rr[tr]], y_tr[tr],
            ([X_ev_spec[ho], X_ev_rr[ho]], y_ev[ho]), LEARNING_RATE, args.epochs, args.batch_size,
        ))
        tflite_path = os.path.join(out_dir, "folds", f"fold{k}_int8.tflite")
        with open(tflite_path, "wb") as f:
            f.write(export.convert_to_int8_tflite(m, X_tr_spec[tr], X_tr_rr[tr],
                                                  num_samples=args.calib_samples, seed=seed))
        oof_float.append(evaluate.abnormal_score(m.predict([X_ev_spec[ho], X_ev_rr[ho]])))
        oof_int8.append(evaluate.abnormal_score(evaluate.predict_tflite(tflite_path, X_ev_spec[ho], X_ev_rr[ho])))
        test_int8_by_fold.append(evaluate.abnormal_score(evaluate.predict_tflite(tflite_path, X_te_spec, X_te_rr)))

    pooled_float, covered = cv.pool_oof(oof_float, held_out_masks)
    pooled_int8, _ = cv.pool_oof(oof_int8, held_out_masks)
    y_oof, rec_oof, sym_oof = y_ev[covered], rec_ev[covered], sym_ev[covered]
    pooled_float, pooled_int8 = pooled_float[covered], pooled_int8[covered]

    threshold, sweep = evaluate.select_threshold(
        y_oof, pooled_int8, strategy=args.threshold_strategy, min_precision=args.min_precision)
    float_threshold, float_sweep = evaluate.select_threshold(
        y_oof, pooled_float, strategy=args.threshold_strategy, min_precision=args.min_precision)
    print(f"\nPooled-OOF threshold: INT8 {threshold:.2f}, float {float_threshold:.2f}")

    folds = [
        fold_report(k, train.CV_FOLDS[k], best_epochs[k], y_ev[ho], sym_ev[ho], oof_int8[k],
                    y_te, test_int8_by_fold[k], threshold, evaluate)
        for k, ho in enumerate(held_out_masks)
    ]

    n_epochs = cv.final_epochs(best_epochs)
    print(f"\n=== seed {seed}, final model: all DS1, {n_epochs} epochs (fold best: {best_epochs}) ===", flush=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = train.build_model(args.variant, spec_shape, tf, ecg_model)
    train.fit_model(model, tf, ecg_model, [X_tr_spec, X_tr_rr], y_tr, None, LEARNING_RATE, n_epochs,
                    args.batch_size)
    model.save(os.path.join(out_dir, "ecg_multiscale_model.keras"))
    tflite_bytes = export.convert_to_int8_tflite(model, X_tr_spec, X_tr_rr,
                                                 num_samples=args.calib_samples, seed=seed)
    tflite_path = os.path.join(out_dir, "model_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    test_float = evaluate.abnormal_score(model.predict([X_te_spec, X_te_rr]))
    test_int8 = evaluate.abnormal_score(evaluate.predict_tflite(tflite_path, X_te_spec, X_te_rr))
    oof = SimpleNamespace(y=y_oof, record_ids=rec_oof, symbols=sym_oof, float=pooled_float, int8=pooled_int8,
                          threshold=threshold, sweep=sweep, float_threshold=float_threshold, float_sweep=float_sweep)
    write_outputs(args, seed, out_dir, n_epochs, best_epochs, folds, oof, ds2, test_float, test_int8,
                  tflite_bytes, tflite_path, evaluate)


def write_outputs(args, seed, out_dir, n_epochs, best_epochs, folds, oof, ds2, test_float, test_int8,
                  tflite_bytes, tflite_path, evaluate):
    X_te_spec, X_te_rr, y_te, rec_te, sym_te = ds2
    threshold, float_threshold, sweep = oof.threshold, oof.float_threshold, oof.sweep
    float_overall, float_per_record = evaluate.evaluate_predictions(y_te, test_float, rec_te, threshold)
    float_own, _ = evaluate.evaluate_predictions(y_te, test_float, rec_te, float_threshold)
    int8_overall, int8_per_record = evaluate.evaluate_predictions(y_te, test_int8, rec_te, threshold)
    keep = rec_te != train.OVERLAP_RECORD

    def excl(probs, t):
        return evaluate.evaluate_predictions(y_te[keep], probs[keep], rec_te[keep], t)[0]

    model_size_kb = len(tflite_bytes) / 1024
    levels_idx = np.random.default_rng(seed).permutation(len(y_te))[:train.LEVELS_SAMPLE_SIZE]
    levels, levels_tensor = evaluate.min_int8_levels(tflite_path, X_te_spec[levels_idx], X_te_rr[levels_idx])

    with open(os.path.join(out_dir, "threshold.json"), "w") as f:
        json.dump({
            "threshold": threshold,
            "float_threshold": float_threshold,
            "source": "pooled out-of-fold INT8 scores over DS1 (201 never held out)",
            "strategy": args.threshold_strategy,
            "min_precision": args.min_precision if args.threshold_strategy == "recall_at_precision" else None,
            "oof_metrics_at_threshold": sweep.loc[sweep["threshold"] == threshold].iloc[0].to_dict(),
        }, f, indent=2)
    with open(os.path.join(out_dir, "folds.json"), "w") as f:
        json.dump({"best_epochs": best_epochs, "final_epochs": n_epochs, "threshold": threshold,
                   "folds": folds}, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "variant": args.variant,
            "calib_samples": args.calib_samples,
            "final_epochs": n_epochs,
            "model_size_kb": model_size_kb,
            "min_int8_levels": levels,
            "min_int8_levels_tensor": levels_tensor,
            "float": float_overall,
            "float_own_threshold": float_own,
            "int8": int8_overall,
            f"float_excl_{train.OVERLAP_RECORD}": excl(test_float, threshold),
            f"float_own_threshold_excl_{train.OVERLAP_RECORD}": excl(test_float, float_threshold),
            f"int8_excl_{train.OVERLAP_RECORD}": excl(test_int8, threshold),
        }, f, indent=2)
    pd.concat([float_per_record.assign(model="float"), int8_per_record.assign(model="int8")],
              ignore_index=True).to_csv(os.path.join(out_dir, "per_record_metrics.csv"), index=False)

    preds = {"float": (oof.float, oof.float_sweep, test_float), "int8": (oof.int8, sweep, test_int8)}
    row_args = SimpleNamespace(variant=args.variant, calib_samples=args.calib_samples,
                               target_recall=args.target_recall, seed=seed)
    train.comparison_rows(
        args.variant, row_args, oof.y, y_te, rec_te, sym_te, preds, threshold, float_threshold,
        model_size_kb, levels, evaluate,
    ).assign(final_epochs=n_epochs).to_csv(os.path.join(out_dir, "comparison_rows.csv"), index=False)
    train.val_per_record_rows(
        oof.y, oof.record_ids, preds, threshold, float_threshold, evaluate,
    ).to_csv(os.path.join(out_dir, "oof_per_record_metrics.csv"), index=False)
    np.savez(os.path.join(out_dir, "oof_predictions.npz"), record_ids=oof.record_ids.astype(str), y=oof.y,
             symbols=oof.symbols.astype(str), float=oof.float, int8=oof.int8)


def main():
    args = parse_args()
    if not args.aggregate_only:
        import tensorflow as tf

        from ecg import cv, data, evaluate, export
        from ecg import model as ecg_model

        # Evaluation views are seed-independent: uncapped, unaugmented.
        ds1_eval = data.build_dataset(train.DS1_RECORDS, args.data_dir, max_normal_per_record=None,
                                      augment=False, return_symbols=True)
        ds2 = data.build_dataset(train.DS2_RECORDS, args.data_dir, max_normal_per_record=None,
                                 augment=False, return_symbols=True)
        for seed in args.seeds:
            start = time.monotonic()
            run_seed(args, seed, ds1_eval, ds2, tf, ecg_model, data, evaluate, export, cv)
            print(f"=== seed {seed} finished in {(time.monotonic() - start) / 60:.1f} min ===", flush=True)
    compare_variants.aggregate(args.artifacts_dir, CV_TABLE)


if __name__ == "__main__":
    main()
