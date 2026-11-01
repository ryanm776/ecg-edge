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
`--classes nsv` trains an N/S/V softmax model instead of the binary one, and `--loss-weights none` drops the
focal loss's class weighting (Stage 2 step 2a). `--loss-weights balanced` (N/S/V only, step 2b) weights each class
by inverse frequency in the model's own training view, normalized to a frequency-weighted mean of 1, and early-stops
on the unweighted validation loss, the criterion of `--loss-weights none`. `--rr-lookback` sets how many R-R intervals
`local_mean_rr` averages, for every view (training, OOF, DS2; Stage 2 step 3a). One artifacts dir holds one configuration.
Writes <artifacts-dir>/cv/<variant>_seed<S>/ per seed and <artifacts-dir>/cv_comparison.{csv,md} plus
cv_comparison_seeds.csv over all runs found there. In the tables, `int8_val`/`float_val` and `val_f1` refer
to the pooled out-of-fold predictions, which play the role of validation.
"""
from __future__ import annotations

import argparse
import glob
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
from ecg.features import (  # noqa: E402
    NSV_CLASSES, RR_LOOKBACK, balanced_class_weights, loss_scale, nsv_labels, type_recall,
)

VARIANTS = ["bn_folded"]
LEARNING_RATE = 0.001
# Stage 2 step 2a (diagnosis.md, "Stage 2 plan"): focal gamma 2 in every run. "matched" weights each beat the
# same in both models (binary alpha 0.25 on abnormal = N 0.75 / S 0.25 / V 0.25); "none" weights every beat 1.
FOCAL_GAMMA = 2.0
LOSS_WEIGHTS = {
    ("binary", "matched"): 0.25,
    ("binary", "none"): None,
    ("nsv", "matched"): (0.75, 0.25, 0.25),
    ("nsv", "none"): None,
    ("nsv", "balanced"): "balanced",  # computed per training view (balanced_class_weights)
}
# Step 2b: a balanced fold model early-stops on the unweighted val loss (plus L2), as `none` does, not on its
# S-heavy weighted val loss over the patients it is then scored on (diagnosis.md, "Step 2b design").
UNWEIGHTED_MONITOR = {"balanced"}
# The configuration of runs written before --classes / --loss-weights / --rr-lookback existed.
DEFAULT_CONFIG = {"classes": "binary", "loss_weights": "matched", "rr_lookback": RR_LOOKBACK}
CV_COLUMNS = [
    "run", "variant", "calib_samples", "seed", "final_epochs", "model", "threshold_from", "threshold", "val_f1",
    "test_precision", "test_recall", "test_f1", "tp", "fp", "fn", "se_V", "pp_V", "se_S", "pp_S", "S_as_V", "V_as_S",
    "f1_excl_202", "fp_excl_202", "fn_excl_202", "pr_auc_excl_202",
    "pr_auc", "roc_auc", "r95_threshold", "r95_test_precision", "r95_test_recall", "r95_fp",
    "tflite_kb", "min_int8_levels",
]
CV_TABLE = compare_variants.Table(
    subdir="cv", name="cv_comparison", title="Grouped CV over DS1: final model on DS2",
    columns=CV_COLUMNS,
    summary_columns=["pr_auc", "test_f1", "fp", "fn", "se_V", "se_S", "pp_V", "pp_S", "S_as_V", "V_as_S",
                     "f1_excl_202", "fp_excl_202", "fn_excl_202", "pr_auc_excl_202", "threshold", "final_epochs"],
    optional_columns=["S_as_V", "V_as_S"],
    notes=[
        "DS2, one row per run (seed in `seed`), final model trained on all 22 DS1 records for `final_epochs`",
        "(median of the 5 folds' best epochs). `threshold_from`: `int8_val` is the threshold chosen on the",
        "pooled out-of-fold INT8 scores (the deployed convention); `float_val` is the float model's own pooled-OOF",
        "threshold. `val_f1` is the pooled-OOF F1 at the row's threshold. `r95_*`: the highest threshold reaching",
        ">= 0.95 pooled-OOF recall, then DS2 precision / recall / FP there. Record 201 is never held out, so it",
        "is not in the OOF pool.",
        "`se_V` / `se_S`: AAMI sensitivity for V (V, E) and S (A, a, J, S). `pp_V` / `pp_S` (+P) are undefined",
        "for a binary model (an abnormal prediction names neither class) and show `—`. For an N/S/V model",
        "(`+nsv` in `variant`) a beat is called N unless 1 - P(N) > threshold, else the likelier of S and V;",
        "Se and +P count that call, so an S beat called V is an S miss. `S_as_V` / `V_as_S` count those",
        "mix-ups (`—` for binary). `pr_auc`, `test_*`, `tp`/`fp`/`fn` are N vs. abnormal for every model.",
        "N/S/V models drop F and Q beats, like the binary ones.",
        "`*_excl_202`: sensitivity check, DS2 without record 202 (same patient as DS1 record 201);",
        "reported alongside the full-DS2 columns, not instead of them.",
        "Per-fold OOF results and a diagnostic DS2 evaluation of each fold model are in each run's folds.json.",
    ],
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, default="bn_folded")
    parser.add_argument("--classes", choices=["binary", "nsv"], default="binary",
                        help="binary: one sigmoid P(abnormal); nsv: softmax over N/S/V (thresholds still on 1 - P(N)).")
    parser.add_argument("--loss-weights", choices=["matched", "none", "balanced"], default="matched",
                        help="matched: binary alpha 0.25 / N/S/V weights 0.75, 0.25, 0.25; none: weight 1 on every "
                             "beat; balanced (N/S/V only): inverse class frequency in the training view, "
                             "frequency-weighted mean 1, early stopping on the unweighted val loss.")
    parser.add_argument("--rr-lookback", type=int, default=RR_LOOKBACK,
                        help=f"Intervals averaged into local_mean_rr (default {RR_LOOKBACK}; step 3a: 30).")
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
    if args.rr_lookback < 1:
        parser.error("--rr-lookback must be at least 1")
    if (args.classes, args.loss_weights) not in LOSS_WEIGHTS:
        parser.error(f"--loss-weights {args.loss_weights} needs --classes nsv")
    existing = [run_dir(args, s) for s in args.seeds if os.path.exists(os.path.join(run_dir(args, s), "folds.json"))]
    if existing and not args.aggregate_only and not args.overwrite:
        parser.error(f"runs already exist: {existing}; pass a new --artifacts-dir or --overwrite")
    others = {path: cfg for path, cfg in existing_configs(args.artifacts_dir).items() if cfg != run_config(args)}
    if others:
        parser.error(f"{args.artifacts_dir} holds runs of another configuration ({others}); "
                     f"this one is {run_config(args)}. Use one --artifacts-dir per configuration.")
    return args


def run_config(args):
    return {"classes": args.classes, "loss_weights": args.loss_weights, "rr_lookback": args.rr_lookback}


def existing_configs(artifacts_dir):
    """{run dir: {"classes", "loss_weights", "rr_lookback"}} of the runs under <artifacts_dir>/cv/ (DEFAULT_CONFIG for runs
    written before the flags existed)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(artifacts_dir, "cv", "*", "metrics.json"))):
        with open(path) as f:
            metrics = json.load(f)
        out[os.path.dirname(path)] = {k: metrics.get(k, v) for k, v in DEFAULT_CONFIG.items()}
    return out


def row_label(args):
    """`variant` in the comparison tables: e.g. bn_folded, bn_folded+unweighted, bn_folded+nsv, bn_folded+nsv+rr30."""
    label = args.variant + ("+nsv" if args.classes == "nsv" else "")
    label += {"matched": "", "none": "+unweighted", "balanced": "+balanced"}[args.loss_weights]
    return label + ("" if args.rr_lookback == RR_LOOKBACK else f"+rr{args.rr_lookback}")


def loss_weights(args, fit_labels):
    """The configuration's loss weights for a model trained on `fit_labels`: binary alpha, a per-class
    tuple, or None (unweighted). Balanced weights depend on the training view; the others are fixed."""
    weights = LOSS_WEIGHTS[(args.classes, args.loss_weights)]
    if weights == "balanced":
        return tuple(float(w) for w in balanced_class_weights(fit_labels, len(NSV_CLASSES)))
    return weights


def make_loss(args, ecg_model, weights):
    if args.classes == "binary":
        return ecg_model.focal_loss(gamma=FOCAL_GAMMA, alpha=weights)
    return ecg_model.categorical_focal_loss(gamma=FOCAL_GAMMA, class_weights=weights)


def monitor_loss(args, ecg_model):
    """The loss fold models early-stop on, besides L2: None for the training loss's own val_loss."""
    if args.loss_weights in UNWEIGHTED_MONITOR:
        return make_loss(args, ecg_model, None)
    return None


def early_stopping_monitor(args):
    return "unweighted_val_loss" if args.loss_weights in UNWEIGHTED_MONITOR else "val_loss"


def loss_record(args, fit_labels, weights):
    """Weights and loss scale (frequency-weighted mean weight) of one model's loss, for folds.json / metrics.json."""
    if args.classes == "binary":
        per_class = None if weights is None else (1 - weights, weights)
        scale = loss_scale(fit_labels, per_class, n_classes=2)
    else:
        scale = loss_scale(fit_labels, weights, n_classes=len(NSV_CLASSES))
    return {"weights": list(weights) if isinstance(weights, tuple) else weights, "loss_scale": scale}


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
    X_tr_spec, X_tr_rr, y_tr, rec_tr, sym_tr = data.build_dataset(
        train.DS1_RECORDS, args.data_dir, max_normal_per_record=args.max_normal_per_record_train,
        augment=True, seed=seed, return_symbols=True, rr_lookback=args.rr_lookback,
    )
    spec_shape = X_tr_spec.shape[1:]
    # Training targets; the binary y_ev / y_te stay the labels for every threshold and metric.
    multiclass = args.classes == "nsv"
    n_classes = len(NSV_CLASSES) if multiclass else 1
    fit_tr = nsv_labels(sym_tr) if multiclass else y_tr
    fit_ev = nsv_labels(sym_ev) if multiclass else y_ev
    train_masks = [m[0] for m in cv.fold_masks(rec_tr, train.CV_FOLDS)]
    held_out_masks = [m[1] for m in cv.fold_masks(rec_ev, train.CV_FOLDS)]

    # Per fold: raw model outputs on the held-out fold ((n,) binary, (n, 3) N/S/V), and the fold model's
    # DS2 abnormal scores (diagnostic).
    best_epochs, oof_float, oof_int8, test_int8_by_fold, fold_losses = [], [], [], [], []
    stop_on = monitor_loss(args, ecg_model)
    for k, (tr, ho) in enumerate(zip(train_masks, held_out_masks)):
        print(f"\n=== seed {seed}, fold {k}: hold out {train.CV_FOLDS[k]} ===", flush=True)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        m = train.build_model(args.variant, spec_shape, tf, ecg_model, n_classes=n_classes)
        weights = loss_weights(args, fit_tr[tr])
        fold_losses.append(loss_record(args, fit_tr[tr], weights))
        best_epochs.append(train.fit_model(
            m, tf, ecg_model, [X_tr_spec[tr], X_tr_rr[tr]], fit_tr[tr],
            ([X_ev_spec[ho], X_ev_rr[ho]], fit_ev[ho]), LEARNING_RATE, args.epochs, args.batch_size,
            loss=make_loss(args, ecg_model, weights), monitor_loss=stop_on,
        ))
        tflite_path = os.path.join(out_dir, "folds", f"fold{k}_int8.tflite")
        with open(tflite_path, "wb") as f:
            f.write(export.convert_to_int8_tflite(m, X_tr_spec[tr], X_tr_rr[tr],
                                                  num_samples=args.calib_samples, seed=seed))
        oof_float.append(_squeeze(m.predict([X_ev_spec[ho], X_ev_rr[ho]])))
        oof_int8.append(evaluate.predict_tflite(tflite_path, X_ev_spec[ho], X_ev_rr[ho]))
        test_int8_by_fold.append(evaluate.abnormal_score(evaluate.predict_tflite(tflite_path, X_te_spec, X_te_rr)))

    probs_float, covered = cv.pool_oof(oof_float, held_out_masks)
    probs_int8, _ = cv.pool_oof(oof_int8, held_out_masks)
    y_oof, rec_oof, sym_oof = y_ev[covered], rec_ev[covered], sym_ev[covered]
    probs_float, probs_int8 = probs_float[covered], probs_int8[covered]
    pooled_float, pooled_int8 = evaluate.abnormal_score(probs_float), evaluate.abnormal_score(probs_int8)

    threshold, sweep = evaluate.select_threshold(
        y_oof, pooled_int8, strategy=args.threshold_strategy, min_precision=args.min_precision)
    float_threshold, float_sweep = evaluate.select_threshold(
        y_oof, pooled_float, strategy=args.threshold_strategy, min_precision=args.min_precision)
    print(f"\nPooled-OOF threshold: INT8 {threshold:.2f}, float {float_threshold:.2f}")

    folds = [
        fold_report(k, train.CV_FOLDS[k], best_epochs[k], y_ev[ho], sym_ev[ho], evaluate.abnormal_score(oof_int8[k]),
                    y_te, test_int8_by_fold[k], threshold, evaluate) | {"loss": fold_losses[k]}
        for k, ho in enumerate(held_out_masks)
    ]

    n_epochs = cv.final_epochs(best_epochs)
    print(f"\n=== seed {seed}, final model: all DS1, {n_epochs} epochs (fold best: {best_epochs}) ===", flush=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = train.build_model(args.variant, spec_shape, tf, ecg_model, n_classes=n_classes)
    final_weights = loss_weights(args, fit_tr)
    final_loss = loss_record(args, fit_tr, final_weights)
    train.fit_model(model, tf, ecg_model, [X_tr_spec, X_tr_rr], fit_tr, None, LEARNING_RATE, n_epochs,
                    args.batch_size, loss=make_loss(args, ecg_model, final_weights))
    model.save(os.path.join(out_dir, "ecg_multiscale_model.keras"))
    tflite_bytes = export.convert_to_int8_tflite(model, X_tr_spec, X_tr_rr,
                                                 num_samples=args.calib_samples, seed=seed)
    tflite_path = os.path.join(out_dir, "model_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    test_probs = {"float": _squeeze(model.predict([X_te_spec, X_te_rr])),
                  "int8": evaluate.predict_tflite(tflite_path, X_te_spec, X_te_rr)}
    test_float, test_int8 = (evaluate.abnormal_score(test_probs[m]) for m in ("float", "int8"))
    oof = SimpleNamespace(y=y_oof, record_ids=rec_oof, symbols=sym_oof, float=pooled_float, int8=pooled_int8,
                          probs={"float": probs_float, "int8": probs_int8},
                          threshold=threshold, sweep=sweep, float_threshold=float_threshold, float_sweep=float_sweep)
    write_outputs(args, seed, out_dir, n_epochs, best_epochs, folds, oof, ds2, test_float, test_int8,
                  tflite_bytes, tflite_path, evaluate, final_loss, test_class_probs=test_probs if multiclass else None)


def _squeeze(probs):
    """Keras output as predict_tflite returns it: (n,) for a one-unit output, (n, k) otherwise."""
    probs = np.asarray(probs)
    return probs[:, 0] if probs.ndim == 2 and probs.shape[1] == 1 else probs


def write_outputs(args, seed, out_dir, n_epochs, best_epochs, folds, oof, ds2, test_float, test_int8,
                  tflite_bytes, tflite_path, evaluate, final_loss, test_class_probs=None):
    """`final_loss`: loss_record of the final model's loss; each fold's is in its folds.json entry."""
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
                   "early_stopping_monitor": early_stopping_monitor(args), "folds": folds}, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "variant": args.variant,
            **run_config(args),
            "loss": {"focal_gamma": FOCAL_GAMMA, **final_loss,
                     "classes": list(NSV_CLASSES) if args.classes == "nsv" else ["N", "abnormal"],
                     "early_stopping_monitor": early_stopping_monitor(args)},
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
        row_label(args), row_args, oof.y, y_te, rec_te, sym_te, preds, threshold, float_threshold,
        model_size_kb, levels, evaluate, class_probs=test_class_probs,
    ).assign(final_epochs=n_epochs).to_csv(os.path.join(out_dir, "comparison_rows.csv"), index=False)
    train.val_per_record_rows(
        oof.y, oof.record_ids, preds, threshold, float_threshold, evaluate,
    ).to_csv(os.path.join(out_dir, "oof_per_record_metrics.csv"), index=False)
    # `float` / `int8` are abnormal scores; an N/S/V run also saves the (n, 3) probabilities, columns NSV_CLASSES.
    class_probs = {}
    if args.classes == "nsv":
        class_probs = {"float_probs": oof.probs["float"], "int8_probs": oof.probs["int8"],
                       "classes": np.array(NSV_CLASSES)}
    np.savez(os.path.join(out_dir, "oof_predictions.npz"), record_ids=oof.record_ids.astype(str), y=oof.y,
             symbols=oof.symbols.astype(str), float=oof.float, int8=oof.int8, **class_probs)


def main():
    args = parse_args()
    if not args.aggregate_only:
        import tensorflow as tf

        from ecg import cv, data, evaluate, export
        from ecg import model as ecg_model

        # Evaluation views are seed-independent: uncapped, unaugmented.
        ds1_eval = data.build_dataset(train.DS1_RECORDS, args.data_dir, max_normal_per_record=None,
                                      augment=False, return_symbols=True, rr_lookback=args.rr_lookback)
        ds2 = data.build_dataset(train.DS2_RECORDS, args.data_dir, max_normal_per_record=None,
                                 augment=False, return_symbols=True, rr_lookback=args.rr_lookback)
        for seed in args.seeds:
            start = time.monotonic()
            run_seed(args, seed, ds1_eval, ds2, tf, ecg_model, data, evaluate, export, cv)
            print(f"=== seed {seed} finished in {(time.monotonic() - start) / 60:.1f} min ===", flush=True)
    compare_variants.aggregate(args.artifacts_dir, CV_TABLE)


if __name__ == "__main__":
    main()
