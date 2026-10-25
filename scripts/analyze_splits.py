#!/usr/bin/env python
"""Diagnose the validation/test mismatch with the saved models (read-only, no training).

Reports, written to <out-dir>/summary.md plus CSVs and a PNG:
  1. Beat counts by annotation symbol per split and per record (including F and Q, which
     the dataset drops), for the current split and the de Chazal DS1/DS2 split.
  2. Per-record metrics on val (and test, for comparison).
  3. Validation F1 / PR-AUC with and without record 232 (only while 232 is in val; under the
     de Chazal split it is in test).
  4. Recall by abnormal type (ventricular V,E vs. supraventricular A,a,J,S) and per-type
     ROC-AUC / PR-AUC against normals, on val and test.
  5. Predicted-probability distribution of normal beats, val vs. test.

Threshold convention matches artifacts/variant_comparison.md: float models at both the
INT8-val threshold and their own float-val threshold, INT8 at the INT8-val threshold.
Baseline INT8 is left out on purpose: its behavior is dominated by the quantization bug
(artifacts/diagnosis.md), not by the split.

`--counts-only` writes section 1 and stops, without loading any model. Use it after a split
change: models trained on another split may have seen the new val/test patients, so their
metrics there are meaningless until they are retrained.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from ecg.features import (  # noqa: E402  (numpy/scipy only)
    ABNORMAL_TYPES, SUPRAVENTRICULAR, VENTRICULAR, WINDOW_SIZE, type_recall,
)
from train import (  # noqa: E402  (numpy/pandas only)
    CV_ALWAYS_TRAIN, CV_FOLDS, DS1_RECORDS, DS2_RECORDS, TEST_RECORDS, TRAIN_RECORDS, VAL_RECORDS,
)

COUNT_SYMBOLS = ["N", "L", "R", "e", "j", "V", "E", "A", "a", "J", "S", "F", "Q"]
EXCLUDED_RECORD = "232"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "split_analysis"))
    parser.add_argument("--baseline-model", default=os.path.join("artifacts", "ecg_multiscale_model.keras"))
    parser.add_argument("--baseline-dir", default=os.path.join("artifacts", "variants", "baseline"),
                        help="Holds the baseline's threshold.json and comparison_rows.csv.")
    parser.add_argument("--bn-folded-dir", default=os.path.join("artifacts", "variants", "bn_folded"))
    parser.add_argument("--no-dechazal", action="store_true",
                        help="Skip DS1/DS2 symbol counts (needs records 116 117 122 207 228 234).")
    parser.add_argument("--counts-only", action="store_true",
                        help="Write only the beat counts by split (section 1); no models are loaded.")
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Load <out-dir>/predictions.npz instead of re-running the models.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------- pure helpers (tested)

def usable_symbol_counts(samples, symbols, signal_len, window_size=WINDOW_SIZE):
    """Count beats per symbol under the same usability rule as extract_beats_from_record
    (skip the first/last annotation and beats whose window doesn't fit), but for every
    symbol in COUNT_SYMBOLS, including F and Q, which the dataset drops."""
    half_window = window_size // 2
    counts = Counter()
    for i in range(1, len(samples) - 1):
        sym = symbols[i]
        if sym not in COUNT_SYMBOLS:
            continue
        if samples[i] - half_window < 0 or samples[i] + half_window >= signal_len:
            continue
        counts[sym] += 1
    return counts


def add_group_columns(df):
    """Normal / ventricular / supraventricular / F / Q totals from per-symbol columns."""
    df = df.copy()
    df["normal"] = df[["N", "L", "R", "e", "j"]].sum(axis=1)
    df["ventricular"] = df[list(VENTRICULAR)].sum(axis=1)
    df["supraventricular"] = df[list(SUPRAVENTRICULAR)].sum(axis=1)
    abnormal = df["ventricular"] + df["supraventricular"]
    df["s_share_of_abnormal"] = (df["supraventricular"] / abnormal.where(abnormal > 0)).round(3)
    return df


def threshold_rows(variant_dir, has_int8):
    """(model, threshold_from, threshold) rows following variant_comparison's convention.
    The INT8-val threshold is in threshold.json; the float model's own val threshold is
    only in the float_val row of comparison_rows.csv."""
    with open(os.path.join(variant_dir, "threshold.json")) as f:
        t_int8 = float(json.load(f)["threshold"])
    rows = pd.read_csv(os.path.join(variant_dir, "comparison_rows.csv"))
    float_val = rows[(rows["model"] == "float") & (rows["threshold_from"] == "float_val")]
    if len(float_val) != 1:
        raise ValueError(f"expected one float/float_val row in {variant_dir}/comparison_rows.csv")
    t_float = float(float_val["threshold"].iloc[0])

    out = [("float", "int8_val", t_int8), ("float", "float_val", t_float)]
    if has_int8:
        out.append(("int8", "int8_val", t_int8))
    return out


def type_vs_normal_subset(y, symbols, abnormal_type):
    """Mask selecting normals plus one abnormal type (the other abnormal type excluded)."""
    y, symbols = np.asarray(y), np.asarray(symbols)
    return (y == 0) | ((y == 1) & np.isin(symbols, ABNORMAL_TYPES[abnormal_type]))


def excluding_record(record_ids, record):
    return np.asarray(record_ids) != record


def normal_prob_summary(y, probs, thresholds):
    """Quantiles of P(abnormal) for normal beats and the fraction above each threshold."""
    p = np.asarray(probs)[np.asarray(y) == 0]
    out = {"n_normal": int(len(p))}
    for q in (50, 90, 95, 99):
        out[f"p{q}"] = float(np.percentile(p, q))
    for label, t in thresholds:
        out[f"frac_above_{label}"] = float((p > t).mean())
    return out


def df_to_md(df, floatfmt="{:.3f}"):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "—"
            return str(int(v)) if float(v).is_integer() and abs(v) >= 1 else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# ---------------------------------------------------------------- data and models

def symbol_counts(data_dir, include_dechazal):
    from ecg import data

    records = sorted(set(TRAIN_RECORDS + VAL_RECORDS + TEST_RECORDS)
                     | (set(DS1_RECORDS + DS2_RECORDS) if include_dechazal else set()))
    data.ensure_mitdb_cached(data_dir, records)
    current = {**{r: "train" for r in TRAIN_RECORDS}, **{r: "val" for r in VAL_RECORDS},
               **{r: "test" for r in TEST_RECORDS}}
    dechazal = {**{r: "DS1" for r in DS1_RECORDS}, **{r: "DS2" for r in DS2_RECORDS}}
    cv_fold = {**{r: f"fold{k}" for k, fold in enumerate(CV_FOLDS) for r in fold},
               **{r: "always_train" for r in CV_ALWAYS_TRAIN}}

    rows = []
    for rec in records:
        record, annotation = data.load_record(rec, data_dir)
        counts = usable_symbol_counts(annotation.sample, annotation.symbol, record.sig_len)
        rows.append({"record_id": rec, "current_split": current.get(rec, "unused"),
                     "dechazal_split": dechazal.get(rec, "—"), "cv_fold": cv_fold.get(rec, "—"),
                     **{s: counts.get(s, 0) for s in COUNT_SYMBOLS}})
    return add_group_columns(pd.DataFrame(rows))


def predict_all(args, X_val, X_test):
    """{model_key: {"val": probs, "test": probs}} for baseline float, bn_folded float/INT8."""
    import tensorflow as tf

    from ecg import evaluate

    preds = {}
    for key, path in [("baseline/float", args.baseline_model),
                      ("bn_folded/float", os.path.join(args.bn_folded_dir, "ecg_multiscale_model.keras"))]:
        print(f"Predicting with {path}...")
        model = tf.keras.models.load_model(path, compile=False)
        preds[key] = {split: model.predict(list(X), verbose=0).flatten() for split, X in
                      (("val", X_val), ("test", X_test))}
    tflite_path = os.path.join(args.bn_folded_dir, "model_int8.tflite")
    print(f"Predicting with {tflite_path}...")
    preds["bn_folded/int8"] = {split: evaluate.predict_tflite(tflite_path, *X) for split, X in
                               (("val", X_val), ("test", X_test))}
    return preds


def load_or_predict(args):
    path = os.path.join(args.out_dir, "predictions.npz")
    if args.reuse_predictions:
        z = np.load(path, allow_pickle=True)
        labels = {split: (z[f"{split}_y"], z[f"{split}_records"], z[f"{split}_symbols"]) for split in ("val", "test")}
        preds = {}
        for name in z.files:
            if name.startswith("p__"):
                _, variant, model, split = name.split("__")
                preds.setdefault(f"{variant}/{model}", {})[split] = z[name]
        return labels, preds

    from ecg import data

    split_data = {}
    for split, recs in (("val", VAL_RECORDS), ("test", TEST_RECORDS)):
        X_spec, X_rr, y, record_ids, symbols = data.build_dataset(
            recs, args.data_dir, max_normal_per_record=None, augment=False, seed=args.seed, return_symbols=True,
        )
        split_data[split] = ((X_spec, X_rr), (y, record_ids, symbols))
    preds = predict_all(args, split_data["val"][0], split_data["test"][0])
    labels = {split: split_data[split][1] for split in split_data}

    arrays = {f"{s}_{k}": v for s, (y, r, sym) in labels.items() for k, v in (("y", y), ("records", r), ("symbols", sym))}
    arrays.update({f"p__{key.replace('/', '__')}__{s}": p for key, d in preds.items() for s, p in d.items()})
    np.savez_compressed(path, **arrays)
    return labels, preds


# ---------------------------------------------------------------- reports

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    md =["# Validation/test split analysis", "",
          "Generated by `scripts/analyze_splits.py`. Thresholds follow `variant_comparison.md`: "
          "float at the INT8-val threshold and at its own float-val threshold; INT8 at the INT8-val threshold.", ""]

    # 1. Symbol counts
    counts = symbol_counts(args.data_dir, not args.no_dechazal)
    counts.to_csv(os.path.join(args.out_dir, "symbol_counts_by_record.csv"), index=False)
    total_cols = COUNT_SYMBOLS + ["normal", "ventricular", "supraventricular"]
    md += ["## 1. Beat counts by annotation symbol", "",
           "Usable beats (same window rule as the dataset; train is uncapped and unaugmented). "
           "F and Q are dropped by `SYMBOL_MAP`.", ""]
    # cv_fold: grouped-CV folds over DS1 (scripts/train_cv.py); "—" rows are DS2.
    for split_col in ["current_split"] + ([] if args.no_dechazal else ["dechazal_split", "cv_fold"]):
        totals = add_group_columns(counts.groupby(split_col)[COUNT_SYMBOLS].sum().reset_index())
        totals.to_csv(os.path.join(args.out_dir, f"symbol_counts_by_{split_col}.csv"), index=False)
        md += [f"### Totals by `{split_col}`", "", df_to_md(totals[[split_col] + total_cols + ["s_share_of_abnormal"]]), ""]
    md += ["### Per record", "", df_to_md(counts), ""]
    if args.counts_only:
        write_summary(args.out_dir, md)
        return

    from ecg import evaluate  # imports TensorFlow

    # Predictions and threshold rows
    labels, preds = load_or_predict(args)
    rows = [("baseline", m, frm, t) for m, frm, t in threshold_rows(args.baseline_dir, has_int8=False)]
    rows += [("bn_folded", m, frm, t) for m, frm, t in threshold_rows(args.bn_folded_dir, has_int8=True)]

    def row_label(variant, model, frm, t):
        return f"{variant} {model} @ {frm} ({t:.2f})"

    # Overall + 2. per-record metrics
    overall_rows, per_record_frames = [], []
    for variant, model, frm, t in rows:
        for split in ("val", "test"):
            y, record_ids, _ = labels[split]
            probs = preds[f"{variant}/{model}"][split]
            overall, per_record = evaluate.evaluate_predictions(y, probs, record_ids, t)
            (_, fp), (fn, tp) = overall["confusion_matrix"]
            overall_rows.append({"row": row_label(variant, model, frm, t), "split": split,
                                 "precision": overall["precision"], "recall": overall["recall"], "f1": overall["f1"],
                                 "tp": tp, "fp": fp, "fn": fn, **evaluate.ranking_metrics(y, probs)})
            per_record_frames.append(per_record.assign(row=row_label(variant, model, frm, t), split=split))
    overall_df = pd.DataFrame(overall_rows)
    per_record_df = pd.concat(per_record_frames, ignore_index=True)
    overall_df.to_csv(os.path.join(args.out_dir, "overall_metrics.csv"), index=False)
    per_record_df.to_csv(os.path.join(args.out_dir, "per_record_metrics.csv"), index=False)
    md += ["## Overall metrics (sanity check against variant_comparison.md)", "", df_to_md(overall_df), ""]
    per_record_cols = ["record_id", "n_beats", "n_abnormal", "n_normal", "tp", "fp", "fn", "tn",
                       "accuracy", "precision", "recall", "f1"]
    md += ["## 2. Per-record metrics", ""]
    for split in ("val", "test"):
        for label in overall_df["row"].unique():
            sub = per_record_df[(per_record_df["split"] == split) & (per_record_df["row"] == label)]
            md += [f"### {split}: {label}", "", df_to_md(sub[per_record_cols]), ""]

    # 3. Validation with and without record 232
    if EXCLUDED_RECORD in VAL_RECORDS:
        y, record_ids, _ = labels["val"]
        keep = excluding_record(record_ids, EXCLUDED_RECORD)
        excl_rows = []
        for variant, model, frm, t in rows:
            probs = preds[f"{variant}/{model}"]["val"]
            for subset, mask in (("all val", np.ones_like(keep)), (f"val excl. {EXCLUDED_RECORD}", keep)):
                overall, _ = evaluate.evaluate_predictions(y[mask], probs[mask], record_ids[mask], t)
                best_t, sweep = evaluate.select_threshold(y[mask], probs[mask])
                excl_rows.append({"row": row_label(variant, model, frm, t), "subset": subset,
                                  "n_abnormal": int(y[mask].sum()), "f1_at_threshold": overall["f1"],
                                  "precision": overall["precision"], "recall": overall["recall"],
                                  "best_f1": float(sweep["f1"].max()), "best_threshold": best_t,
                                  "pr_auc": evaluate.ranking_metrics(y[mask], probs[mask])["pr_auc"]})
        excl_df = pd.DataFrame(excl_rows)
        excl_df.to_csv(os.path.join(args.out_dir, "val_excluding_232.csv"), index=False)
        md += [f"## 3. Validation with and without record {EXCLUDED_RECORD}", "",
               "`best_f1` / `best_threshold` sweep the subset itself; `pr_auc` is per model, independent of the row threshold.", "",
               df_to_md(excl_df), ""]
    else:
        md += [f"## 3. Validation with and without record {EXCLUDED_RECORD}", "",
               f"Skipped: record {EXCLUDED_RECORD} is not in the validation set.", ""]

    # 4. Recall by abnormal type + per-type ranking metrics
    type_rows, type_record_rows, type_auc_rows = [], [], []
    for variant, model, frm, t in rows:
        for split in ("val", "test"):
            y, record_ids, symbols = labels[split]
            probs = preds[f"{variant}/{model}"][split]
            label = row_label(variant, model, frm, t)
            for name, (n, detected, recall) in type_recall(y, symbols, probs, t).items():
                type_rows.append({"row": label, "split": split, "type": name, "n": n, "detected": detected, "recall": recall})
            for rec in sorted(set(record_ids.tolist())):
                m = record_ids == rec
                for name, (n, detected, recall) in type_recall(y[m], symbols[m], probs[m], t).items():
                    if n:
                        type_record_rows.append({"row": label, "split": split, "record_id": rec, "type": name,
                                                 "n": n, "detected": detected, "recall": recall})
    for key, d in preds.items():
        for split in ("val", "test"):
            y, _, symbols = labels[split]
            for name in ABNORMAL_TYPES:
                m = type_vs_normal_subset(y, symbols, name)
                type_auc_rows.append({"model": key, "split": split, "task": f"{name} vs normal",
                                      "n_abnormal": int(y[m].sum()), **evaluate.ranking_metrics(y[m], d[split][m])})
    type_df, type_record_df, type_auc_df = map(pd.DataFrame, (type_rows, type_record_rows, type_auc_rows))
    type_df.to_csv(os.path.join(args.out_dir, "recall_by_type.csv"), index=False)
    type_record_df.to_csv(os.path.join(args.out_dir, "recall_by_type_per_record.csv"), index=False)
    type_auc_df.to_csv(os.path.join(args.out_dir, "auc_by_type.csv"), index=False)
    md += ["## 4. Abnormal types: ventricular (V, E) vs. supraventricular (A, a, J, S)", "",
           "### Recall by type", "", df_to_md(type_df), "",
           "### Threshold-independent, each type vs. normals (other abnormal type excluded)", "", df_to_md(type_auc_df), "",
           "### Recall by type, per record", "", df_to_md(type_record_df), ""]

    # 5. Normal-beat probability distribution
    prob_rows, prob_record_rows = [], []
    for key in preds:
        variant, model = key.split("/")
        thresholds = [(f"{frm}_{t:.2f}", t) for v, m, frm, t in rows if v == variant and m == model]
        for split in ("val", "test"):
            y, record_ids, _ = labels[split]
            probs = preds[key][split]
            prob_rows.append({"model": key, "split": split, **normal_prob_summary(y, probs, thresholds)})
            for rec in sorted(set(record_ids.tolist())):
                m = record_ids == rec
                prob_record_rows.append({"model": key, "split": split, "record_id": rec,
                                         **normal_prob_summary(y[m], probs[m], thresholds)})
    prob_df, prob_record_df = pd.DataFrame(prob_rows), pd.DataFrame(prob_record_rows)
    prob_df.to_csv(os.path.join(args.out_dir, "normal_prob_summary.csv"), index=False)
    prob_record_df.to_csv(os.path.join(args.out_dir, "normal_prob_per_record.csv"), index=False)
    plot_normal_histograms(labels, preds, os.path.join(args.out_dir, "normal_prob_hist.png"))
    md += ["## 5. P(abnormal) for normal beats, val vs. test", "",
           "`frac_above_<threshold_from>_<t>`: fraction of normal beats that would be false positives.", "",
           df_to_md(prob_df, floatfmt="{:.4f}"), "", "### Per record", "",
           df_to_md(prob_record_df, floatfmt="{:.4f}"), "", "![normal-beat histograms](normal_prob_hist.png)", ""]

    write_summary(args.out_dir, md)
    print(df_to_md(overall_df[overall_df["split"] == "test"]))


def write_summary(out_dir, md):
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nWrote {out_dir}/summary.md")


def plot_normal_histograms(labels, preds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 3.5), sharey=True)
    bins = np.linspace(0, 1, 51)
    for ax, (key, d) in zip(np.atleast_1d(axes), preds.items()):
        for split in ("val", "test"):
            y = labels[split][0]
            ax.hist(d[split][y == 0], bins=bins, histtype="step", density=True, label=split)
        ax.set_yscale("log")
        ax.set_title(key)
        ax.set_xlabel("P(abnormal), normal beats")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
