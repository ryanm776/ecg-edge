#!/usr/bin/env python
"""Stage 1 diagnosis of missed S beats and wide-QRS false positives (read-only, no training).

Uses the grouped-CV runs in <cv-dir>/<variant>_seed<S>/ (oof_predictions.npz, threshold.json, the fold
models in folds/, the final model) and writes <out-dir>/step<k>*.{md,csv}. Every Stage 2 decision rests on
DS1 out-of-fold (OOF) results; DS2 appears only in outputs labeled descriptive.

This diagnoses the pre-fix pipeline: the CV runs it reads were trained on all-annotation R-R features, so
it rebuilds the data with build_dataset(rr_from_all_annotations=True, lead=None) (channel 0, so V5 for
record 114). It doesn't apply to runs trained after the beat-only R-R fix (Stage 2 step 1) or the lead fix
(step 1b).

Steps (--steps, default 0 1 2 4):
  0. R-R audit. extract_beats_from_record takes R-R intervals between consecutive annotations, including
     non-beat ones (rhythm "+", noise "~", artifact "|", ...). Per record and AAMI class: how many beats'
     R-R features differ from beat-only R-R intervals, and by how much. Also checks that the recomputed
     features match build_dataset's X_rr exactly.
  1. OOF score ranking by class: pooled and within-patient S-vs-N and V-vs-N AUC, score quantiles.
  2. Is the S information in the R-R features? (a) per-patient feature distributions, S vs. N, including
     232 (DS2, descriptive); (b) R-R-only S-vs-N classifiers under the same grouped folds, with mean-,
     median-, 90th-percentile- and max-based prematurity ratios, from all-annotation and beat-only R-R.
  3. Shuffle tests (slow): re-score each held-out fold with its fold model after shuffling the R-R
     inputs, the spectrograms, or both, within each record. Shows whether S scores depend on timing or shape.
  4. Wide-QRS false positives: OOF normal-beat scores by symbol (N, L, R), L/R coverage of each fold's
     training set, and the final models' DS2 false positives by symbol (descriptive).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from analyze_splits import df_to_md  # noqa: E402  (numpy/pandas only)
from ecg.features import (  # noqa: E402  (numpy/scipy only)
    SYMBOL_MAP, WINDOW_SIZE, aami_class, beat_annotation_mask, compute_rr_timing_features, rr_quantile_ratio,
)
from train import CV_FOLDS, DS1_RECORDS, DS2_RECORDS  # noqa: E402  (numpy/pandas only)

BASE_FEATURES = ["pre_rr", "post_rr", "local_mean_rr", "rr_ratio", "pre_post_ratio"]
RATIO_LOOKBACKS = (10, 30)
RATIO_QUANTILES = {"q50": 0.5, "q90": 0.9, "max": 1.0}
RATIO_FEATURES = [f"ratio_{stat}{n}" for n in RATIO_LOOKBACKS for stat in ("mean", *RATIO_QUANTILES)]
VERSIONS = ("all_ann", "beat_only")
MIN_CLASS_BEATS = 20
DESCRIPTIVE_DS2_RECORD = "232"
FP_SYMBOLS = ["N", "L", "R", "e", "j"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 1, 2, 4], choices=[0, 1, 2, 3, 4])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cv-dir", default=os.path.join("artifacts", "dechazal", "cv"))
    parser.add_argument("--out-dir", default=os.path.join("artifacts", "dechazal", "stage1"))
    parser.add_argument("--variant", default="bn_folded")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--shuffle-seeds", nargs="+", type=int, default=[0, 1, 2], help="Step 3 repeats.")
    parser.add_argument("--max-normal-per-record-train", type=int, default=600,
                        help="Normal-beat cap for the step 2b training folds; matches train_cv.py.")
    return parser.parse_args()


# ---------------------------------------------------------------- pure helpers (tested)

def _rr_features(version, rr_intervals, i):
    out = dict(zip((f"{version}:{f}" for f in BASE_FEATURES), compute_rr_timing_features(rr_intervals, i)))
    for n in RATIO_LOOKBACKS:
        out[f"{version}:ratio_mean{n}"] = compute_rr_timing_features(rr_intervals, i, lookback=n)[3]
        for stat, q in RATIO_QUANTILES.items():
            out[f"{version}:ratio_{stat}{n}"] = rr_quantile_ratio(rr_intervals, i, lookback=n, q=q)
    return out


def feature_columns(version):
    return [f"{version}:{f}" for f in BASE_FEATURES + RATIO_FEATURES]


def rr_feature_rows(samples, symbols, fs, signal_len, window_size=WINDOW_SIZE):
    """One row per beat extract_beats_from_record keeps, in its order, with R-R features computed two
    ways: `all_ann` (between consecutive annotations, which is what the model gets) and `beat_only`
    (between consecutive beat annotations). beat_only is NaN for a record's first and last beat.
    `pre_nonbeat` / `post_nonbeat` / `window10_nonbeat`: a non-beat annotation is the previous / next
    annotation, or falls in the 10-interval lookback that local_mean_rr averages."""
    samples = np.asarray(samples)
    is_beat = beat_annotation_mask(symbols)
    beat_idx = np.flatnonzero(is_beat)
    beat_pos = {a: j for j, a in enumerate(beat_idx)}
    rr_all = np.diff(samples) / fs
    rr_beat = np.diff(samples[beat_idx]) / fs
    half_window = window_size // 2

    rows = []
    for i in range(1, len(samples) - 1):
        sym = symbols[i]
        if sym not in SYMBOL_MAP:
            continue
        if samples[i] - half_window < 0 or samples[i] + half_window >= signal_len:
            continue
        row = {"ann_index": i, "symbol": sym, "pre_nonbeat": not is_beat[i - 1], "post_nonbeat": not is_beat[i + 1],
               "window10_nonbeat": not is_beat[max(0, i - 10):i].all()}
        row.update(_rr_features("all_ann", rr_all, i))
        j = beat_pos[i]
        if 0 < j < len(beat_idx) - 1:
            row.update(_rr_features("beat_only", rr_beat, j))
        else:
            row.update({c: np.nan for c in feature_columns("beat_only")})
        rows.append(row)
    return rows


def capped_normal_mask(record_ids, y, cap):
    """True for every abnormal beat and for the first `cap` normal beats of each record, in order:
    the beats extract_beats_from_record keeps with max_normal=cap."""
    record_ids, y = np.asarray(record_ids), np.asarray(y)
    keep = np.ones(len(y), dtype=bool)
    seen = Counter()
    for k, (rec, label) in enumerate(zip(record_ids, y)):
        if label == 0:
            seen[rec] += 1
            keep[k] = seen[rec] <= cap
    return keep


def within_record_permutation(record_ids, rng):
    """Index array that shuffles beats only among beats of the same record."""
    record_ids = np.asarray(record_ids)
    perm = np.arange(len(record_ids))
    for rec in pd.unique(record_ids):
        idx = np.flatnonzero(record_ids == rec)
        perm[idx] = rng.permutation(idx)
    return perm


def class_vs_normal_auc(aami, scores, cls):
    """(n_cls, ROC-AUC, PR-AUC) of `cls` beats against N beats; NaN when either side is empty."""
    aami, scores = np.asarray(aami), np.asarray(scores)
    m = np.isin(aami, ["N", cls])
    y = aami[m] == cls
    if y.all() or not y.any():
        return int(y.sum()), float("nan"), float("nan")
    return int(y.sum()), float(roc_auc_score(y, scores[m])), float(average_precision_score(y, scores[m]))


def class_score_summary(aami, scores, threshold):
    """Per AAMI class: n, mean, median, p90, p99 and fraction above `threshold`."""
    rows = []
    for cls in ("N", "S", "V"):
        s = np.asarray(scores)[np.asarray(aami) == cls]
        if len(s):
            rows.append({"class": cls, "n": len(s), "mean": float(s.mean()), "p50": float(np.median(s)),
                         "p90": float(np.percentile(s, 90)), "p99": float(np.percentile(s, 99)),
                         "frac_above_t": float((s > threshold).mean())})
    return rows


def mean_std(df, by, cols):
    """`mean ± std` strings over seeds (ddof=1), as in the comparison tables."""
    g = df.groupby(by, sort=False)[cols]
    mean, std, n = g.mean(), g.std(ddof=1), g.size()
    out = mean.copy().astype(object)
    for c in cols:
        out[c] = [f"{m:.3f}" if k == 1 or np.isnan(s) else f"{m:.3f} ± {s:.3f}"
                  for m, s, k in zip(mean[c], std[c], n)]
    return out.reset_index()


# ---------------------------------------------------------------- data

def rr_table(records, data_dir):
    from ecg import data

    data.ensure_mitdb_cached(data_dir, records)
    frames = []
    for rec in records:
        record, annotation = data.load_record(rec, data_dir)
        rows = rr_feature_rows(annotation.sample, annotation.symbol, record.fs, record.sig_len)
        nonbeat = Counter(s for s in annotation.symbol if not beat_annotation_mask([s])[0])
        df = pd.DataFrame(rows)
        df.insert(0, "record_id", rec)
        df["n_nonbeat_annotations"] = sum(nonbeat.values())
        df["nonbeat_symbols"] = " ".join(f"{s}:{c}" for s, c in sorted(nonbeat.items()))
        frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    table["aami"] = aami_class(table["symbol"].to_numpy())
    table["y"] = table["symbol"].map(SYMBOL_MAP).astype(int)
    return table


def build_eval_view(records, data_dir):
    """The uncapped, unaugmented view the pre-fix train_cv.py runs scored (X_spec, X_rr, y, record_ids, symbols)."""
    from ecg import data

    return data.build_dataset(records, data_dir, max_normal_per_record=None, augment=False, return_symbols=True,
                              rr_from_all_annotations=True, lead=None)


def reconcile_table(table, view, name):
    """The recomputed all-annotation features and labels must equal build_dataset's, beat for beat."""
    _, X_rr, y, record_ids, symbols = view
    checks = {
        "n_beats": len(table) == len(y),
        "record_ids": np.array_equal(table["record_id"].to_numpy(), record_ids.astype(str)),
        "symbols": np.array_equal(table["symbol"].to_numpy(), symbols.astype(str)),
        "y": np.array_equal(table["y"].to_numpy(), y),
        "X_rr": np.array_equal(table[feature_columns("all_ann")[:5]].to_numpy(np.float32), X_rr),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"{name}: recomputed R-R table doesn't match build_dataset on {failed}")
    print(f"{name}: R-R table matches build_dataset ({len(y)} beats)")


def oof_rows(table):
    """Rows of the DS1 table that are in the OOF pool (every CV fold; 201 is never held out)."""
    return table[table["record_id"].isin([r for fold in CV_FOLDS for r in fold])].reset_index(drop=True)


def load_oof(args, seed, oof_table):
    run = os.path.join(args.cv_dir, f"{args.variant}_seed{seed}")
    z = np.load(os.path.join(run, "oof_predictions.npz"), allow_pickle=True)
    if not (np.array_equal(z["record_ids"].astype(str), oof_table["record_id"].to_numpy())
            and np.array_equal(z["y"], oof_table["y"].to_numpy())):
        raise AssertionError(f"{run}: oof_predictions.npz is not aligned with the rebuilt DS1 OOF rows")
    with open(os.path.join(run, "threshold.json")) as f:
        threshold = float(json.load(f)["threshold"])
    return {"int8": z["int8"], "float": z["float"]}, threshold, run


def records_with(table, cls, min_beats=MIN_CLASS_BEATS):
    counts = table[table["aami"] == cls].groupby("record_id").size()
    return sorted(counts[counts >= min_beats].index)


def write_md(out_dir, name, lines):
    path = os.path.join(out_dir, f"{name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


# ---------------------------------------------------------------- step 0

def step0(args, tables):
    all_cols, beat_cols = feature_columns("all_ann")[:5], feature_columns("beat_only")[:5]
    rows, examples = [], []
    for split, table in tables.items():
        a = table[all_cols].to_numpy(np.float64)
        b = table[beat_cols].to_numpy(np.float64)
        undefined = np.isnan(b).any(axis=1)
        differs = undefined | ~np.isclose(a, np.nan_to_num(b, nan=-1.0), rtol=1e-9, atol=0).all(axis=1)
        t = table.assign(differs=differs, beat_only_undefined=undefined,
                         rr_ratio_rel_err=np.abs(a[:, 3] - b[:, 3]) / b[:, 3],
                         pre_rr_rel_err=np.abs(a[:, 0] - b[:, 0]) / b[:, 0])
        for rec, g in t.groupby("record_id", sort=False):
            row = {"split": split, "record_id": rec, "nonbeat_symbols": g["nonbeat_symbols"].iloc[0],
                   "n_beats": len(g), "n_differ": int(g["differs"].sum()),
                   "share_differ": float(g["differs"].mean()),
                   "n_pre_nonbeat": int(g["pre_nonbeat"].sum()), "n_post_nonbeat": int(g["post_nonbeat"].sum()),
                   "n_beat_only_undefined": int(g["beat_only_undefined"].sum())}
            for cls in ("N", "S", "V"):
                c = g[g["aami"] == cls]
                row[f"{cls}_differ"] = f"{int(c['differs'].sum())}/{len(c)}"
            d = g[g["differs"] & ~g["beat_only_undefined"]]
            row["median_rr_ratio_rel_err"] = float(d["rr_ratio_rel_err"].median()) if len(d) else float("nan")
            row["max_rr_ratio_rel_err"] = float(d["rr_ratio_rel_err"].max()) if len(d) else float("nan")
            rows.append(row)
        examples.append(t[t["differs"] & ~t["beat_only_undefined"]]
                        .nlargest(10, "rr_ratio_rel_err")
                        .assign(split=split)[["split", "record_id", "ann_index", "symbol", "pre_nonbeat",
                                              "post_nonbeat", "all_ann:pre_rr", "beat_only:pre_rr",
                                              "all_ann:rr_ratio", "beat_only:rr_ratio"]])
        tables[split] = t

    per_record = pd.DataFrame(rows)
    per_record.to_csv(os.path.join(args.out_dir, "step0_rr_audit_per_record.csv"), index=False)
    by_class = []
    for split, t in tables.items():
        for cls in ("N", "S", "V"):
            c = t[t["aami"] == cls]
            d = c[c["differs"] & ~c["beat_only_undefined"]]
            by_class.append({"split": split, "class": cls, "n": len(c), "n_differ": int(c["differs"].sum()),
                             "share_differ": float(c["differs"].mean()),
                             "n_pre_nonbeat": int(c["pre_nonbeat"].sum()),
                             "median_rr_ratio_rel_err": float(d["rr_ratio_rel_err"].median()) if len(d) else np.nan,
                             "p90_rr_ratio_rel_err": float(d["rr_ratio_rel_err"].quantile(0.9)) if len(d) else np.nan,
                             "median_pre_rr_rel_err": float(d["pre_rr_rel_err"].median()) if len(d) else np.nan})
    by_class = pd.DataFrame(by_class)
    by_class.to_csv(os.path.join(args.out_dir, "step0_rr_audit_by_class.csv"), index=False)
    examples = pd.concat(examples, ignore_index=True)

    affected = per_record[per_record["n_differ"] > 0].sort_values("share_differ", ascending=False)
    write_md(args.out_dir, "step0_rr_audit", [
        "# Step 0: R-R features across non-beat annotations", "",
        "`all_ann`: R-R between consecutive annotations (the model's current input). `beat_only`: between "
        "consecutive beat annotations. A beat `differs` when any of the 5 R-R features differs (relative "
        "tolerance 1e-9) or its beat-only features are undefined (first/last beat of a record). Relative errors "
        "are |all_ann - beat_only| / beat_only, over differing beats with defined beat-only features.", "",
        "Both tables were checked beat-for-beat against `build_dataset`'s `X_rr`.", "",
        "## By split and AAMI class", "", df_to_md(by_class, floatfmt="{:.4f}"), "",
        "## Records with any differing beat", "", df_to_md(affected, floatfmt="{:.4f}"), "",
        "## Largest rr_ratio errors", "", df_to_md(examples, floatfmt="{:.4f}"), ""])


# ---------------------------------------------------------------- step 1

def step1(args, ds1):
    oof = oof_rows(ds1)
    aami = oof["aami"].to_numpy()
    pooled, within, summary = [], [], []
    s_records, v_records = records_with(oof, "S"), records_with(oof, "V")
    for seed in args.seeds:
        scores, threshold, _ = load_oof(args, seed, oof)
        for model, sc in scores.items():
            for cls in ("S", "V"):
                n, roc, pr = class_vs_normal_auc(aami, sc, cls)
                pooled.append({"seed": seed, "model": model, "task": f"{cls} vs N", "n": n, "roc_auc": roc,
                               "pr_auc": pr, "recall_at_t": float((sc[aami == cls] > threshold).mean())})
            for row in class_score_summary(aami, sc, threshold):
                summary.append({"seed": seed, "model": model, **row})
            for cls, recs in (("S", s_records), ("V", v_records)):
                for rec in recs:
                    m = (oof["record_id"] == rec).to_numpy()
                    n, roc, _ = class_vs_normal_auc(aami[m], sc[m], cls)
                    within.append({"seed": seed, "model": model, "task": f"{cls} vs N", "record_id": rec, "n": n,
                                   "n_normal": int((aami[m] == "N").sum()), "roc_auc": roc,
                                   "recall_at_t": float((sc[m][aami[m] == cls] > threshold).mean()),
                                   f"{cls}_median": float(np.median(sc[m][aami[m] == cls])),
                                   "N_median": float(np.median(sc[m][aami[m] == "N"])),
                                   "N_frac_above_t": float((sc[m][aami[m] == "N"] > threshold).mean())})
    pooled, within, summary = map(pd.DataFrame, (pooled, within, summary))
    pooled.to_csv(os.path.join(args.out_dir, "step1_pooled_auc.csv"), index=False)
    within.to_csv(os.path.join(args.out_dir, "step1_within_patient_auc.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "step1_score_summary.csv"), index=False)

    w_int8 = within[within["model"] == "int8"]
    write_md(args.out_dir, "step1_ranking", [
        "# Step 1: OOF score ranking by class", "",
        f"Pooled OOF scores of the grouped-CV fold models (seeds {' '.join(map(str, args.seeds))}). Each class is "
        "ranked against N beats only (the other abnormal class excluded). Within-patient: the patient's own "
        f"S (or V) beats against the patient's own N beats, for records with >= {MIN_CLASS_BEATS} beats of the "
        "class. `recall_at_t` and `frac_above_t` use each run's pooled-OOF threshold. Mean ± std over seeds.", "",
        "## Pooled", "", df_to_md(mean_std(pooled, ["model", "task"], ["roc_auc", "pr_auc", "recall_at_t"])), "",
        "## Within patient (INT8)", "",
        df_to_md(mean_std(w_int8, ["task", "record_id", "n", "n_normal"],
                          ["roc_auc", "recall_at_t", "N_frac_above_t"])), "",
        "## Score distribution by class (INT8)", "",
        df_to_md(mean_std(summary[summary["model"] == "int8"], ["class", "n"],
                          ["mean", "p50", "p90", "p99", "frac_above_t"])), ""])


# ---------------------------------------------------------------- step 2

DISTRIBUTION_FEATURES = ["rr_ratio", "pre_post_ratio"] + RATIO_FEATURES


def step2a(args, ds1, ds2):
    oof = oof_rows(ds1)
    groups = [("DS1 OOF", oof, r) for r in records_with(oof, "S")]
    groups.append(("DS2 (descriptive)", ds2, DESCRIPTIVE_DS2_RECORD))
    rows = []
    for split, table, rec in groups + [("DS1 OOF", oof, "pooled")]:
        g = table if rec == "pooled" else table[table["record_id"] == rec]
        g = g[g["aami"].isin(["N", "S"])]
        for version in VERSIONS:
            for feat in DISTRIBUTION_FEATURES:
                col = f"{version}:{feat}"
                x = g[col].to_numpy(np.float64)
                ok = ~np.isnan(x)
                s, n = x[ok & (g["aami"] == "S").to_numpy()], x[ok & (g["aami"] == "N").to_numpy()]
                # A premature beat has a low ratio, so the score is the negated feature.
                _, roc, _ = class_vs_normal_auc(g["aami"].to_numpy()[ok], -x[ok], "S")
                rows.append({"split": split, "record_id": rec, "version": version, "feature": feat,
                             "n_S": len(s), "n_N": len(n), "S_p25": np.percentile(s, 25), "S_p50": np.median(s),
                             "S_p75": np.percentile(s, 75), "N_p25": np.percentile(n, 25), "N_p50": np.median(n),
                             "N_p75": np.percentile(n, 75), "auc_S_vs_N": roc})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "step2a_feature_distributions.csv"), index=False)
    pivot = df.pivot_table(index=["split", "record_id", "n_S", "n_N", "feature"], columns="version",
                           values="auc_S_vs_N", sort=False).reset_index()
    write_md(args.out_dir, "step2a_feature_distributions", [
        "# Step 2a: R-R features of S vs. N beats, per patient", "",
        "Single-feature S-vs-N ROC-AUC within each patient (score = -feature: a premature beat has a low ratio), "
        "model-free. `ratio_<stat><N>` = pre_rr / <stat> of the last N R-R intervals (including pre_rr); "
        "`ratio_mean10` is the current `rr_ratio`. Record 232 is DS2 and descriptive only. Quartiles per class "
        "are in the CSV.", "",
        df_to_md(pivot), ""])


def _fit_predict(kind, X_tr, y_tr, X_te):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        X_tr, X_te = np.log(X_tr), np.log(X_te)
    else:
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def feature_sets():
    sets = {f"5 features, {r}": BASE_FEATURES[:3] + [r] + BASE_FEATURES[4:] for r in RATIO_FEATURES}
    sets.update({f"{r} alone": [r] for r in RATIO_FEATURES})
    sets["pre_post_ratio alone"] = ["pre_post_ratio"]
    return sets


def step2b(args, ds1):
    # Same beats as the CNN's training/OOF views; V beats dropped (this asks only S vs N), and beats whose
    # beat-only features are undefined dropped from every set so all rows compare identical beats.
    cap = capped_normal_mask(ds1["record_id"], ds1["y"], args.max_normal_per_record_train)
    defined = ~ds1[feature_columns("beat_only")].isna().any(axis=1).to_numpy()
    sv = ds1["aami"].isin(["N", "S"]).to_numpy()
    n_dropped = int((sv & ~defined).sum())
    fold_of = {r: k for k, fold in enumerate(CV_FOLDS) for r in fold}
    fold = ds1["record_id"].map(fold_of).to_numpy(dtype=float)  # NaN for 201 (always trains)
    held = sv & defined & ~np.isnan(fold)
    y = (ds1["aami"] == "S").to_numpy().astype(int)
    rec = ds1["record_id"].to_numpy()
    s_records = records_with(ds1[held], "S")

    def score_rows(label, scores):
        out = []
        _, roc, pr = class_vs_normal_auc(ds1["aami"].to_numpy()[held], scores, "S")
        out.append({**label, "record_id": "pooled", "roc_auc": roc, "pr_auc": pr})
        for r in s_records:
            m = rec[held] == r
            _, roc_r, _ = class_vs_normal_auc(ds1["aami"].to_numpy()[held][m], scores[m], "S")
            out.append({**label, "record_id": r, "roc_auc": roc_r, "pr_auc": np.nan})
        return out

    rows = []
    for version in VERSIONS:
        for set_name, feats in feature_sets().items():
            X = ds1[[f"{version}:{f}" for f in feats]].to_numpy(np.float64)
            for kind in ("logreg", "hgb"):
                scores = np.empty(held.sum())
                for k in range(len(CV_FOLDS)):
                    tr = sv & defined & cap & (fold != k)
                    te = held & (fold == k)
                    scores[(fold[held] == k)] = _fit_predict(kind, X[tr], y[tr], X[te])
                rows += score_rows({"source": f"R-R only, {kind}", "version": version, "feature_set": set_name},
                                   scores)
            print(f"step 2b: {version} / {set_name} done", flush=True)

    # The CNN on the identical beats, for comparison.
    oof = oof_rows(ds1)
    in_oof = ds1["record_id"].isin(oof["record_id"].unique()).to_numpy()
    held_in_oof = held[in_oof]
    for seed in args.seeds:
        scores, _, _ = load_oof(args, seed, oof)
        rows += score_rows({"source": f"CNN int8 seed {seed}", "version": "all_ann", "feature_set": "spectrogram + 5"},
                           scores["int8"][held_in_oof])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "step2b_rr_only_baseline.csv"), index=False)
    pooled = df[df["record_id"] == "pooled"]
    wide = df.pivot_table(index=["source", "version", "feature_set"], columns="record_id", values="roc_auc",
                          sort=False).reset_index()
    write_md(args.out_dir, "step2b_rr_only_baseline", [
        "# Step 2b: R-R-only S-vs-N baseline under the grouped folds", "",
        "Each classifier is trained on DS1 minus the held-out fold (201 always trains; N capped at "
        f"{args.max_normal_per_record_train} per record like train_cv.py; V beats dropped) and scores the held-out "
        "fold. `logreg`: balanced logistic regression on standardized log features. `hgb`: balanced "
        "HistGradientBoosting. Columns are OOF S-vs-N ROC-AUC, pooled and within each patient with "
        f">= {MIN_CLASS_BEATS} S beats. `5 features, <ratio>` swaps rr_ratio for that ratio; "
        "`5 features, ratio_mean10` is the current input set. CNN rows are the grouped-CV fold models' OOF INT8 "
        f"scores on the same beats. {n_dropped} N/S beats with undefined beat-only features are dropped from every row.",
        "", "## ROC-AUC", "", df_to_md(wide), "",
        "## Pooled PR-AUC", "", df_to_md(pooled[["source", "version", "feature_set", "pr_auc"]]), ""])


# ---------------------------------------------------------------- step 3

def step3(args, ds1, view):
    X_spec, X_rr, _, record_ids, _ = view
    oof = oof_rows(ds1)
    covered = ds1["record_id"].isin(oof["record_id"].unique()).to_numpy()
    fold_of = {r: k for k, fold in enumerate(CV_FOLDS) for r in fold}
    fold = np.array([fold_of.get(r, -1) for r in record_ids.astype(str)])
    aami = oof["aami"].to_numpy()
    s_records = records_with(oof, "S")
    conditions = ("rr", "spec", "both")

    from ecg import evaluate

    def score(seed, spec, rr):
        run = os.path.join(args.cv_dir, f"{args.variant}_seed{seed}")
        out = np.empty(len(record_ids), dtype=np.float32)
        for k in range(len(CV_FOLDS)):
            m = fold == k
            out[m] = evaluate.predict_tflite(os.path.join(run, "folds", f"fold{k}_int8.tflite"), spec[m], rr[m])
        return out[covered]

    rows = []
    for seed in args.seeds:
        t0 = time.monotonic()
        scores, threshold, _ = load_oof(args, seed, oof)
        base = score(seed, X_spec, X_rr)
        if not np.array_equal(base, scores["int8"]):
            raise AssertionError(f"seed {seed}: unshuffled fold-model scores don't reproduce oof_predictions.npz")
        runs = [("none", -1, base)]
        for s in args.shuffle_seeds:
            rng = np.random.default_rng(s)
            p_rr, p_spec = within_record_permutation(record_ids, rng), within_record_permutation(record_ids, rng)
            runs.append(("rr", s, score(seed, X_spec, X_rr[p_rr])))
            runs.append(("spec", s, score(seed, X_spec[p_spec], X_rr)))
            runs.append(("both", s, score(seed, X_spec[p_spec], X_rr[p_rr])))
        for cond, s, sc in runs:
            label = {"seed": seed, "shuffled": cond, "shuffle_seed": s}
            for cls in ("S", "V"):
                n, roc, pr = class_vs_normal_auc(aami, sc, cls)
                rows.append({**label, "metric": f"{cls} vs N roc_auc", "record_id": "pooled", "value": roc})
            for row in class_score_summary(aami, sc, threshold):
                for k in ("mean", "p50", "p90", "frac_above_t"):
                    rows.append({**label, "metric": f"{row['class']} {k}", "record_id": "pooled", "value": row[k]})
            for r in s_records:
                m = (oof["record_id"] == r).to_numpy()
                _, roc, _ = class_vs_normal_auc(aami[m], sc[m], "S")
                rows.append({**label, "metric": "S vs N roc_auc", "record_id": r, "value": roc})
                rows.append({**label, "metric": "S mean", "record_id": r,
                             "value": float(sc[m][aami[m] == "S"].mean())})
        print(f"step 3: seed {seed} done in {(time.monotonic() - t0) / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "step3_shuffle.csv"), index=False)
    # Average the shuffle repeats within a seed, then mean ± std over seeds.
    per_seed = df.groupby(["seed", "shuffled", "metric", "record_id"], sort=False)["value"].mean().reset_index()
    wide = per_seed.pivot_table(index=["seed", "metric", "record_id"], columns="shuffled", values="value",
                                sort=False).reset_index()
    for c in conditions:
        wide[f"delta_{c}"] = wide[c] - wide["none"]
    cols = ["none"] + list(conditions) + [f"delta_{c}" for c in conditions]
    pooled = mean_std(wide[wide["record_id"] == "pooled"], ["metric"], cols)
    per_record = mean_std(wide[wide["record_id"] != "pooled"], ["record_id", "metric"], cols)
    write_md(args.out_dir, "step3_shuffle", [
        "# Step 3: within-record shuffle of the fold models' inputs", "",
        "Each held-out fold is re-scored by its own INT8 fold model after shuffling the R-R vectors (`rr`), the "
        "spectrograms (`spec`) or both (independently) among beats of the same record. Values are averaged over "
        f"shuffle seeds {' '.join(map(str, args.shuffle_seeds))} within a seed, then mean ± std over model seeds "
        f"{' '.join(map(str, args.seeds))}. `delta_*` = shuffled - unshuffled. The unshuffled scores were checked "
        "to reproduce oof_predictions.npz exactly.", "",
        "## Pooled OOF", "", df_to_md(pooled), "",
        f"## Records with >= {MIN_CLASS_BEATS} S beats", "", df_to_md(per_record), ""])


# ---------------------------------------------------------------- step 4

def step4(args, ds1, ds2, ds2_view):
    oof = oof_rows(ds1)
    rows = []
    for seed in args.seeds:
        scores, threshold, _ = load_oof(args, seed, oof)
        sc = scores["int8"]
        for (rec, sym), g in oof[oof["aami"] == "N"].groupby(["record_id", "symbol"], sort=False):
            s = sc[g.index.to_numpy()]
            rows.append({"seed": seed, "record_id": rec, "symbol": sym, "n": len(s), "p50": float(np.median(s)),
                         "p90": float(np.percentile(s, 90)), "frac_above_t": float((s > threshold).mean())})
    oof_n = pd.DataFrame(rows)
    oof_n.to_csv(os.path.join(args.out_dir, "step4_oof_normal_scores.csv"), index=False)
    by_symbol = []
    for seed in args.seeds:
        scores, threshold, _ = load_oof(args, seed, oof)
        for sym in FP_SYMBOLS:
            m = (oof["symbol"] == sym).to_numpy()
            if m.any():
                by_symbol.append({"seed": seed, "symbol": sym, "n": int(m.sum()),
                                  "fp": int((scores["int8"][m] > threshold).sum()),
                                  "frac_above_t": float((scores["int8"][m] > threshold).mean())})
    by_symbol = pd.DataFrame(by_symbol)

    # L/R coverage of each fold model's training set (DS1 minus the fold, normals capped like train_cv.py).
    cap = capped_normal_mask(ds1["record_id"], ds1["y"], args.max_normal_per_record_train)
    coverage = []
    for k, fold in enumerate(CV_FOLDS):
        tr = ~ds1["record_id"].isin(fold).to_numpy()
        row = {"fold": k, "held_out": " ".join(fold)}
        for sym in ("L", "R"):
            m = tr & (ds1["symbol"] == sym).to_numpy()
            row[f"{sym}_patients"] = " ".join(sorted(ds1.loc[m, "record_id"].unique())) or "—"
            row[f"{sym}_beats_capped"] = int((m & cap).sum())
        row["N_beats_capped"] = int((tr & cap & (ds1["symbol"] == "N").to_numpy()).sum())
        coverage.append(row)
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(os.path.join(args.out_dir, "step4_fold_training_coverage.csv"), index=False)

    # DS2, descriptive: the final models' false positives by normal-beat symbol.
    from ecg import evaluate

    X_spec, X_rr, y, record_ids, _ = ds2_view
    ds2_rows = []
    for seed in args.seeds:
        run = os.path.join(args.cv_dir, f"{args.variant}_seed{seed}")
        with open(os.path.join(run, "threshold.json")) as f:
            threshold = float(json.load(f)["threshold"])
        sc = evaluate.predict_tflite(os.path.join(run, "model_int8.tflite"), X_spec, X_rr)
        pred = sc > threshold
        saved = pd.read_csv(os.path.join(run, "per_record_metrics.csv"), dtype={"record_id": str})
        saved = saved[saved["model"] == "int8"].set_index("record_id")
        for rec, g in ds2.groupby("record_id", sort=False):
            m = g.index.to_numpy()
            fp, fn = int((pred[m] & (y[m] == 0)).sum()), int((~pred[m] & (y[m] == 1)).sum())
            if (fp, fn) != (int(saved.loc[rec, "fp"]), int(saved.loc[rec, "fn"])):
                raise AssertionError(f"seed {seed}, record {rec}: FP/FN {fp}/{fn} don't match per_record_metrics.csv")
            if fp:
                row = {"seed": seed, "record_id": rec, "fp": fp}
                for sym in FP_SYMBOLS:
                    s = (g["symbol"] == sym).to_numpy()
                    row[sym] = f"{int(pred[m][s].sum())}/{int(s.sum())}" if s.any() else "—"
                ds2_rows.append(row)
    ds2_fp = pd.DataFrame(ds2_rows)
    ds2_fp.to_csv(os.path.join(args.out_dir, "step4_ds2_fp_by_symbol.csv"), index=False)

    oof_bbb = oof_n[oof_n["symbol"].isin(["L", "R"]) | oof_n["record_id"].isin(
        oof_n.loc[oof_n["symbol"].isin(["L", "R"]), "record_id"].unique())]
    write_md(args.out_dir, "step4_wide_qrs_fp", [
        "# Step 4: wide-QRS false positives", "",
        "## OOF normal beats by symbol (INT8, pooled-OOF threshold)", "",
        df_to_md(mean_std(by_symbol, ["symbol", "n"], ["fp", "frac_above_t"])), "",
        "### Records with L or R beats", "",
        df_to_md(mean_std(oof_bbb, ["record_id", "symbol", "n"], ["p50", "p90", "frac_above_t"])), "",
        "## L/R coverage of each fold model's training set", "", df_to_md(coverage), "",
        "## DS2 (descriptive): final INT8 model false positives by normal-beat symbol", "",
        "`<sym>`: false positives / normal beats of that symbol in the record. Per-record FP/FN were checked "
        "against each run's per_record_metrics.csv.", "", df_to_md(ds2_fp), ""])


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    steps = set(args.steps)

    ds1 = rr_table(DS1_RECORDS, args.data_dir)
    ds2 = rr_table(DS2_RECORDS, args.data_dir) if steps & {0, 2, 4} else None
    views = {}
    if steps & {0, 3}:
        views["DS1"] = build_eval_view(DS1_RECORDS, args.data_dir)
        reconcile_table(ds1, views["DS1"], "DS1")
    if steps & {0, 4}:
        views["DS2"] = build_eval_view(DS2_RECORDS, args.data_dir)
        reconcile_table(ds2, views["DS2"], "DS2")

    if 0 in steps:
        step0(args, {"DS1": ds1.copy(), "DS2": ds2.copy()})
    if 1 in steps:
        step1(args, ds1)
    if 2 in steps:
        step2a(args, ds1, ds2)
        step2b(args, ds1)
    if 3 in steps:
        step3(args, ds1, views["DS1"])
    if 4 in steps:
        step4(args, ds1, ds2, views["DS2"])


if __name__ == "__main__":
    main()
