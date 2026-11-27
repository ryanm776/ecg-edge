#!/usr/bin/env python
"""Stage 2 decision table: pooled out-of-fold (OOF) per-class AUC of grouped-CV runs (read-only, no training).

Each --runs entry is LABEL=CV_DIR, where CV_DIR holds train_cv.py's <variant>_seed<S>/ run dirs. The first
entry is the reference. Per run and seed, for the INT8 and float OOF scores (evaluate.abnormal_score):
ROC-AUC and PR-AUC of S vs. N and of V vs. N (other classes left out), plus descriptive numbers at the run's
pooled-OOF threshold: S/V recall, the N false-positive rate and the median S score.

Decision rule (diagnosis.md, "Stage 2 plan"): a change counts only if its gain over the reference exceeds the
seed-to-seed spread. The INT8 rows decide (INT8 is what's deployed, as with threshold selection); float rows
are shown alongside. The gain table reports `exceeds_spread` (gain > the larger of the two runs' std over
seeds) and `separated` (every candidate seed beats every reference seed). DS2 is never read.

N/S/V runs (train_cv.py --classes nsv) also save per-class probabilities. For them, the descriptive columns add
S vs. N AUC on P(S) and V vs. N AUC on P(V), and the S <-> V mix-ups at the threshold (evaluate.predicted_classes);
these are NaN for binary runs and never decide. A table of INT8 minus float decision metrics per run shows the
quantization gap.

Operating points (descriptive): per run, seed and model, the threshold that reaches 0.90 and 0.95 pooled-OOF
abnormal recall, with precision, TP/FP, S and V beats flagged abnormal, and for N/S/V runs Se and +P of S and V
under the call rule.

Beat labels: runs from train_cv.py save each OOF beat's symbol in oof_predictions.npz. For earlier runs the
symbols are rebuilt from the DS1 annotations, as the beat-only (Stage 2 step 1) or the all-annotation
(pre-fix) row set, whichever matches the saved record ids and labels exactly.

Writes <out-dir>/oof_comparison.md, oof_comparison_seeds.csv, oof_comparison_records.csv and
oof_comparison_operating_points.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from analyze_splits import df_to_md  # noqa: E402  (numpy/pandas only)
from diagnose_s_beats import class_vs_normal_auc  # noqa: E402  (numpy/pandas/sklearn only)
from ecg.features import NSV_CLASSES, aami_class  # noqa: E402

DECISION_METRICS = ["S_roc_auc", "S_pr_auc", "V_roc_auc", "V_pr_auc"]
# Descriptive only, N/S/V runs only (NaN for binary): AUC on the class's own probability, and S <-> V mix-ups.
CLASS_PROB_METRICS = ["S_roc_auc_pS", "S_pr_auc_pS", "V_roc_auc_pV", "V_pr_auc_pV", "S_as_V", "V_as_S"]
DESCRIPTIVE_METRICS = ["threshold", "S_recall", "V_recall", "N_fp_rate", "S_median_score"] + CLASS_PROB_METRICS
MODELS = ("int8", "float")
MIN_RECORD_S_BEATS = 20
# Descriptive operating points: the highest threshold at which pooled-OOF abnormal recall reaches each target recall.
TARGET_RECALLS = (0.90, 0.95)
# Every run: S and V beats flagged abnormal (abnormal_score > threshold). N/S/V runs only (NaN for binary): Se and +P
# of S and V under the call rule (evaluate.predicted_classes), where an S beat called V is an S miss.
CALL_RULE_METRICS = ["Se_S_call", "Se_V_call", "pP_S_call", "pP_V_call"]
OPERATING_POINT_METRICS = (["threshold", "abnormal_recall", "precision", "TP", "FP", "Se_S_flagged", "Se_V_flagged"]
                           + CALL_RULE_METRICS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=CV_DIR",
                        help="First entry is the reference, e.g. step1=artifacts/dechazal/stage2/step1_beat_rr/cv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variant", default="bn_folded")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    try:
        runs = [_split_run(r) for r in args.runs]
    except ValueError as e:
        parser.error(str(e))
    if len({label for label, _ in runs}) != len(runs):
        parser.error("--runs has duplicate labels")
    args.runs = dict(runs)
    return args


def _split_run(entry):
    label, sep, path = entry.partition("=")
    if not sep or not label or not path:
        raise ValueError(f"--runs entry {entry!r} is not LABEL=CV_DIR")
    return label, path


# ---------------------------------------------------------------- pure helpers (tested)

def oof_metrics(aami, scores, threshold, probs=None):
    """Decision metrics (per-class AUC) and descriptive metrics at `threshold` for one set of OOF scores.

    `probs`: an N/S/V run's (n, 3) probabilities (columns NSV_CLASSES), for CLASS_PROB_METRICS; NaN without.
    """
    aami, scores = np.asarray(aami), np.asarray(scores)
    out = {"threshold": float(threshold)}
    for cls in ("S", "V"):
        _, out[f"{cls}_roc_auc"], out[f"{cls}_pr_auc"] = class_vs_normal_auc(aami, scores, cls)
        out[f"{cls}_recall"] = float((scores[aami == cls] > threshold).mean())
    out["N_fp_rate"] = float((scores[aami == "N"] > threshold).mean())
    out["S_median_score"] = float(np.median(scores[aami == "S"]))
    out.update(dict.fromkeys(CLASS_PROB_METRICS, float("nan")))
    if probs is not None:
        from ecg.evaluate import predicted_classes  # imports TF; only needed for N/S/V runs

        probs = np.asarray(probs)
        for cls in ("S", "V"):
            _, out[f"{cls}_roc_auc_p{cls}"], out[f"{cls}_pr_auc_p{cls}"] = class_vs_normal_auc(
                aami, probs[:, NSV_CLASSES.index(cls)], cls)
        pred = predicted_classes(probs, threshold)
        out["S_as_V"] = int(((aami == "S") & (pred == "V")).sum())
        out["V_as_S"] = int(((aami == "V") & (pred == "S")).sum())
    return out


def operating_point(aami, scores, target_recall, probs=None):
    """Metrics at the highest threshold whose abnormal recall (S and V beats with score > threshold) reaches
    `target_recall`. Ties in the scores (INT8 is coarse) can push the achieved recall above the target.

    `probs`: an N/S/V run's (n, 3) probabilities, for CALL_RULE_METRICS; NaN without.
    """
    aami, scores = np.asarray(aami), np.asarray(scores, dtype=float)
    abnormal = aami != "N"
    ranked = np.sort(scores[abnormal])[::-1]
    kth = ranked[int(np.ceil(target_recall * len(ranked))) - 1]
    threshold = float(np.nextafter(kth, -np.inf))  # every score >= kth is flagged, under the usual `>` rule
    flagged = scores > threshold
    tp, fp = int((flagged & abnormal).sum()), int((flagged & ~abnormal).sum())
    out = {"target_recall": float(target_recall), "threshold": threshold,
           "abnormal_recall": tp / int(abnormal.sum()), "precision": tp / (tp + fp), "TP": tp, "FP": fp,
           "Se_S_flagged": float(flagged[aami == "S"].mean()), "Se_V_flagged": float(flagged[aami == "V"].mean())}
    out.update(dict.fromkeys(CALL_RULE_METRICS, float("nan")))
    if probs is not None:
        from ecg.evaluate import predicted_classes  # imports TF; only needed for N/S/V runs

        pred = predicted_classes(np.asarray(probs), threshold)
        for cls in ("S", "V"):
            hit = int(((aami == cls) & (pred == cls)).sum())
            out[f"Se_{cls}_call"] = hit / int((aami == cls).sum())
            n_called = int((pred == cls).sum())
            out[f"pP_{cls}_call"] = hit / n_called if n_called else float("nan")
    return out


def gain_table(seeds_df, reference, metrics=DECISION_METRICS):
    """Per non-reference run, model and metric: mean ± std of both runs over seeds, the gain in means, and
    whether it exceeds the seed spread (gain > max std) and whether the runs' seeds don't overlap."""
    rows = []
    for (label, model), g in seeds_df.groupby(["run", "model"], sort=False):
        if label == reference:
            continue
        ref = seeds_df[(seeds_df["run"] == reference) & (seeds_df["model"] == model)]
        for metric in metrics:
            a, b = ref[metric].to_numpy(float), g[metric].to_numpy(float)
            gain = b.mean() - a.mean()
            spread = max(np.std(a, ddof=1) if len(a) > 1 else np.nan, np.std(b, ddof=1) if len(b) > 1 else np.nan)
            separated = b.min() > a.max() if gain > 0 else b.max() < a.min()
            rows.append({"run": label, "model": model, "metric": metric,
                         "reference": _mean_std(a), "candidate": _mean_std(b), "gain": gain, "spread": spread,
                         "exceeds_spread": bool(abs(gain) > spread), "separated": bool(separated)})
    return pd.DataFrame(rows)


def int8_float_gap(seeds_df, metrics=DECISION_METRICS):
    """Per run and metric: INT8 minus float over seeds (mean ± std, and the largest |gap| of any seed)."""
    rows = []
    for label, g in seeds_df.groupby("run", sort=False):
        int8 = g[g["model"] == "int8"].set_index("seed")
        flt = g[g["model"] == "float"].set_index("seed")
        for metric in metrics:
            gap = (int8[metric] - flt[metric]).dropna().to_numpy(float)
            rows.append({"run": label, "metric": metric, "int8_minus_float": _mean_std(gap),
                         "max_abs_gap": float(np.abs(gap).max()) if len(gap) else float("nan")})
    return pd.DataFrame(rows)


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.isnan(values).all():
        return "—"
    return f"{values.mean():.3f}" if len(values) < 2 else f"{values.mean():.3f} ± {np.std(values, ddof=1):.3f}"


def match_symbols(record_ids, y, candidates):
    """Symbols of the first candidate (name -> DataFrame with record_id, y, symbol) whose rows match
    `record_ids` and `y` exactly; ValueError if none does."""
    record_ids, y = np.asarray(record_ids).astype(str), np.asarray(y)
    for name, rows in candidates.items():
        if (len(rows) == len(y) and np.array_equal(rows["record_id"].to_numpy().astype(str), record_ids)
                and np.array_equal(rows["y"].to_numpy(), y)):
            return name, rows["symbol"].to_numpy().astype(str)
    raise ValueError(f"OOF rows match none of the rebuilt row sets: {list(candidates)}")


# ---------------------------------------------------------------- data

class SymbolSource:
    """Rebuilds OOF symbols for runs that predate saving them (built once, on first use)."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._candidates = None

    def candidates(self):
        if self._candidates is None:
            import diagnose_s_beats as diag

            table = diag.rr_table(diag.DS1_RECORDS, self.data_dir)
            self._candidates = {
                "beat_only": diag.oof_rows(table.dropna(subset=diag.feature_columns("beat_only"))),
                "all_annotations": diag.oof_rows(table),
            }
        return self._candidates


def load_run(run, symbols_source):
    z = np.load(os.path.join(run, "oof_predictions.npz"), allow_pickle=True)
    if "symbols" in z.files:
        symbols, how = z["symbols"].astype(str), "saved"
    else:
        how, symbols = match_symbols(z["record_ids"], z["y"], symbols_source.candidates())
    with open(os.path.join(run, "threshold.json")) as f:
        threshold = float(json.load(f)["threshold"])
    return z, symbols, how, threshold


def collect(args):
    source = SymbolSource(args.data_dir)
    seed_rows, record_rows, op_rows = [], [], []
    for label, cv_dir in args.runs.items():
        for seed in args.seeds:
            run = os.path.join(cv_dir, f"{args.variant}_seed{seed}")
            z, symbols, how, threshold = load_run(run, source)
            aami = aami_class(symbols)
            print(f"{run}: {len(symbols)} OOF beats, symbols {how}")
            for model in MODELS:
                probs = z[f"{model}_probs"] if f"{model}_probs" in z.files else None
                seed_rows.append({"run": label, "seed": seed, "model": model, "n_beats": len(symbols),
                                  **oof_metrics(aami, z[model], threshold, probs)})
                op_rows += [{"run": label, "seed": seed, "model": model, **operating_point(aami, z[model], r, probs)}
                            for r in TARGET_RECALLS]
            record_ids = z["record_ids"].astype(str)
            for rec in sorted(set(record_ids)):
                m = record_ids == rec
                n_s, roc, _ = class_vs_normal_auc(aami[m], z["int8"][m], "S")
                if n_s >= MIN_RECORD_S_BEATS:
                    record_rows.append({"run": label, "seed": seed, "record_id": rec, "n_S": n_s, "S_roc_auc": roc,
                                        "S_recall": float((z["int8"][m][aami[m] == "S"] > threshold).mean())})
    return pd.DataFrame(seed_rows), pd.DataFrame(record_rows), pd.DataFrame(op_rows)


def write_outputs(args, seeds_df, records_df, op_df):
    os.makedirs(args.out_dir, exist_ok=True)
    reference = next(iter(args.runs))
    seeds_df.to_csv(os.path.join(args.out_dir, "oof_comparison_seeds.csv"), index=False)
    records_df.to_csv(os.path.join(args.out_dir, "oof_comparison_records.csv"), index=False)
    op_df.to_csv(os.path.join(args.out_dir, "oof_comparison_operating_points.csv"), index=False)
    op_summary = op_df[op_df["model"] == "int8"].groupby(["run", "target_recall"], sort=False)[
        OPERATING_POINT_METRICS].agg(lambda s: _mean_std(s.to_numpy(float))).reset_index()

    summary = seeds_df.groupby(["run", "model"], sort=False)[DECISION_METRICS + DESCRIPTIVE_METRICS].agg(
        lambda s: _mean_std(s.to_numpy(float))).reset_index()
    gains = gain_table(seeds_df, reference)
    per_record = records_df.pivot_table(index="record_id", columns=["run", "seed"], values="S_roc_auc")
    per_record.columns = [f"{run} s{seed}" for run, seed in per_record.columns]
    lines = [
        "# Pooled OOF per-class AUC", "",
        "Runs: " + ", ".join(f"`{k}` = `{v}`" for k, v in args.runs.items()) + f". Reference: `{reference}`.",
        f"Seeds {' '.join(map(str, args.seeds))}; mean ± sample std (ddof=1) over seeds. Scores are",
        "`abnormal_score`; S vs. N and V vs. N leave the other class out. DS2 is not used.", "",
        "## Decision: gain over the reference", "",
        "`exceeds_spread`: |gain| > the larger of the two runs' std over seeds. `separated`: every candidate",
        "seed is on the gain's side of every reference seed. The INT8 rows decide; float rows are shown alongside.", "",
        df_to_md(gains, floatfmt="{:.3f}") if len(gains) else "(single run: no gains)", "",
        "## Mean ± std over seeds", "",
        "Decision metrics first; `threshold` onward are descriptive, at each run's pooled-OOF INT8 threshold.",
        "`*_pS` / `*_pV`: S vs. N AUC on P(S), V vs. N AUC on P(V); `S_as_V` / `V_as_S`: S beats called V and",
        "V beats called S (N unless 1 - P(N) > threshold, else the likelier of S and V). N/S/V runs only (`—`",
        "for binary); descriptive, never decisive.", "",
        df_to_md(summary), "",
        "## INT8 minus float, decision metrics", "",
        "Per run, over seeds: mean ± std of (INT8 − float) and the largest |gap| of any seed.", "",
        df_to_md(int8_float_gap(seeds_df), floatfmt="{:.3f}"), "",
        "## Operating points at fixed abnormal recall, INT8 (descriptive)", "",
        "Per run and target recall (`target_recall`: 0.90, 0.95): the highest threshold at which pooled-OOF",
        "abnormal recall (S and V beats with abnormal_score > threshold) reaches the target recall. `abnormal_recall`",
        "is the recall achieved; INT8 score ties can push it above the target recall.",
        "`precision`, `TP` and `FP` are for abnormal vs. N over every OOF beat (uncapped).",
        "`Se_*_flagged` (every run): the class's beats flagged abnormal. `Se_*_call` / `pP_*_call` (N/S/V runs",
        "only, `—` for binary): Se and +P under the call rule, so an S beat called V is an S miss and counts",
        "against +P_V. Float values are in `oof_comparison_operating_points.csv`.", "",
        df_to_md(op_summary), "",
        "## Per seed", "",
        df_to_md(seeds_df, floatfmt="{:.4f}"), "",
        f"## Per record: S vs. N ROC-AUC, INT8 (records with >= {MIN_RECORD_S_BEATS} S beats)", "",
        df_to_md(per_record.reset_index(), floatfmt="{:.3f}"), "",
    ]
    path = os.path.join(args.out_dir, "oof_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def main():
    args = parse_args()
    seeds_df, records_df, op_df = collect(args)
    write_outputs(args, seeds_df, records_df, op_df)


if __name__ == "__main__":
    main()
