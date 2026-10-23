#!/usr/bin/env python
"""Run scripts/train.py once per variant and aggregate the float vs. INT8 comparison.

Each run is a separate process (Keras 3 and tf_keras, which QAT needs, can't share one)
writing to <artifacts-dir>/variants/<run>/ with --no-firmware-export. The baseline is
re-evaluated from its saved model, not retrained. Every run found under
<artifacts-dir>/variants/ (this invocation's and earlier ones) is aggregated into
<artifacts-dir>/variant_comparison.{csv,md}.

With several --calib-samples sizes, each PTQ variant is trained once (on the first size)
and the saved float model is re-converted for the remaining sizes. QAT needs no
calibration set, so it runs once regardless.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "scripts", "train.py")
VARIANTS = ["baseline", "bn_folded", "qat"]
DEFAULT_CALIB_SAMPLES = 100  # train.py's --calib-samples default
COLUMNS = [
    "run", "variant", "calib_samples", "model", "threshold_from", "threshold", "val_f1",
    "test_precision", "test_recall", "test_f1", "tp", "fp", "fn", "fp_rec214", "fp_excl_214",
    "f1_excl_202", "fp_excl_202", "fn_excl_202", "pr_auc_excl_202",
    "pr_auc", "roc_auc", "r95_threshold", "r95_test_precision", "r95_test_recall", "r95_fp",
    "tflite_kb", "min_int8_levels",
]
FLOAT_COLUMNS_3DP = [
    "threshold", "val_f1", "test_precision", "test_recall", "test_f1",
    "f1_excl_202", "pr_auc_excl_202", "pr_auc", "roc_auc", "r95_test_precision", "r95_test_recall",
]
FLOAT_COLUMNS_4DP = ["r95_threshold"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--normalize-rr", action="store_true", help="Pass --normalize-rr to every run.")
    parser.add_argument("--calib-samples", nargs="+", type=int, default=[DEFAULT_CALIB_SAMPLES],
                        help="PTQ calibration-set sizes to evaluate (e.g. 100 500 1000 5000).")
    parser.add_argument("--load-model", default=None,
                        help="Reuse this saved float model for the PTQ (non-baseline) variants instead of training.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--baseline-model", default=None,
                        help="Saved baseline model (default: <artifacts-dir>/ecg_multiscale_model.keras).")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregate-only", action="store_true", help="Skip the runs; just rebuild the tables.")
    args = parser.parse_args()

    if args.normalize_rr and "baseline" in args.variants:
        parser.error("the saved baseline model was trained without R-R normalization; drop baseline from --variants")
    return args


def run_name(variant, normalize_rr, calib_samples):
    name = variant + ("_rr_norm" if normalize_rr else "")
    if calib_samples is not None and calib_samples != DEFAULT_CALIB_SAMPLES:
        name += f"_calib{calib_samples}"
    return name


def run_variants(args):
    baseline_model = args.baseline_model or os.path.join(args.artifacts_dir, "ecg_multiscale_model.keras")

    for variant in args.variants:
        sizes = [None] if variant == "qat" else args.calib_samples
        trained_model = baseline_model if variant == "baseline" else (None if variant == "qat" else args.load_model)

        for calib_samples in sizes:
            out_dir = os.path.join(args.artifacts_dir, "variants", run_name(variant, args.normalize_rr, calib_samples))
            cmd = [
                sys.executable, TRAIN_SCRIPT,
                "--variant", variant,
                "--artifacts-dir", out_dir,
                "--data-dir", args.data_dir,
                "--epochs", str(args.epochs),
                "--seed", str(args.seed),
                "--no-firmware-export",
            ]
            if args.normalize_rr:
                cmd.append("--normalize-rr")
            if calib_samples is not None:
                cmd += ["--calib-samples", str(calib_samples)]
            if trained_model:
                cmd += ["--load-model", trained_model]

            env = dict(os.environ)
            if variant == "qat":
                env["TF_USE_LEGACY_KERAS"] = "1"
            print(f"\n=== {out_dir} ===\n{' '.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

            if trained_model is None and variant != "qat":
                trained_model = os.path.join(out_dir, "ecg_multiscale_model.keras")


def aggregate(artifacts_dir):
    paths = sorted(glob.glob(os.path.join(artifacts_dir, "variants", "*", "comparison_rows.csv")))
    if not paths:
        raise SystemExit(f"No runs found under {artifacts_dir}/variants/")
    frames = []
    for p in paths:
        run = os.path.basename(os.path.dirname(p))
        frame = pd.read_csv(p, dtype={"variant": str}).assign(run=run)
        missing = [c for c in COLUMNS if c not in frame.columns]
        if missing:
            print(f"Skipping {run}: written by an older train.py (missing {', '.join(missing)}); re-run it to include it.")
            continue
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No up-to-date runs found under {artifacts_dir}/variants/")
    df = pd.concat(frames, ignore_index=True)

    base = df["variant"].str.split("+").str[0]
    df = df.assign(
        _variant=base.map({v: i for i, v in enumerate(VARIANTS)}),
        _rr_norm=df["variant"].str.contains(r"\+rr_norm", regex=True),
        _calib=df["calib_samples"].fillna(-1),
        _model=df["model"].map({"float": 0, "int8": 1}),
        _from=df["threshold_from"].map({"int8_val": 0, "float_val": 1}),
    ).sort_values(["_variant", "_rr_norm", "_calib", "_model", "_from"])
    df = df[COLUMNS]

    csv_path = os.path.join(artifacts_dir, "variant_comparison.csv")
    md_path = os.path.join(artifacts_dir, "variant_comparison.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(df))
    print(f"\nWrote {csv_path} and {md_path} ({len(frames)} runs).")
    return df


def to_markdown(df):
    def fmt(col, value):
        if pd.isna(value):
            return "—"
        if col in FLOAT_COLUMNS_3DP:
            return f"{value:.3f}"
        if col in FLOAT_COLUMNS_4DP:
            return f"{value:.4f}"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    lines = [
        "# Float vs. INT8 by variant",
        "",
        "Test set, seed 42. `threshold_from`: `int8_val` is the threshold chosen on the INT8 model's",
        "validation predictions (the deployed convention); `float_val` is the float model's own",
        "val-optimal threshold. For `qat`, the float row is the float model before QAT fine-tuning.",
        "`val_f1` is the row's own model's validation F1 at the row's threshold.",
        "Threshold-independent columns: `pr_auc` (average precision) and `roc_auc` on test, and the",
        "fixed-recall operating point `r95_*`: the highest threshold reaching >= 0.95 recall on the",
        "model's own validation predictions, then test precision / recall / FP at that threshold.",
        "`min_int8_levels` is the smallest number of INT8 levels any intermediate tensor spans.",
        "`*_excl_202`: sensitivity check, test re-evaluated without record 202 (same patient as train",
        "record 201); reported alongside the full-test columns, not instead of them.",
        "",
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(c, r[c]) for c in df.columns) + " |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if not args.aggregate_only:
        run_variants(args)
    aggregate(args.artifacts_dir)


if __name__ == "__main__":
    main()
