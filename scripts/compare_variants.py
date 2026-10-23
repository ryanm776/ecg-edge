#!/usr/bin/env python
"""Run scripts/train.py once per variant and aggregate the float vs. INT8 comparison.

Each run is a separate process (Keras 3 and tf_keras, which QAT needs, can't share one)
writing to <artifacts-dir>/variants/<run>/ with --no-firmware-export. Every run found under
<artifacts-dir>/variants/ (this invocation's and earlier ones) is aggregated into
<artifacts-dir>/variant_comparison.{csv,md}.

<artifacts-dir> defaults to artifacts/dechazal (the de Chazal split). artifacts/variants/ and
artifacts/variant_comparison.* are the record of the old hand-picked split; don't write there.
Every variant, baseline included, is trained unless a saved model is passed explicitly; a
saved model must have been trained on the same split, or val/test patients leak into training.

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
import time
from dataclasses import dataclass, field

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "scripts", "train.py")
VARIANTS = ["baseline", "bn_folded", "qat"]
DEFAULT_CALIB_SAMPLES = 100  # train.py's --calib-samples default
COLUMNS = [
    "run", "variant", "calib_samples", "seed", "model", "threshold_from", "threshold", "val_f1",
    "test_precision", "test_recall", "test_f1", "tp", "fp", "fn",
    "recall_ventricular", "recall_supraventricular", "fp_rec214", "fp_excl_214",
    "f1_excl_202", "fp_excl_202", "fn_excl_202", "pr_auc_excl_202",
    "pr_auc", "roc_auc", "r95_threshold", "r95_test_precision", "r95_test_recall", "r95_fp",
    "tflite_kb", "min_int8_levels",
]
FLOAT_COLUMNS_3DP = [
    "threshold", "val_f1", "test_precision", "test_recall", "test_f1",
    "recall_ventricular", "recall_supraventricular", "se_V", "pp_V", "se_S", "pp_S",
    "f1_excl_202", "pr_auc_excl_202", "pr_auc", "roc_auc", "r95_test_precision", "r95_test_recall",
]
FLOAT_COLUMNS_4DP = ["r95_threshold"]
# Rows that differ only in seed are one configuration; these are averaged over its seeds.
CONFIG_COLUMNS = ["variant", "calib_samples", "model", "threshold_from"]
SEED_SUMMARY_COLUMNS = [
    "pr_auc", "test_f1", "fp", "fn", "recall_ventricular", "recall_supraventricular",
    "f1_excl_202", "fp_excl_202", "fn_excl_202", "pr_auc_excl_202", "threshold",
]


@dataclass(frozen=True)
class Table:
    """One aggregated comparison: runs under <artifacts_dir>/<subdir>/*/comparison_rows.csv are written to
    <artifacts_dir>/<name>.{csv,md} and <name>_seeds.csv."""
    subdir: str
    name: str
    title: str
    columns: list
    summary_columns: list
    notes: list = field(default_factory=list)


VARIANT_TABLE = Table(
    subdir="variants", name="variant_comparison", title="Float vs. INT8 by variant",
    columns=COLUMNS, summary_columns=SEED_SUMMARY_COLUMNS,
    notes=[
        "Test set, one row per run (seed in `seed`). `threshold_from`: `int8_val` is the threshold chosen on the INT8 model's",
        "validation predictions (the deployed convention); `float_val` is the float model's own",
        "val-optimal threshold. For `qat`, the float row is the float model before QAT fine-tuning.",
        "`val_f1` is the row's own model's validation F1 at the row's threshold.",
        "Threshold-independent columns: `pr_auc` (average precision) and `roc_auc` on test, and the",
        "fixed-recall operating point `r95_*`: the highest threshold reaching >= 0.95 recall on the",
        "model's own validation predictions, then test precision / recall / FP at that threshold.",
        "`min_int8_levels` is the smallest number of INT8 levels any intermediate tensor spans.",
        "`*_excl_202`: sensitivity check, test re-evaluated without record 202 (same patient as train",
        "record 201); reported alongside the full-test columns, not instead of them.",
        "`recall_ventricular` (V, E) / `recall_supraventricular` (A, a, J, S): test recall per abnormal type.",
    ],
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--normalize-rr", action="store_true", help="Pass --normalize-rr to every run.")
    parser.add_argument("--calib-samples", nargs="+", type=int, default=[DEFAULT_CALIB_SAMPLES],
                        help="PTQ calibration-set sizes to evaluate (e.g. 100 500 1000 5000).")
    parser.add_argument("--load-model", default=None,
                        help="Reuse this saved float model for the PTQ (non-baseline) variants instead of training "
                             "(must be from the same split).")
    parser.add_argument("--artifacts-dir", default=os.path.join("artifacts", "dechazal"),
                        help="Default: artifacts/dechazal. artifacts/ itself holds the old-split record.")
    parser.add_argument("--baseline-model", default=None,
                        help="Reuse this saved baseline model instead of training it (must be from the same split).")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Train every variant once per seed (e.g. 42 43 44); tables report mean ± std.")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip the runs; just rebuild the tables.")
    args = parser.parse_args()

    if args.normalize_rr and args.baseline_model:
        parser.error("--baseline-model was trained without R-R normalization; drop it with --normalize-rr")
    if len(args.seeds) > 1 and (args.baseline_model or args.load_model):
        parser.error("a saved model is one seed's training; it can't be reused across --seeds")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds has duplicates")
    return args


def run_name(variant, normalize_rr, calib_samples, seed):
    name = variant + ("_rr_norm" if normalize_rr else "")
    if calib_samples is not None and calib_samples != DEFAULT_CALIB_SAMPLES:
        name += f"_calib{calib_samples}"
    return name + f"_seed{seed}"


def run_variants(args):
    for seed in args.seeds:
        for variant in args.variants:
            run_variant(args, variant, seed)


def run_variant(args, variant, seed):
    sizes = [None] if variant == "qat" else args.calib_samples
    trained_model = args.baseline_model if variant == "baseline" else (None if variant == "qat" else args.load_model)

    for calib_samples in sizes:
        out_dir = os.path.join(args.artifacts_dir, "variants",
                               run_name(variant, args.normalize_rr, calib_samples, seed))
        cmd = train_command(args, variant, seed, calib_samples, trained_model, out_dir)

        env = dict(os.environ)
        if variant == "qat":
            env["TF_USE_LEGACY_KERAS"] = "1"
        print(f"\n=== {out_dir} ===\n{' '.join(cmd)}", flush=True)
        start = time.monotonic()
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        print(f"=== {out_dir} finished in {(time.monotonic() - start) / 60:.1f} min ===", flush=True)

        if trained_model is None and variant != "qat":
            trained_model = os.path.join(out_dir, "ecg_multiscale_model.keras")


def train_command(args, variant, seed, calib_samples, trained_model, out_dir):
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--variant", variant,
        "--artifacts-dir", out_dir,
        "--data-dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--seed", str(seed),
        "--no-firmware-export",
    ]
    if args.normalize_rr:
        cmd.append("--normalize-rr")
    if calib_samples is not None:
        cmd += ["--calib-samples", str(calib_samples)]
    if trained_model:
        cmd += ["--load-model", trained_model]
    return cmd


def aggregate(artifacts_dir, table=VARIANT_TABLE):
    runs_dir = os.path.join(artifacts_dir, table.subdir)
    paths = sorted(glob.glob(os.path.join(runs_dir, "*", "comparison_rows.csv")))
    if not paths:
        raise SystemExit(f"No runs found under {runs_dir}/")
    frames = []
    for p in paths:
        run = os.path.basename(os.path.dirname(p))
        frame = pd.read_csv(p, dtype={"variant": str}).assign(run=run)
        missing = [c for c in table.columns if c not in frame.columns]
        if missing:
            print(f"Skipping {run}: written by an older train.py (missing {', '.join(missing)}); re-run it to include it.")
            continue
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No up-to-date runs found under {runs_dir}/")
    df = pd.concat(frames, ignore_index=True)

    base = df["variant"].str.split("+").str[0]
    df = df.assign(
        _variant=base.map({v: i for i, v in enumerate(VARIANTS)}),
        _rr_norm=df["variant"].str.contains(r"\+rr_norm", regex=True),
        _calib=df["calib_samples"].fillna(-1),
        _model=df["model"].map({"float": 0, "int8": 1}),
        _from=df["threshold_from"].map({"int8_val": 0, "float_val": 1}),
    ).sort_values(["_variant", "_rr_norm", "_calib", "_model", "_from", "seed"])
    df = df[table.columns]
    summary = seed_summary(df, table.summary_columns)

    csv_path = os.path.join(artifacts_dir, f"{table.name}.csv")
    seeds_csv_path = os.path.join(artifacts_dir, f"{table.name}_seeds.csv")
    md_path = os.path.join(artifacts_dir, f"{table.name}.md")
    df.to_csv(csv_path, index=False)
    summary.to_csv(seeds_csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(df, summary, table))
    print(f"\nWrote {csv_path}, {seeds_csv_path} and {md_path} ({len(frames)} runs).")
    return df


def seed_summary(df, summary_columns=SEED_SUMMARY_COLUMNS):
    """Mean and sample std (ddof=1; NaN with one seed) of `summary_columns` per configuration,
    i.e. over the rows that differ only in seed. Keeps the per-run table's row order."""
    groups = df.groupby(CONFIG_COLUMNS, dropna=False, sort=False)
    out = groups["seed"].agg(n_seeds="count", seeds=lambda s: " ".join(str(int(v)) for v in sorted(s)))
    for col in summary_columns:
        out[f"{col}_mean"] = groups[col].mean()
        out[f"{col}_std"] = groups[col].std(ddof=1)
    return out.reset_index()


def _fmt(col, value):
    if pd.isna(value):
        return "—"
    if col in FLOAT_COLUMNS_3DP:
        return f"{value:.3f}"
    if col in FLOAT_COLUMNS_4DP:
        return f"{value:.4f}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fmt_mean_std(col, mean, std):
    if col in FLOAT_COLUMNS_3DP or col in FLOAT_COLUMNS_4DP:
        return f"{_fmt(col, mean)} ± {_fmt(col, std)}"
    # Counts (fp, fn, ...): one decimal so a non-integer mean isn't rounded away.
    return f"{mean:.1f} ± {'—' if pd.isna(std) else f'{std:.1f}'}"


def to_markdown(df, summary, table=VARIANT_TABLE):
    summary_cols = CONFIG_COLUMNS + ["n_seeds", "seeds"] + table.summary_columns
    lines = [
        f"# {table.title}",
        "",
        "## Mean ± std over seeds",
        "",
        "Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full",
        "test set; float and INT8 are separate rows.",
        "",
        "| " + " | ".join(summary_cols) + " |",
        "|" + "|".join("---" for _ in summary_cols) + "|",
    ]
    for _, r in summary.iterrows():
        cells = [_fmt(c, r[c]) for c in CONFIG_COLUMNS + ["n_seeds", "seeds"]]
        cells += [_fmt_mean_std(c, r[f"{c}_mean"], r[f"{c}_std"]) for c in table.summary_columns]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Per run",
        "",
        *table.notes,
        "",
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(c, r[c]) for c in df.columns) + " |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if not args.aggregate_only:
        run_variants(args)
    aggregate(args.artifacts_dir)


if __name__ == "__main__":
    main()
