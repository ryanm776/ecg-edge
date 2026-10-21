"""Argument guards and result aggregation for scripts/train.py and scripts/compare_variants.py.
No training: train.py only imports numpy/pandas at module level, so parse_args is importable."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = _load("train")
compare_variants = _load("compare_variants")


def _parse(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    return module.parse_args()


def test_train_calib_samples_defaults_to_100(monkeypatch):
    args = _parse(train, monkeypatch)
    assert args.calib_samples == 100 and args.variant == "baseline"


@pytest.mark.parametrize("argv", [
    ["--variant", "qat", "--load-model", "m.keras"],
    ["--normalize-rr"],  # firmware doesn't apply R-R normalization yet
])
def test_train_rejects_unsupported_combinations(monkeypatch, argv):
    with pytest.raises(SystemExit):
        _parse(train, monkeypatch, *argv)


def test_compare_rejects_rr_norm_with_saved_baseline(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(compare_variants, monkeypatch, "--normalize-rr")
    args = _parse(compare_variants, monkeypatch, "--variants", "bn_folded", "--normalize-rr",
                  "--calib-samples", "100", "500", "1000", "5000")
    assert args.calib_samples == [100, 500, 1000, 5000]


def test_run_name():
    assert compare_variants.run_name("bn_folded", False, 100) == "bn_folded"
    assert compare_variants.run_name("bn_folded", True, 500) == "bn_folded_rr_norm_calib500"
    assert compare_variants.run_name("qat", False, None) == "qat"


def _rows(variant, calib):
    base = {"variant": variant, "calib_samples": calib, "threshold": 0.35, "val_f1": 0.5,
            "test_precision": 0.8, "test_recall": 0.9, "test_f1": 0.85, "tp": 10, "fp": 2, "fn": 1,
            "fp_rec214": 1, "fp_excl_214": 1, "pr_auc": 0.9, "roc_auc": 0.95, "r95_threshold": 0.12345,
            "r95_test_precision": 0.7, "r95_test_recall": 0.96, "r95_fp": 5,
            "tflite_kb": None, "min_int8_levels": None}
    return pd.DataFrame([
        {**base, "model": "int8", "threshold_from": "int8_val", "tflite_kb": 40.5, "min_int8_levels": 37},
        {**base, "model": "float", "threshold_from": "float_val"},
        {**base, "model": "float", "threshold_from": "int8_val"},
    ])


def test_aggregate_orders_runs_and_writes_tables(tmp_path):
    for run, variant, calib in [("qat", "qat", None), ("bn_folded_calib500", "bn_folded", 500),
                                ("baseline", "baseline", 100), ("bn_folded", "bn_folded", 100)]:
        (tmp_path / "variants" / run).mkdir(parents=True)
        _rows(variant, calib).to_csv(tmp_path / "variants" / run / "comparison_rows.csv", index=False)

    df = compare_variants.aggregate(str(tmp_path))

    assert df["run"].drop_duplicates().tolist() == ["baseline", "bn_folded", "bn_folded_calib500", "qat"]
    assert df.loc[df["run"] == "baseline", ["model", "threshold_from"]].values.tolist() == [
        ["float", "int8_val"], ["float", "float_val"], ["int8", "int8_val"]]
    md = (tmp_path / "variant_comparison.md").read_text(encoding="utf-8")  # "—" must survive on Windows
    assert "| qat | qat | — | int8 | int8_val | 0.350 |" in md
    assert "| 0.900 | 0.950 | 0.1235 | 0.700 | 0.960 | 5 | 40.5 | 37 |" in md
    assert (tmp_path / "variant_comparison.csv").exists()


def test_aggregate_skips_runs_from_an_older_train_py(tmp_path, capsys):
    for run, rows in [("bn_folded", _rows("bn_folded", 100)),
                      ("qat", _rows("qat", None).drop(columns=["pr_auc", "r95_fp"]))]:
        (tmp_path / "variants" / run).mkdir(parents=True)
        rows.to_csv(tmp_path / "variants" / run / "comparison_rows.csv", index=False)

    df = compare_variants.aggregate(str(tmp_path))

    assert df["run"].unique().tolist() == ["bn_folded"]
    assert "Skipping qat" in capsys.readouterr().out


def test_comparison_rows_use_each_models_own_val_sweep():
    # Regression: the float @ int8_val row used to report the INT8 model's val F1.
    import numpy as np
    from types import SimpleNamespace

    from ecg import evaluate

    rng = np.random.default_rng(0)
    y_val = np.array([0] * 50 + [1] * 50)
    y_test = np.array([0] * 40 + [1] * 40)
    record_ids = np.array(["214"] * 20 + ["100"] * 60)
    good = lambda y: np.where(y == 1, rng.uniform(0.6, 1.0, len(y)), rng.uniform(0.0, 0.4, len(y)))
    noisy = lambda y: rng.uniform(0.0, 1.0, len(y))
    preds = {}
    for name, make in [("float", good), ("int8", noisy)]:
        val_probs = make(y_val)
        _, sweep = evaluate.select_threshold(y_val, val_probs)
        preds[name] = (val_probs, sweep, make(y_test))
    args = SimpleNamespace(variant="bn_folded", calib_samples=100, target_recall=0.95)

    df = train.comparison_rows("bn_folded", args, y_val, y_test, record_ids, preds,
                               threshold=0.5, float_threshold=0.5, tflite_kb=10.0, levels=200,
                               evaluate=evaluate)

    def val_f1(name):
        sweep = preds[name][1]
        return sweep.loc[sweep["threshold"] == 0.5, "f1"].iloc[0]
    float_row, int8_row = df.iloc[0], df.iloc[2]
    assert float_row["val_f1"] == val_f1("float") != val_f1("int8")
    assert int8_row["val_f1"] == val_f1("int8")
    assert float_row["roc_auc"] > int8_row["roc_auc"]
    assert float_row["r95_threshold"] == evaluate.threshold_at_recall(y_val, preds["float"][0], 0.95)
    assert pd.isna(float_row["tflite_kb"]) and int8_row["min_int8_levels"] == 200
