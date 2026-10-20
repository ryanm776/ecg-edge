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
            "fp_rec214": 1, "fp_excl_214": 1, "tflite_kb": None, "min_int8_levels": None}
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
    md = (tmp_path / "variant_comparison.md").read_text()
    assert "| qat | qat | — | int8 | int8_val | 0.350 |" in md
    assert "| 40.5 | 37 |" in md
    assert (tmp_path / "variant_comparison.csv").exists()
