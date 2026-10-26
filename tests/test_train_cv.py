"""scripts/train_cv.py: argument guards, and one end-to-end run on a tiny synthetic DS1/DS2 (real TF and
INT8 conversion, 1 epoch; no MIT-BIH data)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train_cv = _load("train_cv")
train = sys.modules["train"]

SPEC_SHAPE = (33, 12, 1)  # compute_spectrogram's shape for a 250-sample window
SYMBOLS = np.array(["N", "N", "N", "N", "V", "A"], dtype=object)  # per record


def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["train_cv", *argv])
    return train_cv.parse_args()


def test_train_cv_args(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the default --artifacts-dir is relative; keep real runs out of the overwrite guard
    args = _parse(monkeypatch)
    assert args.variant == "bn_folded" and args.seeds == [42]
    assert Path(args.artifacts_dir) == Path("artifacts") / "dechazal"
    for argv in (["--variant", "baseline"], ["--seeds", "42", "42"]):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, *argv)



def test_train_cv_refuses_to_overwrite_existing_runs(monkeypatch, tmp_path):
    run = tmp_path / "cv" / "bn_folded_seed43"
    run.mkdir(parents=True)
    (run / "folds.json").write_text("{}")
    base = ["--artifacts-dir", str(tmp_path), "--seeds", "42", "43"]

    with pytest.raises(SystemExit):
        _parse(monkeypatch, *base)
    assert _parse(monkeypatch, *base, "--overwrite").overwrite
    assert _parse(monkeypatch, *base, "--aggregate-only").aggregate_only
    assert _parse(monkeypatch, "--artifacts-dir", str(tmp_path), "--seeds", "42").seeds == [42]

def _fake_build_dataset(record_list, data_dir, max_normal_per_record=None, augment=False, seed=None,
                        return_symbols=False, rr_from_all_annotations=False):
    rng = np.random.default_rng(len(record_list) + int(augment))
    n = len(SYMBOLS) * len(record_list)
    symbols = np.tile(SYMBOLS, len(record_list))
    y = np.isin(symbols, ["V", "A"]).astype(np.int32)
    X_spec = rng.uniform(size=(n, *SPEC_SHAPE)).astype(np.float32)
    X_spec[y == 1] *= 0.3  # make abnormal beats separable enough to train for an epoch
    X_rr = rng.uniform(0.5, 1.5, size=(n, 5)).astype(np.float32)
    record_ids = np.repeat(np.array(record_list, dtype=object), len(SYMBOLS))
    if return_symbols:
        return X_spec, X_rr, y, record_ids, symbols
    return X_spec, X_rr, y, record_ids


def test_train_cv_end_to_end_on_synthetic_data(monkeypatch, tmp_path):
    from ecg import data, evaluate

    monkeypatch.setattr(data, "build_dataset", _fake_build_dataset)
    monkeypatch.setattr(sys, "argv", ["train_cv", "--epochs", "1", "--seeds", "42", "--calib-samples", "10",
                                      "--artifacts-dir", str(tmp_path)])

    train_cv.main()

    run = tmp_path / "cv" / "bn_folded_seed42"
    folds = json.loads((run / "folds.json").read_text())
    assert [f["records"] for f in folds["folds"]] == train.CV_FOLDS
    assert folds["best_epochs"] == [1] * 5 and folds["final_epochs"] == 1
    for f in folds["folds"]:
        assert f["oof_int8"]["n_beats"] == len(SYMBOLS) * len(f["records"])
        assert set(f["ds2_int8_diagnostic"]) == {"pr_auc", "f1", "fp", "fn"}

    # The OOF pool is DS1 without 201, and the threshold is select_threshold on the pooled INT8 scores.
    oof = np.load(run / "oof_predictions.npz")
    assert set(oof["record_ids"]) == set(train.DS1_RECORDS) - {"201"}
    threshold = json.loads((run / "threshold.json").read_text())["threshold"]
    assert threshold == evaluate.select_threshold(oof["y"], oof["int8"])[0] == folds["threshold"]

    rows = pd.read_csv(run / "comparison_rows.csv")
    assert rows[["model", "threshold_from"]].values.tolist() == [
        ["float", "int8_val"], ["float", "float_val"], ["int8", "int8_val"]]
    assert (rows["final_epochs"] == 1).all() and rows.loc[2, "threshold"] == threshold
    per_record = pd.read_csv(run / "oof_per_record_metrics.csv", dtype={"record_id": str})
    assert set(per_record["record_id"]) == set(train.DS1_RECORDS) - {"201"}

    summary = pd.read_csv(tmp_path / "cv_comparison_seeds.csv")
    assert len(summary) == 3 and (summary["n_seeds"] == 1).all()
    assert "Grouped CV over DS1" in (tmp_path / "cv_comparison.md").read_text(encoding="utf-8")
