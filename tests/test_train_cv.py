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
    # The defaults are the step 1b configuration.
    assert train_cv.run_config(args) == train_cv.DEFAULT_CONFIG == {"classes": "binary", "loss_weights": "matched"}
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
    # Saved symbols stay aligned with the labels and records (compare_oof.py derives AAMI classes from them).
    for rec in set(oof["record_ids"]):
        assert oof["symbols"][oof["record_ids"] == rec].tolist() == SYMBOLS.tolist()
    np.testing.assert_array_equal(oof["y"], np.isin(oof["symbols"], ["V", "A"]))
    threshold = json.loads((run / "threshold.json").read_text())["threshold"]
    assert threshold == evaluate.select_threshold(oof["y"], oof["int8"])[0] == folds["threshold"]

    rows = pd.read_csv(run / "comparison_rows.csv")
    assert rows[["model", "threshold_from"]].values.tolist() == [
        ["float", "int8_val"], ["float", "float_val"], ["int8", "int8_val"]]
    assert (rows["final_epochs"] == 1).all() and rows.loc[2, "threshold"] == threshold
    per_record = pd.read_csv(run / "oof_per_record_metrics.csv", dtype={"record_id": str})
    assert set(per_record["record_id"]) == set(train.DS1_RECORDS) - {"201"}

    assert rows["S_as_V"].isna().all() and rows["V_as_S"].isna().all()
    metrics = json.loads((run / "metrics.json").read_text())
    assert (metrics["classes"], metrics["loss_weights"]) == ("binary", "matched")
    assert metrics["loss"]["weights"] == 0.25
    assert "int8_probs" not in np.load(run / "oof_predictions.npz").files

    summary = pd.read_csv(tmp_path / "cv_comparison_seeds.csv")
    assert len(summary) == 3 and (summary["n_seeds"] == 1).all()
    assert "Grouped CV over DS1" in (tmp_path / "cv_comparison.md").read_text(encoding="utf-8")


def _write_run(artifacts_dir, name, metrics):
    run = artifacts_dir / "cv" / name
    run.mkdir(parents=True)
    (run / "folds.json").write_text("{}")
    (run / "metrics.json").write_text(json.dumps(metrics))


def test_train_cv_refuses_to_mix_configurations_in_one_dir(monkeypatch, tmp_path):
    _write_run(tmp_path, "bn_folded_seed42", {"variant": "bn_folded"})  # written before the flags existed
    base = ["--artifacts-dir", str(tmp_path), "--seeds", "43"]

    assert _parse(monkeypatch, *base).classes == "binary"
    for argv in (["--classes", "nsv"], ["--loss-weights", "none"]):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, *base, *argv)
        with pytest.raises(SystemExit):
            _parse(monkeypatch, *base, *argv, "--aggregate-only")

    nsv_dir = tmp_path / "nsv"
    _write_run(nsv_dir, "bn_folded_seed42", {"classes": "nsv", "loss_weights": "none"})
    args = _parse(monkeypatch, "--artifacts-dir", str(nsv_dir), "--seeds", "43", "--classes", "nsv",
                  "--loss-weights", "none")
    assert train_cv.run_config(args) == {"classes": "nsv", "loss_weights": "none"}
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--artifacts-dir", str(nsv_dir), "--seeds", "43", "--classes", "nsv")


@pytest.mark.parametrize("classes, weights, label, loss", [
    ("binary", "matched", "bn_folded", ("focal", 2.0, 0.25)),
    ("binary", "none", "bn_folded+unweighted", ("focal", 2.0, None)),
    ("nsv", "matched", "bn_folded+nsv", ("categorical", 2.0, (0.75, 0.25, 0.25))),
    ("nsv", "none", "bn_folded+nsv+unweighted", ("categorical", 2.0, None)),
])
def test_train_cv_loss_and_label_per_configuration(monkeypatch, tmp_path, classes, weights, label, loss):
    from types import SimpleNamespace

    ecg_model = SimpleNamespace(
        focal_loss=lambda gamma, alpha: ("focal", gamma, alpha),
        categorical_focal_loss=lambda gamma, class_weights: ("categorical", gamma, class_weights),
    )
    args = _parse(monkeypatch, "--artifacts-dir", str(tmp_path), "--classes", classes, "--loss-weights", weights)

    assert train_cv.row_label(args) == label
    assert train_cv.make_loss(args, ecg_model) == loss


def test_train_cv_nsv_end_to_end_on_synthetic_data(monkeypatch, tmp_path):
    from ecg import data, evaluate
    from ecg.features import aami_class_metrics, sv_confusion

    monkeypatch.setattr(data, "build_dataset", _fake_build_dataset)
    monkeypatch.setattr(sys, "argv", ["train_cv", "--epochs", "1", "--seeds", "42", "--calib-samples", "10",
                                      "--artifacts-dir", str(tmp_path), "--classes", "nsv"])

    train_cv.main()

    run = tmp_path / "cv" / "bn_folded_seed42"
    metrics = json.loads((run / "metrics.json").read_text())
    assert (metrics["classes"], metrics["loss_weights"]) == ("nsv", "matched")
    assert metrics["loss"] == {"focal_gamma": 2.0, "weights": [0.75, 0.25, 0.25], "classes": ["N", "S", "V"]}

    # Saved OOF scores are 1 - P(N) of the saved probabilities, and the threshold is chosen on them.
    oof = np.load(run / "oof_predictions.npz")
    assert oof["classes"].tolist() == ["N", "S", "V"]
    for model in ("float", "int8"):
        assert oof[f"{model}_probs"].shape == (len(oof["y"]), 3)
        np.testing.assert_array_equal(oof[model], evaluate.abnormal_score(oof[f"{model}_probs"]))
    np.testing.assert_allclose(oof["int8_probs"].sum(axis=1), 1.0, atol=3 / 256)
    threshold = json.loads((run / "threshold.json").read_text())["threshold"]
    assert threshold == evaluate.select_threshold(oof["y"], oof["int8"])[0]

    # The DS2 INT8 row's Se / +P / mix-ups recompute from the saved final model at the row's threshold.
    X_spec, X_rr, y_te, _, sym_te = _fake_build_dataset(train.DS2_RECORDS, "data", return_symbols=True)
    probs = evaluate.predict_tflite(run / "model_int8.tflite", X_spec, X_rr)
    rows = pd.read_csv(run / "comparison_rows.csv")
    int8_row = rows.iloc[2]
    assert int8_row["model"] == "int8" and int8_row["variant"] == "bn_folded+nsv"
    pred_classes = evaluate.predicted_classes(probs, int8_row["threshold"])
    aami = aami_class_metrics(sym_te, pred_classes=pred_classes)
    for cls in ("S", "V"):
        # rtol: comparison_rows.csv round-trips floats through text.
        np.testing.assert_allclose((int8_row[f"se_{cls}"], int8_row[f"pp_{cls}"]), aami[cls][1:], rtol=1e-12)
    assert (int8_row["S_as_V"], int8_row["V_as_S"]) == tuple(sv_confusion(sym_te, pred_classes).values())
    assert np.isclose(int8_row["pr_auc"], evaluate.ranking_metrics(y_te, evaluate.abnormal_score(probs))["pr_auc"],
                      rtol=1e-12)

    summary = pd.read_csv(tmp_path / "cv_comparison_seeds.csv")
    assert (summary["variant"] == "bn_folded+nsv").all()
