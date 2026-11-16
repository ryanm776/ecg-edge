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
    assert train_cv.run_config(args) == train_cv.DEFAULT_CONFIG == {"classes": "binary", "loss_weights": "matched",
                                                                    "rr_lookback": 10, "normalize_rr": False}
    assert Path(args.artifacts_dir) == Path("artifacts") / "dechazal"
    for argv in (["--variant", "baseline"], ["--seeds", "42", "42"], ["--rr-lookback", "0"]):
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
                        return_symbols=False, rr_from_all_annotations=False, rr_lookback=10):
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


@pytest.mark.slow
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
    # Without --normalize-rr no model records statistics.
    assert metrics["rr_norm"] is None and [f["rr_norm"] for f in folds["folds"]] == [None] * len(train.CV_FOLDS)
    assert not (run / "rr_norm.json").exists()
    assert "int8_probs" not in np.load(run / "oof_predictions.npz").files

    summary = pd.read_csv(tmp_path / "cv_comparison_seeds.csv")
    assert len(summary) == 3 and (summary["n_seeds"] == 1).all()
    assert "Grouped CV over DS1" in (tmp_path / "cv_comparison.md").read_text(encoding="utf-8")


@pytest.mark.slow
def test_train_cv_caps_normal_beats_in_training_only(monkeypatch, tmp_path):
    from ecg import data

    calls = []

    def capped_fake(record_list, data_dir, max_normal_per_record=None, augment=False, seed=None,
                    return_symbols=False, rr_from_all_annotations=False, rr_lookback=10):
        calls.append({"records": list(record_list), "cap": max_normal_per_record, "augment": augment})
        X_spec, X_rr, y, record_ids, symbols = _fake_build_dataset(record_list, data_dir, augment=augment,
                                                                   return_symbols=True)
        keep = np.ones(len(y), dtype=bool)
        if max_normal_per_record is not None:  # like extract_beats_from_record: the first `cap` N beats per record
            for rec in set(record_ids):
                normal = np.flatnonzero((record_ids == rec) & (y == 0))
                keep[normal[max_normal_per_record:]] = False
        out = (X_spec[keep], X_rr[keep], y[keep], record_ids[keep], symbols[keep])
        return out if return_symbols else out[:4]

    monkeypatch.setattr(data, "build_dataset", capped_fake)
    base = ["--epochs", "1", "--seeds", "42", "--calib-samples", "10", "--artifacts-dir", str(tmp_path)]
    assert _parse(monkeypatch, *base).max_normal_per_record_train == 600
    monkeypatch.setattr(sys, "argv", ["train_cv", *base, "--max-normal-per-record-train", "2"])

    train_cv.main()

    train_calls = [c for c in calls if c["augment"]]
    eval_calls = [c for c in calls if not c["augment"]]
    assert [(c["records"], c["cap"]) for c in train_calls] == [(train.DS1_RECORDS, 2)]
    assert sorted((tuple(c["records"]), c["cap"]) for c in eval_calls) == sorted(
        [(tuple(train.DS1_RECORDS), None), (tuple(train.DS2_RECORDS), None)])
    # Every OOF record keeps all 4 of its N beats, although training saw only 2 per record.
    oof = np.load(tmp_path / "cv" / "bn_folded_seed42" / "oof_predictions.npz", allow_pickle=True)
    for rec in set(oof["record_ids"]):
        assert oof["symbols"][oof["record_ids"] == rec].tolist() == SYMBOLS.tolist()


@pytest.mark.slow
def test_train_cv_balanced_records_per_fold_weights_and_monitor(monkeypatch, tmp_path):
    from ecg import cv, data
    from ecg.features import balanced_class_weights, nsv_labels

    monkeypatch.setattr(data, "build_dataset", _fake_build_dataset)
    monkeypatch.setattr(sys, "argv", ["train_cv", "--epochs", "1", "--seeds", "42", "--calib-samples", "10",
                                      "--artifacts-dir", str(tmp_path), "--classes", "nsv", "--loss-weights", "balanced"])
    fits = []
    real_fit = train.fit_model
    monkeypatch.setattr(train, "fit_model", lambda *a, **kw: fits.append(kw) or real_fit(*a, **kw))

    train_cv.main()

    run = tmp_path / "cv" / "bn_folded_seed42"
    folds = json.loads((run / "folds.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    # Each fold's weights are balanced on that fold's own training labels; the final model's on all of DS1.
    _, _, _, rec_tr, sym_tr = _fake_build_dataset(train.DS1_RECORDS, "data", augment=True, return_symbols=True)
    labels = nsv_labels(sym_tr)
    for f, (tr, _) in zip(folds["folds"], cv.fold_masks(rec_tr, train.CV_FOLDS)):
        assert f["loss"]["weights"] == pytest.approx(balanced_class_weights(labels[tr]).tolist())
        assert f["loss"]["loss_scale"] == pytest.approx(1.0)
    assert metrics["loss"]["weights"] == pytest.approx(balanced_class_weights(labels).tolist())
    assert metrics["loss"]["loss_scale"] == pytest.approx(1.0)
    assert folds["early_stopping_monitor"] == metrics["loss"]["early_stopping_monitor"] == "unweighted_val_loss"
    # Every fold model early-stops on a monitor loss; the final model has no validation data and none.
    assert [kw["monitor_loss"] is not None for kw in fits[:-1]] == [True] * len(train.CV_FOLDS)
    assert fits[-1].get("monitor_loss") is None
    assert (metrics["classes"], metrics["loss_weights"]) == ("nsv", "balanced")
    assert set(pd.read_csv(tmp_path / "cv_comparison_seeds.csv")["variant"]) == {"bn_folded+nsv+balanced"}


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
    assert train_cv.run_config(args) == {"classes": "nsv", "loss_weights": "none", "rr_lookback": 10,
                                         "normalize_rr": False}
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--artifacts-dir", str(nsv_dir), "--seeds", "43", "--classes", "nsv")
    # Runs written before --rr-lookback existed read as the default lookback, 10.
    with pytest.raises(SystemExit):
        _parse(monkeypatch, *base, "--rr-lookback", "30")
    with pytest.raises(SystemExit):
        _parse(monkeypatch, *base, "--normalize-rr")
    assert _parse(monkeypatch, *base, "--rr-lookback", "10").rr_lookback == 10


# Training labels for the loss tests: 6 N, 1 S, 3 V (binary: 6 normal, 4 abnormal).
NSV_FIT = np.array([0] * 6 + [1] + [2] * 3)
BINARY_FIT = (NSV_FIT > 0).astype(int)


@pytest.mark.parametrize("classes, weights, label, loss, monitor, scale", [
    ("binary", "matched", "bn_folded", ("focal", 2.0, 0.25), None, (0.75 * 6 + 0.25 * 4) / 10),
    ("binary", "none", "bn_folded+unweighted", ("focal", 2.0, None), None, 1.0),
    ("nsv", "matched", "bn_folded+nsv", ("categorical", 2.0, (0.75, 0.25, 0.25)), None, (0.75 * 6 + 0.25 * 4) / 10),
    ("nsv", "none", "bn_folded+nsv+unweighted", ("categorical", 2.0, None), None, 1.0),
    # Balanced: w_c = 10 / (3 * n_c), early stopping on the unweighted loss.
    ("nsv", "balanced", "bn_folded+nsv+balanced", ("categorical", 2.0, (10 / 18, 10 / 3, 10 / 9)),
     ("categorical", 2.0, None), 1.0),
])
def test_train_cv_loss_and_label_per_configuration(monkeypatch, tmp_path, classes, weights, label, loss, monitor,
                                                   scale):
    from types import SimpleNamespace

    ecg_model = SimpleNamespace(
        focal_loss=lambda gamma, alpha: ("focal", gamma, alpha),
        categorical_focal_loss=lambda gamma, class_weights: ("categorical", gamma, class_weights),
    )
    args = _parse(monkeypatch, "--artifacts-dir", str(tmp_path), "--classes", classes, "--loss-weights", weights)
    fit = NSV_FIT if classes == "nsv" else BINARY_FIT

    w = train_cv.loss_weights(args, fit)
    assert train_cv.row_label(args) == label
    built = train_cv.make_loss(args, ecg_model, w)
    assert built[:2] == loss[:2]
    assert built[2] == (None if loss[2] is None else pytest.approx(loss[2]))
    assert train_cv.monitor_loss(args, ecg_model) == monitor
    assert train_cv.early_stopping_monitor(args) == ("val_loss" if monitor is None else "unweighted_val_loss")
    assert train_cv.loss_record(args, fit, w)["loss_scale"] == pytest.approx(scale)


def test_train_cv_balanced_weights_need_nsv(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--artifacts-dir", str(tmp_path), "--loss-weights", "balanced")


@pytest.mark.slow
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
    # The synthetic training view is 4 N, 1 S, 1 V per record, so matched's loss scale is (0.75 * 4 + 0.25 * 2) / 6.
    assert metrics["loss"] == {"focal_gamma": 2.0, "weights": [0.75, 0.25, 0.25], "classes": ["N", "S", "V"],
                               "loss_scale": pytest.approx(3.5 / 6), "early_stopping_monitor": "val_loss"}

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


@pytest.mark.slow
def test_train_cv_rr_lookback_reaches_every_view_and_is_recorded(monkeypatch, tmp_path):
    from ecg import data

    calls = []

    def spy(record_list, data_dir, **kwargs):
        calls.append(kwargs.get("rr_lookback"))
        return _fake_build_dataset(record_list, data_dir, **kwargs)

    monkeypatch.setattr(data, "build_dataset", spy)
    monkeypatch.setattr(sys, "argv", ["train_cv", "--epochs", "1", "--seeds", "42", "--calib-samples", "10",
                                      "--artifacts-dir", str(tmp_path), "--classes", "nsv", "--rr-lookback", "30"])

    train_cv.main()

    assert calls == [30, 30, 30]  # DS1 OOF view, DS2, training view
    metrics = json.loads((tmp_path / "cv" / "bn_folded_seed42" / "metrics.json").read_text())
    assert (metrics["classes"], metrics["loss_weights"], metrics["rr_lookback"]) == ("nsv", "matched", 30)
    assert set(pd.read_csv(tmp_path / "cv_comparison_seeds.csv")["variant"]) == {"bn_folded+nsv+rr30"}
    # The directory now holds a lookback-30 configuration; the default lookback can't be added to it.
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--artifacts-dir", str(tmp_path), "--seeds", "43", "--classes", "nsv")


@pytest.mark.slow
def test_train_cv_normalize_rr_fits_each_model_on_its_own_training_view(monkeypatch, tmp_path):
    from ecg import cv, data, evaluate
    from ecg.features import apply_rr_normalizer, fit_rr_normalizer

    monkeypatch.setattr(data, "build_dataset", _fake_build_dataset)
    monkeypatch.setattr(sys, "argv", ["train_cv", "--epochs", "1", "--seeds", "42", "--calib-samples", "10",
                                      "--artifacts-dir", str(tmp_path), "--classes", "nsv", "--normalize-rr"])

    train_cv.main()

    run = tmp_path / "cv" / "bn_folded_seed42"
    folds = json.loads((run / "folds.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["normalize_rr"] is True
    X_spec_ev, X_rr_ev, _, rec_ev, _ = _fake_build_dataset(train.DS1_RECORDS, "data", return_symbols=True)
    _, X_rr_tr, _, rec_tr, _ = _fake_build_dataset(train.DS1_RECORDS, "data", augment=True, return_symbols=True)
    X_spec_te, X_rr_te, y_te, _, _ = _fake_build_dataset(train.DS2_RECORDS, "data", return_symbols=True)

    # Each fold's statistics are those of its own training view (DS1 minus the held-out fold, augmented),
    # so they don't depend on the held-out beats; the final model's are those of all DS1.
    stats = []
    for f, (tr, _) in zip(folds["folds"], cv.fold_masks(rec_tr, train.CV_FOLDS)):
        mean, std = fit_rr_normalizer(X_rr_tr[tr])
        np.testing.assert_allclose(f["rr_norm"]["mean"], mean)
        np.testing.assert_allclose(f["rr_norm"]["std"], std)
        stats.append((mean, std))
    final_mean, final_std = fit_rr_normalizer(X_rr_tr)
    np.testing.assert_allclose(metrics["rr_norm"]["mean"], final_mean)
    np.testing.assert_allclose(metrics["rr_norm"]["std"], final_std)
    assert json.loads((run / "rr_norm.json").read_text()) == metrics["rr_norm"]

    # Every fold model's OOF scores come from held-out inputs normalized with that fold's statistics.
    oof = np.load(run / "oof_predictions.npz")
    covered = rec_ev != "201"
    for k, (_, ho) in enumerate(cv.fold_masks(rec_ev, train.CV_FOLDS)):
        expected = evaluate.predict_tflite(run / "folds" / f"fold{k}_int8.tflite", X_spec_ev[ho],
                                           apply_rr_normalizer(X_rr_ev[ho], *stats[k]))
        np.testing.assert_array_equal(oof["int8_probs"][ho[covered]], expected)

    # The final model's DS2 row is computed on DS2 inputs normalized with the final statistics.
    probs = evaluate.predict_tflite(run / "model_int8.tflite", X_spec_te,
                                    apply_rr_normalizer(X_rr_te, final_mean, final_std))
    int8_row = pd.read_csv(run / "comparison_rows.csv").iloc[2]
    assert int8_row["model"] == "int8" and int8_row["variant"] == "bn_folded+nsv+rr_norm"
    assert np.isclose(int8_row["pr_auc"], evaluate.ranking_metrics(y_te, evaluate.abnormal_score(probs))["pr_auc"],
                      rtol=1e-12)
