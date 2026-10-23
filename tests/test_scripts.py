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
        _parse(compare_variants, monkeypatch, "--normalize-rr", "--baseline-model", "m.keras")
    args = _parse(compare_variants, monkeypatch, "--variants", "bn_folded", "--normalize-rr",
                  "--calib-samples", "100", "500", "1000", "5000")
    assert args.calib_samples == [100, 500, 1000, 5000]


def test_compare_defaults_to_dechazal_dir_and_trains_baseline(monkeypatch):
    # artifacts/ holds the old-split record; the saved baseline there was trained on the old
    # split, so reusing it on the new one would leak val/test patients.
    args = _parse(compare_variants, monkeypatch)
    assert Path(args.artifacts_dir) == Path("artifacts") / "dechazal"
    assert args.baseline_model is None


def test_run_name():
    assert compare_variants.run_name("bn_folded", False, 100, 42) == "bn_folded_seed42"
    assert compare_variants.run_name("bn_folded", True, 500, 43) == "bn_folded_rr_norm_calib500_seed43"
    assert compare_variants.run_name("qat", False, None, 44) == "qat_seed44"


def test_compare_seeds_args(monkeypatch):
    assert _parse(compare_variants, monkeypatch).seeds == [42]
    assert _parse(compare_variants, monkeypatch, "--seeds", "42", "43", "44").seeds == [42, 43, 44]
    for argv in (["--seeds", "42", "43", "--load-model", "m.keras"],
                 ["--seeds", "42", "43", "--baseline-model", "m.keras"],
                 ["--seeds", "42", "42"]):
        with pytest.raises(SystemExit):
            _parse(compare_variants, monkeypatch, *argv)


def test_run_variants_trains_each_variant_once_per_seed(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(compare_variants.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    args = _parse(compare_variants, monkeypatch, "--variants", "baseline", "bn_folded",
                  "--seeds", "42", "43", "--calib-samples", "100", "500", "--artifacts-dir", str(tmp_path))

    compare_variants.run_variants(args)

    def opt(cmd, flag):
        return cmd[cmd.index(flag) + 1] if flag in cmd else None
    runs = [(Path(opt(c, "--artifacts-dir")).name, opt(c, "--seed"), opt(c, "--load-model")) for c in calls]
    # Each (variant, seed) trains on the first calibration size; the second size re-converts
    # that same seed's model, never another seed's.
    model = lambda run: str(tmp_path / "variants" / run / "ecg_multiscale_model.keras")
    assert runs == [
        ("baseline_seed42", "42", None), ("baseline_calib500_seed42", "42", model("baseline_seed42")),
        ("bn_folded_seed42", "42", None), ("bn_folded_calib500_seed42", "42", model("bn_folded_seed42")),
        ("baseline_seed43", "43", None), ("baseline_calib500_seed43", "43", model("baseline_seed43")),
        ("bn_folded_seed43", "43", None), ("bn_folded_calib500_seed43", "43", model("bn_folded_seed43")),
    ]


def _rows(variant, calib, seed=42):
    base = {"variant": variant, "calib_samples": calib, "seed": seed, "threshold": 0.35, "val_f1": 0.5,
            "test_precision": 0.8, "test_recall": 0.9, "test_f1": 0.85, "tp": 10, "fp": 2, "fn": 1,
            "recall_ventricular": 0.95, "recall_supraventricular": 0.4, "fp_rec214": 1, "fp_excl_214": 1,
            "f1_excl_202": 0.84, "fp_excl_202": 1, "fn_excl_202": 1, "pr_auc_excl_202": 0.88, "pr_auc": 0.9, "roc_auc": 0.95, "r95_threshold": 0.12345,
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
    assert "| qat | qat | — | 42 | int8 | int8_val | 0.350 |" in md
    assert "| 0.900 | 0.950 | 0.1235 | 0.700 | 0.960 | 5 | 40.5 | 37 |" in md
    assert (tmp_path / "variant_comparison.csv").exists()


def test_aggregate_reports_mean_and_std_over_seeds(tmp_path):
    per_seed = {42: (0.80, 10, 4, 0.30), 43: (0.90, 20, 6, 0.50), 44: (0.85, 30, 8, 0.40)}
    for seed, (pr_auc, fp, fn, s_recall) in per_seed.items():
        rows = _rows("bn_folded", 100, seed).assign(pr_auc=pr_auc, fp=fp, fn=fn, recall_supraventricular=s_recall)
        (tmp_path / "variants" / f"bn_folded_seed{seed}").mkdir(parents=True)
        rows.to_csv(tmp_path / "variants" / f"bn_folded_seed{seed}" / "comparison_rows.csv", index=False)
    (tmp_path / "variants" / "qat_seed42").mkdir(parents=True)
    _rows("qat", None).to_csv(tmp_path / "variants" / "qat_seed42" / "comparison_rows.csv", index=False)

    compare_variants.aggregate(str(tmp_path))

    summary = pd.read_csv(tmp_path / "variant_comparison_seeds.csv")
    int8 = summary[(summary["variant"] == "bn_folded") & (summary["model"] == "int8")].iloc[0]
    assert int8["n_seeds"] == 3 and int8["seeds"] == "42 43 44"
    assert int8["pr_auc_mean"] == pytest.approx(0.85) and int8["pr_auc_std"] == pytest.approx(0.05)
    assert int8["fp_mean"] == pytest.approx(20) and int8["fp_std"] == pytest.approx(10)
    assert int8["recall_supraventricular_std"] == pytest.approx(0.1)
    # 3 bn_folded configs (float@int8_val, float@float_val, int8) + 3 qat configs; qat has one seed.
    assert len(summary) == 6
    qat = summary[summary["variant"] == "qat"]
    assert (qat["n_seeds"] == 1).all() and qat["pr_auc_std"].isna().all()

    md = (tmp_path / "variant_comparison.md").read_text(encoding="utf-8")
    assert "| bn_folded | 100 | int8 | int8_val | 3 | 42 43 44 | 0.850 ± 0.050 | 0.850 ± 0.000 | 20.0 ± 10.0 | 6.0 ± 2.0 |" in md
    assert "| qat | — | int8 | int8_val | 1 | 42 | 0.900 ± — |" in md


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
    record_ids = np.array(["214"] * 20 + ["202"] * 20 + ["100"] * 40, dtype=object)
    symbols = np.array(["N"] * 40 + ["V"] * 25 + ["A"] * 15, dtype=object)
    good = lambda y: np.where(y == 1, rng.uniform(0.6, 1.0, len(y)), rng.uniform(0.0, 0.4, len(y)))
    noisy = lambda y: rng.uniform(0.0, 1.0, len(y))
    preds = {}
    for name, make in [("float", good), ("int8", noisy)]:
        val_probs = make(y_val)
        _, sweep = evaluate.select_threshold(y_val, val_probs)
        preds[name] = (val_probs, sweep, make(y_test))
    args = SimpleNamespace(variant="bn_folded", calib_samples=100, target_recall=0.95, seed=43)

    df = train.comparison_rows("bn_folded", args, y_val, y_test, record_ids, symbols, preds,
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

    # Every row's excl-202 columns equal the metrics computed directly on the masked arrays,
    # whatever the model's quality.
    keep = record_ids != "202"
    for _, row_ in df.iterrows():
        test_probs = preds[row_["model"]][2]
        excl, _ = evaluate.evaluate_predictions(y_test[keep], test_probs[keep], record_ids[keep], row_["threshold"])
        (_, fp), (fn, _) = excl["confusion_matrix"]
        assert row_["f1_excl_202"] == excl["f1"]
        assert (row_["fp_excl_202"], row_["fn_excl_202"]) == (fp, fn)
        assert row_["pr_auc_excl_202"] == evaluate.ranking_metrics(y_test[keep], test_probs[keep])["pr_auc"]
        assert row_["n_beats"] == len(y_test)
        assert row_["n_beats_excl_202"] == row_["n_beats"] - int((record_ids == "202").sum())
        # Per-type recall: the fraction of that type's beats above the row's threshold.
        pred = test_probs > row_["threshold"]
        assert row_["recall_ventricular"] == pred[symbols == "V"].mean()
        assert row_["recall_supraventricular"] == pred[symbols == "A"].mean()
        # AAMI Se is the same quantity for a binary model; +P is undefined for it.
        assert (row_["se_V"], row_["se_S"]) == (row_["recall_ventricular"], row_["recall_supraventricular"])
        assert pd.isna(row_["pp_V"]) and pd.isna(row_["pp_S"])
        assert row_["seed"] == 43


def _fake_fit_env(val_loss):
    from types import SimpleNamespace

    calls = {}

    class Model:
        def compile(self, **kw):
            calls["compile"] = kw

        def fit(self, **kw):
            calls["fit"] = kw
            return SimpleNamespace(history={"val_loss": val_loss})

    keras = SimpleNamespace(
        optimizers=SimpleNamespace(Adam=lambda learning_rate: ("adam", learning_rate)),
        callbacks=SimpleNamespace(EarlyStopping=lambda **kw: ("early_stop", kw)),
    )
    ecg_model = SimpleNamespace(focal_loss=lambda gamma, alpha: ("focal", gamma, alpha))
    return Model(), SimpleNamespace(keras=keras), ecg_model, calls


def test_fit_model_returns_best_epoch_with_early_stopping():
    model, tf, ecg_model, calls = _fake_fit_env([0.5, 0.3, 0.4, 0.35])

    best = train.fit_model(model, tf, ecg_model, "x", "y", ("xv", "yv"), lr=0.001, epochs=100, batch_size=64)

    assert best == 2
    assert calls["compile"]["loss"] == ("focal", 2.0, 0.25)
    (_, es), = calls["fit"]["callbacks"]
    assert es == {"monitor": "val_loss", "patience": 12, "restore_best_weights": True}
    assert calls["fit"]["epochs"] == 100 and calls["fit"]["validation_data"] == ("xv", "yv")


def test_fit_model_without_validation_trains_all_epochs():
    model, tf, ecg_model, calls = _fake_fit_env([])

    assert train.fit_model(model, tf, ecg_model, "x", "y", None, lr=0.001, epochs=23, batch_size=64) == 23
    assert calls["fit"]["callbacks"] == [] and calls["fit"]["validation_data"] is None


def test_val_per_record_rows_reconcile_with_overall_val_metrics():
    import numpy as np

    from ecg import evaluate

    rng = np.random.default_rng(1)
    y_val = np.array(([0] * 30 + [1] * 10) * 3)
    record_ids = np.repeat(np.array(["207", "220", "223"], dtype=object), 40)
    preds = {name: (rng.uniform(0.0, 1.0, len(y_val)), None, None) for name in ("float", "int8")}

    df = train.val_per_record_rows(y_val, record_ids, preds, threshold=0.5, float_threshold=0.3,
                                   evaluate=evaluate)

    rows = df[["model", "threshold_from", "threshold"]].drop_duplicates().values.tolist()
    assert rows == [["float", "int8_val", 0.5], ["float", "float_val", 0.3], ["int8", "int8_val", 0.5]]
    for (model, _, t), group in df.groupby(["model", "threshold_from", "threshold"]):
        probs = preds[model][0]
        assert sorted(group["record_id"]) == ["207", "220", "223"]
        # Per-record counts sum to the overall val confusion matrix at the row's threshold...
        overall, _ = evaluate.evaluate_predictions(y_val, probs, record_ids, t)
        (tn, fp), (fn, tp) = overall["confusion_matrix"]
        assert (group["tp"].sum(), group["fp"].sum(), group["fn"].sum(), group["tn"].sum()) == (tp, fp, fn, tn)
        assert group["n_beats"].sum() == len(y_val)
        # ...and each record's row equals metrics computed on that record's beats alone.
        for _, r in group.iterrows():
            mask = record_ids == r["record_id"]
            alone, _ = evaluate.evaluate_predictions(y_val[mask], probs[mask], record_ids[mask], t)
            assert r["f1"] == alone["f1"]
            assert (r["fp"], r["fn"]) == (alone["confusion_matrix"][0][1], alone["confusion_matrix"][1][0])
