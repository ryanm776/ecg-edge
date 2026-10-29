"""scripts/compare_oof.py: metrics, gain rule, label matching, and one run over synthetic run dirs (no data)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("compare_oof", SCRIPTS / "compare_oof.py")
compare_oof = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_oof)


def _oof(rng, n=400):
    aami = rng.choice(np.array(["N", "S", "V"]), n, p=[0.8, 0.1, 0.1])
    scores = rng.uniform(size=n) + 0.3 * (aami != "N")
    return aami, np.clip(scores, 0, 1)


def test_oof_metrics_match_sklearn_on_each_class_vs_normal():
    aami, scores = _oof(np.random.default_rng(0))

    m = compare_oof.oof_metrics(aami, scores, 0.6)

    for cls in ("S", "V"):
        keep = np.isin(aami, ["N", cls])
        y = aami[keep] == cls
        assert m[f"{cls}_roc_auc"] == roc_auc_score(y, scores[keep])
        assert m[f"{cls}_pr_auc"] == average_precision_score(y, scores[keep])
        assert m[f"{cls}_recall"] == (scores[aami == cls] > 0.6).mean()
    assert m["N_fp_rate"] == (scores[aami == "N"] > 0.6).mean()
    assert m["S_median_score"] == np.median(scores[aami == "S"]) and m["threshold"] == 0.6


def _seeds(ref, cand, metric="S_roc_auc"):
    rows = [{"run": "ref", "seed": s, "model": "int8", metric: v} for s, v in zip((42, 43, 44), ref)]
    rows += [{"run": "new", "seed": s, "model": "int8", metric: v} for s, v in zip((42, 43, 44), cand)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("ref,cand,exceeds,separated", [
    ([0.70, 0.71, 0.72], [0.80, 0.81, 0.82], True, True),   # gain 0.10, spread 0.01
    ([0.70, 0.78, 0.86], [0.80, 0.81, 0.82], False, False),  # pre-fix-like spread swamps the gain
    ([0.70, 0.71, 0.72], [0.60, 0.61, 0.62], True, True),   # a loss beyond the spread
])
def test_gain_table_applies_the_spread_rule(ref, cand, exceeds, separated):
    g = compare_oof.gain_table(_seeds(ref, cand), "ref", metrics=["S_roc_auc"]).iloc[0]

    assert g["gain"] == pytest.approx(np.mean(cand) - np.mean(ref))
    assert g["spread"] == pytest.approx(max(np.std(ref, ddof=1), np.std(cand, ddof=1)))
    assert g["exceeds_spread"] == exceeds and g["separated"] == separated
    assert g["run"] == "new"  # the reference gets no row of its own


def test_match_symbols_picks_the_row_set_that_matches_exactly():
    beat_only = pd.DataFrame({"record_id": ["a", "a", "b"], "y": [0, 1, 0], "symbol": ["N", "A", "N"]})
    all_ann = pd.DataFrame({"record_id": ["a", "a", "a", "b"], "y": [0, 0, 1, 0], "symbol": ["N", "N", "A", "N"]})
    candidates = {"beat_only": beat_only, "all_annotations": all_ann}

    assert compare_oof.match_symbols(["a", "a", "a", "b"], [0, 0, 1, 0], candidates)[0] == "all_annotations"
    name, symbols = compare_oof.match_symbols(["a", "a", "b"], [0, 1, 0], candidates)
    assert name == "beat_only" and symbols.tolist() == ["N", "A", "N"]
    with pytest.raises(ValueError):
        compare_oof.match_symbols(["a", "a", "b"], [1, 1, 0], candidates)


def _write_run(path, rng, threshold):
    path.mkdir(parents=True)
    symbols = rng.choice(np.array(["N", "A", "V"]), 300, p=[0.8, 0.1, 0.1])
    records = np.repeat(["118", "209", "220"], 100)
    symbols[:25] = "A"  # 118 gets >= 20 S beats for the per-record table
    int8 = np.clip(rng.uniform(size=300) + 0.3 * (symbols != "N"), 0, 1)
    np.savez(path / "oof_predictions.npz", record_ids=records, y=(symbols != "N").astype(np.int32),
             symbols=symbols, float=int8, int8=int8)
    (path / "threshold.json").write_text(json.dumps({"threshold": threshold}))


def test_main_writes_reconciled_tables(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    for label in ("ref", "new"):
        for seed in (42, 43):
            _write_run(tmp_path / label / f"bn_folded_seed{seed}", rng, 0.5)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["compare_oof", "--runs", f"ref={tmp_path / 'ref'}", f"new={tmp_path / 'new'}",
                                      "--seeds", "42", "43", "--out-dir", str(out)])

    compare_oof.main()

    seeds = pd.read_csv(out / "oof_comparison_seeds.csv")
    assert len(seeds) == 2 * 2 * 2 and set(seeds["n_beats"]) == {300}
    z = np.load(tmp_path / "new" / "bn_folded_seed43" / "oof_predictions.npz")
    row = seeds[(seeds["run"] == "new") & (seeds["seed"] == 43) & (seeds["model"] == "int8")].iloc[0]
    expected = compare_oof.oof_metrics(compare_oof.aami_class(z["symbols"]), z["int8"], 0.5)
    assert row[list(expected)].to_dict() == pytest.approx(expected, nan_ok=True)
    records = pd.read_csv(out / "oof_comparison_records.csv", dtype={"record_id": str})
    assert (records["n_S"] >= compare_oof.MIN_RECORD_S_BEATS).all() and "118" in set(records["record_id"])
    assert "Decision: gain over the reference" in (out / "oof_comparison.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("runs", [["noequals"], ["a=x", "a=y"]])
def test_parse_args_rejects_bad_runs(monkeypatch, runs):
    monkeypatch.setattr(sys, "argv", ["compare_oof", "--out-dir", "o", "--runs", *runs])
    with pytest.raises(SystemExit):
        compare_oof.parse_args()


def test_oof_metrics_class_probability_columns_for_nsv_runs():
    from ecg.evaluate import predicted_classes

    rng = np.random.default_rng(1)
    aami = rng.choice(np.array(["N", "S", "V"]), 400, p=[0.8, 0.1, 0.1])
    probs = rng.dirichlet([1, 1, 1], size=400)
    probs[aami == "S", 1] += 0.5
    probs[aami == "V", 2] += 0.5
    probs /= probs.sum(axis=1, keepdims=True)
    scores = 1 - probs[:, 0]

    m = compare_oof.oof_metrics(aami, scores, 0.5, probs)

    for cls, col in (("S", 1), ("V", 2)):
        keep = np.isin(aami, ["N", cls])
        y = aami[keep] == cls
        assert m[f"{cls}_roc_auc_p{cls}"] == roc_auc_score(y, probs[keep, col])
        assert m[f"{cls}_pr_auc_p{cls}"] == average_precision_score(y, probs[keep, col])
    pred = predicted_classes(probs, 0.5)
    assert m["S_as_V"] == int(((aami == "S") & (pred == "V")).sum())
    assert m["V_as_S"] == int(((aami == "V") & (pred == "S")).sum())
    # The decision metrics don't depend on the probabilities beyond the abnormal score.
    binary = compare_oof.oof_metrics(aami, scores, 0.5)
    assert {k: binary[k] for k in compare_oof.DECISION_METRICS} == {k: m[k] for k in compare_oof.DECISION_METRICS}
    assert all(np.isnan(binary[k]) for k in compare_oof.CLASS_PROB_METRICS)


def _nsv_oof(rng, n=600):
    aami = rng.choice(np.array(["N", "S", "V"]), n, p=[0.8, 0.1, 0.1])
    probs = rng.dirichlet([1, 1, 1], size=n)
    probs[aami == "S", 1] += 0.5
    probs[aami == "V", 2] += 0.5
    probs /= probs.sum(axis=1, keepdims=True)
    probs = np.round(probs * 256) / 256  # INT8-like ties
    return aami, probs


@pytest.mark.parametrize("target", compare_oof.TARGET_RECALLS)
def test_operating_point_is_the_highest_threshold_reaching_the_target(target):
    aami, probs = _nsv_oof(np.random.default_rng(4))
    scores = 1 - probs[:, 0]
    abnormal = aami != "N"

    op = compare_oof.operating_point(aami, scores, target)

    flagged = scores > op["threshold"]
    tp, fp = int((flagged & abnormal).sum()), int((flagged & ~abnormal).sum())
    assert (op["TP"], op["FP"]) == (tp, fp)
    assert op["abnormal_recall"] == tp / abnormal.sum() >= target
    assert op["precision"] == tp / (tp + fp)
    assert op["Se_S_flagged"] == flagged[aami == "S"].mean() and op["Se_V_flagged"] == flagged[aami == "V"].mean()
    # S and V flagged counts add up to TP.
    assert op["Se_S_flagged"] * (aami == "S").sum() + op["Se_V_flagged"] * (aami == "V").sum() == pytest.approx(tp)
    # The next abnormal score up misses the target, so no higher threshold reaches it.
    higher = scores[abnormal][scores[abnormal] > op["threshold"]].min()
    assert ((scores > higher) & abnormal).sum() / abnormal.sum() < target
    assert all(np.isnan(op[k]) for k in compare_oof.CALL_RULE_METRICS)


def test_operating_point_call_rule_metrics_for_nsv_runs():
    from ecg.evaluate import predicted_classes

    aami, probs = _nsv_oof(np.random.default_rng(5))
    scores = 1 - probs[:, 0]

    op = compare_oof.operating_point(aami, scores, 0.90, probs)

    pred = predicted_classes(probs, op["threshold"])
    for cls in ("S", "V"):
        hit = ((aami == cls) & (pred == cls)).sum()
        assert op[f"Se_{cls}_call"] == hit / (aami == cls).sum() <= op[f"Se_{cls}_flagged"]
        assert op[f"pP_{cls}_call"] == hit / (pred == cls).sum()
    # The call rule splits exactly the flagged beats into S and V calls.
    assert (pred != "N").sum() == op["TP"] + op["FP"]
    binary = compare_oof.operating_point(aami, scores, 0.90)
    assert {k: binary[k] for k in binary if k not in compare_oof.CALL_RULE_METRICS} == {
        k: op[k] for k in binary if k not in compare_oof.CALL_RULE_METRICS}


def test_int8_float_gap_is_the_per_seed_difference():
    rows = []
    for run, gaps in (("a", [0.01, -0.02, 0.03]), ("b", [0.0, 0.0, 0.0])):
        for seed, gap in zip((42, 43, 44), gaps):
            rows.append({"run": run, "seed": seed, "model": "float", "S_roc_auc": 0.8})
            rows.append({"run": run, "seed": seed, "model": "int8", "S_roc_auc": 0.8 + gap})
    gap = compare_oof.int8_float_gap(pd.DataFrame(rows), metrics=["S_roc_auc"]).set_index("run")

    expected = np.array([0.01, -0.02, 0.03])
    assert gap.loc["a", "int8_minus_float"] == f"{expected.mean():.3f} ± {np.std(expected, ddof=1):.3f}"
    assert gap.loc["a", "max_abs_gap"] == pytest.approx(0.03)
    assert gap.loc["b", "max_abs_gap"] == 0.0


def test_main_reports_class_probability_columns_only_for_nsv_runs(monkeypatch, tmp_path):
    rng = np.random.default_rng(3)
    for seed in (42, 43):
        _write_run(tmp_path / "binary" / f"bn_folded_seed{seed}", rng, 0.5)
        path = tmp_path / "nsv" / f"bn_folded_seed{seed}"
        _write_run(path, rng, 0.5)
        z = dict(np.load(path / "oof_predictions.npz"))
        probs = rng.dirichlet([1, 1, 1], size=len(z["y"]))
        z.update({"float": 1 - probs[:, 0], "int8": 1 - probs[:, 0], "float_probs": probs, "int8_probs": probs,
                  "classes": np.array(["N", "S", "V"])})
        np.savez(path / "oof_predictions.npz", **z)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["compare_oof", "--runs", f"binary={tmp_path / 'binary'}",
                                      f"nsv={tmp_path / 'nsv'}", "--seeds", "42", "43", "--out-dir", str(out)])

    compare_oof.main()

    seeds = pd.read_csv(out / "oof_comparison_seeds.csv")
    nsv, binary = seeds[seeds["run"] == "nsv"], seeds[seeds["run"] == "binary"]
    assert nsv[compare_oof.CLASS_PROB_METRICS].notna().all().all()
    assert binary[compare_oof.CLASS_PROB_METRICS].isna().all().all()
    assert "INT8 minus float" in (out / "oof_comparison.md").read_text(encoding="utf-8")

    ops = pd.read_csv(out / "oof_comparison_operating_points.csv")
    assert len(ops) == 2 * 2 * 2 * len(compare_oof.TARGET_RECALLS)  # runs x seeds x models x target recalls
    assert ops[ops["run"] == "nsv"][compare_oof.CALL_RULE_METRICS].notna().all().all()
    assert ops[ops["run"] == "binary"][compare_oof.CALL_RULE_METRICS].isna().all().all()
    z = np.load(tmp_path / "nsv" / "bn_folded_seed43" / "oof_predictions.npz")
    row = ops[(ops["run"] == "nsv") & (ops["seed"] == 43) & (ops["model"] == "int8")
              & (ops["target_recall"] == 0.90)].iloc[0]
    expected = compare_oof.operating_point(compare_oof.aami_class(z["symbols"]), z["int8"], 0.90, z["int8_probs"])
    assert row[list(expected)].to_dict() == pytest.approx(expected)
    assert "Operating points at fixed abnormal recall" in (out / "oof_comparison.md").read_text(encoding="utf-8")
