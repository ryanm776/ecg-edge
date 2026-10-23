"""Pure helpers of scripts/analyze_splits.py (no data, no network, no models)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.features import extract_beats_from_record

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("analyze_splits", SCRIPTS / "analyze_splits.py")
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


def _synthetic_record():
    fs = 360
    signal = np.sin(2 * np.pi * 1.2 * np.arange(10 * fs) / fs)
    # First and last annotations are skipped; sample 100 is too close to the start for a
    # 250-sample window; "+" is a rhythm annotation, not a beat.
    samples = np.array([50, 100, 400, 700, 1000, 1300, 1600, 1900, 2200, 2500, 3500])
    symbols = ["N", "V", "N", "F", "A", "Q", "+", "V", "S", "N", "N"]
    return signal, fs, samples, symbols


def test_usable_symbol_counts_matches_extraction_and_counts_dropped_types():
    signal, fs, samples, symbols = _synthetic_record()

    counts = analyze.usable_symbol_counts(samples, symbols, len(signal))
    _, _, _, _, extracted = extract_beats_from_record(signal, fs, samples, symbols, "r")

    mapped = {s: c for s, c in counts.items() if s not in ("F", "Q")}
    assert mapped == dict(pd.Series(extracted).value_counts())
    assert counts["F"] == 1 and counts["Q"] == 1
    assert counts["V"] == 1  # the V at sample 100 doesn't fit a window


def test_threshold_rows_reads_both_thresholds(tmp_path):
    (tmp_path / "threshold.json").write_text(json.dumps({"threshold": 0.45}))
    pd.DataFrame([
        {"model": "float", "threshold_from": "int8_val", "threshold": 0.45},
        {"model": "float", "threshold_from": "float_val", "threshold": 0.40},
        {"model": "int8", "threshold_from": "int8_val", "threshold": 0.45},
    ]).to_csv(tmp_path / "comparison_rows.csv", index=False)

    assert analyze.threshold_rows(tmp_path, has_int8=True) == [
        ("float", "int8_val", 0.45), ("float", "float_val", 0.40), ("int8", "int8_val", 0.45)]
    assert len(analyze.threshold_rows(tmp_path, has_int8=False)) == 2


def test_type_recall_and_type_vs_normal_subset():
    y = np.array([0, 0, 1, 1, 1, 1, 1])
    symbols = np.array(["N", "L", "V", "E", "A", "S", "a"])
    probs = np.array([0.1, 0.6, 0.9, 0.2, 0.7, 0.1, 0.1])

    recall = analyze.type_recall(y, symbols, probs, threshold=0.5)

    assert recall["ventricular"] == (2, 1, 0.5)
    assert recall["supraventricular"] == (3, 1, pytest.approx(1 / 3))
    v_mask = analyze.type_vs_normal_subset(y, symbols, "ventricular")
    assert v_mask.tolist() == [True, True, True, True, False, False, False]


def test_counts_only_writes_section_1_and_loads_no_model(tmp_path, monkeypatch):
    from ecg import data

    signal, _, samples, symbols = _synthetic_record()
    loaded = []

    def fake_load_record(rec, data_dir):
        loaded.append(rec)
        return SimpleNamespace(sig_len=len(signal)), SimpleNamespace(sample=samples, symbol=symbols)

    def no_models(*args, **kwargs):
        raise AssertionError("--counts-only must not load predictions or models")

    monkeypatch.setattr(data, "ensure_mitdb_cached", lambda data_dir, records: None)
    monkeypatch.setattr(data, "load_record", fake_load_record)
    monkeypatch.setattr(analyze, "load_or_predict", no_models)
    monkeypatch.setattr(sys, "argv", ["analyze_splits", "--counts-only", "--out-dir", str(tmp_path)])

    analyze.main()

    # Every split record and every DS1/DS2 record is counted, each exactly once.
    all_records = set(analyze.TRAIN_RECORDS + analyze.VAL_RECORDS + analyze.TEST_RECORDS
                      + analyze.DS1_RECORDS + analyze.DS2_RECORDS)
    assert sorted(loaded) == sorted(all_records)

    per_record = analyze.usable_symbol_counts(samples, symbols, len(signal))
    totals = pd.read_csv(tmp_path / "symbol_counts_by_current_split.csv").set_index("current_split")
    for split, records in [("train", analyze.TRAIN_RECORDS), ("val", analyze.VAL_RECORDS),
                           ("test", analyze.TEST_RECORDS)]:
        assert totals.loc[split, "V"] == per_record["V"] * len(records)
        assert totals.loc[split, "normal"] == (per_record["N"]) * len(records)
    ds_totals = pd.read_csv(tmp_path / "symbol_counts_by_dechazal_split.csv").set_index("dechazal_split")
    assert ds_totals.loc["DS1", "V"] == per_record["V"] * len(analyze.DS1_RECORDS)

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## 1. Beat counts by annotation symbol" in summary
    assert "## Overall metrics" not in summary and "## 2." not in summary
    assert not (tmp_path / "predictions.npz").exists()


def test_excluding_record_and_normal_prob_summary():
    record_ids = np.array(["232", "220", "232"])
    assert analyze.excluding_record(record_ids, "232").tolist() == [False, True, False]

    y = np.array([0, 0, 0, 0, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    summary = analyze.normal_prob_summary(y, probs, [("int8_val_0.45", 0.45), ("float_val_0.15", 0.15)])

    assert summary["n_normal"] == 4
    assert summary["frac_above_int8_val_0.45"] == 0.25
    assert summary["frac_above_float_val_0.15"] == 0.75
