"""Lead selection in ecg/data.py (headers from tests/fixtures, no signals, no network)."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import wfdb

from ecg import data

ROOT = Path(__file__).resolve().parent
HEADERS = ROOT / "fixtures" / "mitdb_headers"
_spec = importlib.util.spec_from_file_location("train", ROOT.parent / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

FS = 360
SAMPLES = np.arange(300, 3300, 300)
SYMBOLS = ["N", "N", "V", "N", "N", "A", "N", "N", "N", "N"]


@pytest.mark.parametrize("record_id", train.DS1_RECORDS + train.DS2_RECORDS)
def test_every_split_record_reads_mlii(record_id):
    names = wfdb.rdheader(str(HEADERS / record_id)).sig_name

    assert names[data.lead_index(names)] == "MLII"


def test_record_114_is_the_only_split_record_without_mlii_in_channel_0():
    swapped = [r for r in train.DS1_RECORDS + train.DS2_RECORDS
               if wfdb.rdheader(str(HEADERS / r)).sig_name[0] != "MLII"]

    assert swapped == ["114"]
    assert data.lead_index(wfdb.rdheader(str(HEADERS / "114")).sig_name) == 1


def test_lead_index_raises_when_the_lead_is_missing():
    with pytest.raises(ValueError):
        data.lead_index(["V1", "V5"])


def _fake_record(sig_names, rng):
    t = np.arange(4000) / FS
    mlii = np.sin(2 * np.pi * 1.2 * t) + 0.1 * rng.standard_normal(len(t))
    other = np.cos(2 * np.pi * 3.0 * t)
    columns = {"MLII": mlii, "V5": other}
    return SimpleNamespace(fs=FS, sig_name=sig_names, p_signal=np.column_stack([columns[n] for n in sig_names]))


def _build(monkeypatch, sig_names, **kwargs):
    record = _fake_record(sig_names, np.random.default_rng(0))
    annotation = SimpleNamespace(sample=SAMPLES, symbol=SYMBOLS)
    monkeypatch.setattr(data, "ensure_mitdb_cached", lambda data_dir, records: None)
    monkeypatch.setattr(data, "load_record", lambda rec, data_dir: (record, annotation))
    return data.build_dataset(["r"], "unused", **kwargs)


def test_build_dataset_reads_mlii_whatever_its_channel(monkeypatch):
    mlii_first = _build(monkeypatch, ["MLII", "V5"])
    mlii_second = _build(monkeypatch, ["V5", "MLII"])

    for a, b in zip(mlii_first, mlii_second):
        np.testing.assert_array_equal(a, b)


def test_build_dataset_lead_none_reads_channel_0(monkeypatch):
    legacy = _build(monkeypatch, ["V5", "MLII"], lead=None)
    v5 = _build(monkeypatch, ["MLII", "V5"], lead="V5")

    for a, b in zip(legacy, v5):
        np.testing.assert_array_equal(a, b)
