"""The train/val/test record lists in scripts/train.py follow the de Chazal DS1/DS2 split."""
import importlib.util
from itertools import combinations
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = _load("train")
SPLITS = {"train": train.TRAIN_RECORDS, "val": train.VAL_RECORDS, "test": train.TEST_RECORDS}


@pytest.mark.parametrize("name", SPLITS)
def test_no_duplicates_within_a_split(name):
    assert len(SPLITS[name]) == len(set(SPLITS[name]))


@pytest.mark.parametrize("a,b", list(combinations(SPLITS, 2)))
def test_no_record_in_two_splits(a, b):
    assert not set(SPLITS[a]) & set(SPLITS[b])


def test_same_patient_exception_is_train_vs_test_only():
    # 201 and 202 are one patient; the published split keeps 201 in DS1 and 202 in DS2.
    # 201 must stay in train so threshold selection on val never sees this patient.
    assert train.SAME_PATIENT_EXCEPTIONS == [("201", "202")]
    assert "201" in train.TRAIN_RECORDS and "201" not in train.VAL_RECORDS
    assert "202" in train.TEST_RECORDS
    assert train.OVERLAP_RECORD == "202"


def test_paced_records_excluded():
    assert set(train.PACED_RECORDS) == {"102", "104", "107", "217"}
    for records in SPLITS.values():
        assert not set(records) & set(train.PACED_RECORDS)


def test_ds1_is_train_plus_val_and_ds2_is_test():
    assert len(train.DS1_RECORDS) == len(set(train.DS1_RECORDS)) == 22
    assert len(train.DS2_RECORDS) == len(set(train.DS2_RECORDS)) == 22
    assert set(train.TRAIN_RECORDS) | set(train.VAL_RECORDS) == set(train.DS1_RECORDS)
    assert set(train.TEST_RECORDS) == set(train.DS2_RECORDS)
    # 44 records + 4 paced = all 48 MIT-BIH records.
    assert len(set(train.DS1_RECORDS) | set(train.DS2_RECORDS) | set(train.PACED_RECORDS)) == 48


def test_cv_folds_partition_ds1_without_201():
    folded = [r for fold in train.CV_FOLDS for r in fold]
    assert len(train.CV_FOLDS) == 5
    assert len(folded) == len(set(folded))
    assert set(folded) | set(train.CV_ALWAYS_TRAIN) == set(train.DS1_RECORDS)
    assert not set(folded) & set(train.CV_ALWAYS_TRAIN)
    # 201 is never held out, so threshold selection never sees the 201/202 patient.
    assert train.CV_ALWAYS_TRAIN == ["201"]


@pytest.mark.parametrize("pair", [("109", "207"), ("118", "124")], ids=["LBBB", "RBBB"])
def test_cv_bundle_branch_block_records_in_different_folds(pair):
    fold_of = {r: i for i, fold in enumerate(train.CV_FOLDS) for r in fold}
    assert fold_of[pair[0]] != fold_of[pair[1]]
