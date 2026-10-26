"""Pure helpers of scripts/diagnose_s_beats.py (no data, no network, no models)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from ecg.features import compute_rr_timing_features, extract_beats_from_record, rr_quantile_ratio

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("diagnose_s_beats", SCRIPTS / "diagnose_s_beats.py")
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)

FS = 360
# Beats every 300 samples except a premature A at 1500 (pre 200) and a "+" rhythm annotation at 2400.
# First and last annotations are skipped; sample 100 is too close to the start for a 250-sample window.
SAMPLES = np.array([100, 400, 700, 1000, 1300, 1500, 1800, 2100, 2400, 2500, 2800, 3100, 3400])
SYMBOLS = ["N", "N", "N", "N", "N", "A", "N", "N", "+", "N", "V", "N", "N"]
SIGNAL = np.sin(2 * np.pi * 1.2 * np.arange(10 * FS) / FS)


def _table():
    return pd.DataFrame(diag.rr_feature_rows(SAMPLES, SYMBOLS, FS, len(SIGNAL)))


def test_rr_feature_rows_keeps_exactly_the_dataset_beats_with_its_features():
    _, X_rr, y, _, syms = extract_beats_from_record(SIGNAL, FS, SAMPLES, SYMBOLS, "r", rr_from_all_annotations=True)
    table = _table()

    assert table["symbol"].tolist() == syms
    np.testing.assert_array_equal(table[diag.feature_columns("all_ann")[:5]].to_numpy(np.float32),
                                  np.array(X_rr, dtype=np.float32))


def test_rr_feature_rows_all_ann_and_beat_only_match_their_interval_sequences():
    table = _table()
    rr_all = np.diff(SAMPLES) / FS
    beat_idx = np.array([i for i, s in enumerate(SYMBOLS) if s != "+"])
    rr_beat = np.diff(SAMPLES[beat_idx]) / FS

    for row in table.itertuples(index=False):
        i = row.ann_index
        j = int(np.flatnonzero(beat_idx == i)[0])
        r = row._asdict()
        for version, rr, k in (("all_ann", rr_all, i), ("beat_only", rr_beat, j)):
            expected = compute_rr_timing_features(rr, k)
            got = [table.loc[table["ann_index"] == i, f"{version}:{f}"].iloc[0] for f in diag.BASE_FEATURES]
            assert got == expected
            for n in diag.RATIO_LOOKBACKS:
                col = table.loc[table["ann_index"] == i, f"{version}:ratio_q90{n}"].iloc[0]
                assert col == rr_quantile_ratio(rr, k, lookback=n, q=0.9)
        assert r["pre_nonbeat"] == (SYMBOLS[i - 1] == "+")
        assert r["post_nonbeat"] == (SYMBOLS[i + 1] == "+")


def test_rr_feature_rows_differ_only_where_the_rhythm_annotation_reaches():
    table = _table().set_index("ann_index")
    base = diag.BASE_FEATURES
    same = lambda i: [table.loc[i, f"all_ann:{f}"] for f in base] == [table.loc[i, f"beat_only:{f}"] for f in base]

    # Beats before the "+" (index 8) have identical features both ways.
    assert all(same(i) for i in (2, 3, 4, 5, 6))
    # 2100 -> post_rr crosses "+"; 2500 -> pre_rr crosses it.
    assert table.loc[7, "post_nonbeat"] and table.loc[7, "all_ann:post_rr"] == 300 / FS
    assert table.loc[7, "beat_only:post_rr"] == 400 / FS
    assert table.loc[9, "pre_nonbeat"] and table.loc[9, "all_ann:pre_rr"] == 100 / FS
    assert table.loc[9, "beat_only:pre_rr"] == 400 / FS
    # Later beats still differ through the 10-interval lookback.
    assert table.loc[10, "window10_nonbeat"] and not same(10)


def test_capped_normal_mask_keeps_first_normals_per_record_and_every_abnormal():
    record_ids = np.array(["a", "a", "a", "a", "b", "b", "b"])
    y = np.array([0, 1, 0, 0, 0, 0, 1])

    keep = diag.capped_normal_mask(record_ids, y, cap=1)

    assert keep.tolist() == [True, True, False, False, True, False, True]


def test_capped_normal_mask_matches_extract_beats_max_normal():
    _, _, y_capped, _, syms_capped = extract_beats_from_record(SIGNAL, FS, SAMPLES, SYMBOLS, "r", max_normal=3,
                                                               rr_from_all_annotations=True)
    table = _table()

    keep = diag.capped_normal_mask(np.zeros(len(table)), table["symbol"].map({"N": 0, "A": 1, "V": 1}), cap=3)

    assert table["symbol"][keep].tolist() == syms_capped


def test_within_record_permutation_only_moves_beats_within_their_record():
    record_ids = np.array(["a"] * 50 + ["b"] * 30 + ["a"] * 5)

    perm = diag.within_record_permutation(record_ids, np.random.default_rng(0))

    assert sorted(perm.tolist()) == list(range(len(record_ids)))
    np.testing.assert_array_equal(record_ids[perm], record_ids)
    assert (perm != np.arange(len(record_ids))).any()
    np.testing.assert_array_equal(perm, diag.within_record_permutation(record_ids, np.random.default_rng(0)))


def test_class_vs_normal_auc_ignores_the_other_class_and_handles_missing_class():
    aami = np.array(["N", "N", "S", "S", "V", "V"])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0])

    n, roc, pr = diag.class_vs_normal_auc(aami, scores, "S")
    assert (n, roc, pr) == (2, 1.0, 1.0)  # V's low scores would break this if they counted as N

    n, roc, pr = diag.class_vs_normal_auc(np.array(["N", "N"]), np.array([0.1, 0.2]), "S")
    assert n == 0 and np.isnan(roc) and np.isnan(pr)


def test_class_score_summary_counts_reconcile_with_the_input():
    rng = np.random.default_rng(0)
    aami = rng.choice(np.array(["N", "S", "V"]), 300)
    scores = rng.uniform(size=300)

    rows = diag.class_score_summary(aami, scores, 0.5)

    assert sum(r["n"] for r in rows) == 300
    for r in rows:
        s = scores[aami == r["class"]]
        assert r["frac_above_t"] == (s > 0.5).mean() and r["p50"] == np.median(s)


def test_stage1_beat_only_features_are_what_the_fixed_pipeline_extracts():
    _, X_rr, _, _, syms = extract_beats_from_record(SIGNAL, FS, SAMPLES, SYMBOLS, "r")
    table = _table()
    # The fixed pipeline keeps exactly the rows whose beat-only features are defined.
    defined = table.dropna(subset=diag.feature_columns("beat_only"))

    assert defined["symbol"].tolist() == syms
    np.testing.assert_array_equal(defined[diag.feature_columns("beat_only")[:5]].to_numpy(np.float32),
                                  np.array(X_rr, dtype=np.float32))
