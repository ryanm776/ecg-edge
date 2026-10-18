import numpy as np

from ecg.evaluate import evaluate_predictions, select_threshold


def _separated_classes(seed, n_per_class=30):
    rng = np.random.default_rng(seed)
    y_val = np.array([0] * n_per_class + [1] * n_per_class)
    probs_val = np.concatenate([
        rng.uniform(0.0, 0.3, size=n_per_class),
        rng.uniform(0.7, 1.0, size=n_per_class),
    ])
    return y_val, probs_val


def test_select_threshold_f1_strategy_lands_in_the_gap():
    y_val, probs_val = _separated_classes(seed=0)

    threshold, sweep_df = select_threshold(y_val, probs_val, strategy="f1")

    assert 0.3 <= threshold < 0.7
    assert set(sweep_df.columns) == {"threshold", "precision", "recall", "f1"}
    best_f1 = sweep_df.loc[sweep_df["threshold"] == threshold, "f1"].iloc[0]
    assert best_f1 == sweep_df["f1"].max()


def test_select_threshold_recall_at_precision_meets_floor():
    y_val, probs_val = _separated_classes(seed=1)
    min_precision = 0.90

    threshold, sweep_df = select_threshold(
        y_val, probs_val, strategy="recall_at_precision", min_precision=min_precision
    )

    row = sweep_df.loc[sweep_df["threshold"] == threshold].iloc[0]
    assert row["precision"] >= min_precision

    eligible = sweep_df[sweep_df["precision"] >= min_precision]
    assert threshold == eligible["threshold"].min()


def test_select_threshold_rejects_unknown_strategy():
    y_val, probs_val = _separated_classes(seed=2)

    try:
        select_threshold(y_val, probs_val, strategy="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_evaluate_predictions_per_record_handles_zero_abnormal():
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.9, 0.8])
    record_ids = np.array(["rec_all_normal", "rec_all_normal", "rec_mixed", "rec_mixed"])

    overall, per_record = evaluate_predictions(y_true, probs, record_ids, threshold=0.5)

    assert overall["precision"] == 1.0
    assert overall["recall"] == 1.0

    normal_row = per_record.loc[per_record["record_id"] == "rec_all_normal"].iloc[0]
    assert normal_row["n_abnormal"] == 0
    assert normal_row["precision"] == 0.0
    assert normal_row["recall"] == 0.0
    assert normal_row["accuracy"] == 1.0

    mixed_row = per_record.loc[per_record["record_id"] == "rec_mixed"].iloc[0]
    assert mixed_row["n_abnormal"] == 2
    assert mixed_row["precision"] == 1.0
    assert mixed_row["recall"] == 1.0
