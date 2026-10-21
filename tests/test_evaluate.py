import numpy as np
import pytest

from ecg.evaluate import (
    _bind_inputs,
    evaluate_predictions,
    ranking_metrics,
    select_threshold,
    threshold_at_recall,
)


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


def _detail(name, shape):
    return {"name": name, "shape": np.array(shape)}


def test_bind_inputs_by_name_when_names_survive_conversion():
    details = [_detail("serving_default_rr_input:0", [1, 5]), _detail("serving_default_spectrogram_input:0", [1, 33, 12, 1])]
    spec, rr = _bind_inputs(details)
    assert "spectrogram_input" in spec["name"] and "rr_input" in rr["name"]


def test_bind_inputs_falls_back_to_element_count():
    # QAT models' inputs are named after tfmot's quantize_layer, not the Keras inputs.
    details = [_detail("model/quantize_layer_1/FakeQuant", [1, 5]), _detail("model/quantize_layer/FakeQuant", [1, 33, 12, 1])]
    spec, rr = _bind_inputs(details)
    assert list(rr["shape"]) == [1, 5] and list(spec["shape"]) == [1, 33, 12, 1]


def test_bind_inputs_rejects_ambiguous_shapes():
    details = [_detail("a", [1, 33, 12, 1]), _detail("b", [1, 33, 12, 1])]
    with pytest.raises(KeyError):
        _bind_inputs(details)


def test_threshold_at_recall_is_highest_threshold_meeting_target():
    rng = np.random.default_rng(3)
    y_val = np.array([0] * 100 + [1] * 40)
    probs = np.concatenate([rng.uniform(0, 0.6, 100), rng.uniform(0.2, 1.0, 40)])

    t = threshold_at_recall(y_val, probs, target_recall=0.95)

    recall = lambda thr: ((probs > thr) & (y_val == 1)).sum() / 40
    assert recall(t) >= 0.95
    # Nudging the threshold up to the next abnormal probability drops below the target.
    next_pos = np.min(probs[(y_val == 1) & (probs > t)])
    assert recall(next_pos) < 0.95


def test_threshold_at_recall_with_tied_quantized_probs():
    # INT8 outputs are multiples of 1/256, so ties are common; all tied beats count.
    y_val = np.array([1, 1, 1, 1, 0])
    probs = np.array([0.5, 0.5, 0.5, 0.9, 0.1])

    t = threshold_at_recall(y_val, probs, target_recall=1.0)

    assert t < 0.5 and ((probs > t) & (y_val == 1)).sum() == 4


def test_ranking_metrics_perfect_separation():
    y_val, probs = _separated_classes(seed=4)
    assert ranking_metrics(y_val, probs) == {"pr_auc": 1.0, "roc_auc": 1.0}
