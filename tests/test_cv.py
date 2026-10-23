import numpy as np
import pytest

from ecg.cv import final_epochs, fold_masks, pool_oof
from ecg.evaluate import abnormal_score, select_threshold

RECORD_IDS = np.array(["a", "a", "b", "c", "c", "c", "201", "d", "201"], dtype=object)
FOLDS = [["a", "c"], ["b"], ["d"]]


def test_fold_masks_hold_out_each_fold_and_train_on_the_rest():
    masks = fold_masks(RECORD_IDS, FOLDS)
    held_out = np.array([m[1] for m in masks])
    # Every folded record's beats are held out exactly once; 201 (in no fold) never is.
    assert (held_out.sum(axis=0) == (RECORD_IDS != "201")).all()
    for (train_mask, held), fold in zip(masks, FOLDS):
        assert (held == np.isin(RECORD_IDS, fold)).all()
        assert (train_mask == ~held).all()
        assert train_mask[RECORD_IDS == "201"].all()


def test_pool_oof_takes_each_beat_from_the_fold_that_held_it_out():
    masks = fold_masks(RECORD_IDS, FOLDS)
    held = [m[1] for m in masks]
    scores = [np.arange(int(h.sum())) + 10.0 * k for k, h in enumerate(held)]

    oof, covered = pool_oof(scores, held)

    assert (covered == (RECORD_IDS != "201")).all()
    assert np.isnan(oof[~covered]).all()
    for k, h in enumerate(held):
        assert (oof[h] == scores[k]).all()


def test_pool_oof_rejects_overlap_and_length_mismatch():
    a = np.array([True, True, False])
    b = np.array([False, True, True])
    with pytest.raises(ValueError, match="more than one fold"):
        pool_oof([np.zeros(2), np.zeros(2)], [a, b])
    with pytest.raises(ValueError, match="held-out beats"):
        pool_oof([np.zeros(3)], [a])


@pytest.mark.parametrize("best, expected", [
    ([10, 30, 20], 20),
    ([10, 20, 30, 40], 25),
    ([10, 20, 21, 40], 21),  # median 20.5 rounds up
    ([7], 7),
])
def test_final_epochs_is_the_rounded_median(best, expected):
    assert final_epochs(best) == expected


def test_abnormal_score_binary_passthrough():
    p = np.array([0.1, 0.7, 0.4])
    assert (abnormal_score(p) == p).all()
    assert (abnormal_score(p[:, None]) == p).all()


def test_abnormal_score_multiclass_is_one_minus_normal():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    assert np.allclose(abnormal_score(p), 1.0 - p[:, 0])
    assert np.allclose(abnormal_score(p, normal_index=2), 1.0 - p[:, 2])


def test_threshold_on_two_class_softmax_matches_binary():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    p_abnormal = np.clip(np.where(y == 1, rng.normal(0.7, 0.2, 80), rng.normal(0.3, 0.2, 80)), 0, 1)
    softmax = np.stack([1.0 - p_abnormal, p_abnormal], axis=1)

    t_binary, sweep_binary = select_threshold(y, abnormal_score(p_abnormal))
    t_multi, sweep_multi = select_threshold(y, abnormal_score(softmax))

    assert t_binary == t_multi
    assert np.allclose(sweep_binary["f1"], sweep_multi["f1"])


def test_abnormal_score_rejects_other_shapes():
    with pytest.raises(ValueError):
        abnormal_score(np.zeros((2, 2, 2)))
