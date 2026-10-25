"""Patient-grouped cross-validation helpers (numpy only; the training loop is scripts/train_cv.py)."""
from __future__ import annotations

import numpy as np


def fold_masks(record_ids, folds):
    """(train_mask, held_out_mask) per fold over a beat-aligned `record_ids` array.

    A fold's held-out beats are its records' beats; everything else in `record_ids` trains, so records
    that are in no fold (e.g. CV_ALWAYS_TRAIN) train in every fold.
    """
    record_ids = np.asarray(record_ids)
    masks = []
    for fold in folds:
        held_out = np.isin(record_ids, fold)
        masks.append((~held_out, held_out))
    return masks


def pool_oof(scores_by_fold, held_out_masks):
    """Pool each fold's held-out scores into one array aligned with the masks' beats.

    `scores_by_fold[k]` holds fold k's scores for its held-out beats, in beat order. Returns (oof, covered):
    `oof` is NaN where no fold held the beat out, and `covered` marks the beats that have an OOF score.
    """
    n = len(held_out_masks[0])
    oof = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    for scores, mask in zip(scores_by_fold, held_out_masks):
        scores = np.asarray(scores, dtype=float)
        if len(scores) != int(mask.sum()):
            raise ValueError(f"Fold has {int(mask.sum())} held-out beats but {len(scores)} scores")
        if (covered & mask).any():
            raise ValueError("A beat is held out by more than one fold")
        oof[mask] = scores
        covered |= mask
    return oof, covered


def final_epochs(best_epochs):
    """Epoch count for the final all-DS1 model: the median of the folds' best epochs, rounded half up."""
    return int(np.floor(np.median(best_epochs) + 0.5))
