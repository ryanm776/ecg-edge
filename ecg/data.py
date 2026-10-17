"""MIT-BIH data acquisition/caching and dataset assembly."""
from __future__ import annotations

import os

import numpy as np
import wfdb

from . import features

DB_NAME = "mitdb"


def ensure_mitdb_cached(data_dir, record_ids):
    """Download only the given records into <data_dir>/mitdb if not already present."""
    db_dir = os.path.join(data_dir, DB_NAME)
    os.makedirs(db_dir, exist_ok=True)

    missing = [
        rec for rec in record_ids
        if not os.path.exists(os.path.join(db_dir, f"{rec}.dat"))
    ]
    if missing:
        wfdb.dl_database(DB_NAME, dl_dir=db_dir, records=missing)

    return db_dir


def load_record(record_id, data_dir):
    db_dir = os.path.join(data_dir, DB_NAME)
    record_path = os.path.join(db_dir, record_id)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, extension="atr")
    return record, annotation


def build_dataset(record_list, data_dir, max_normal_per_record=None, augment=False, seed=None):
    """Load, filter, and feature-extract every usable beat across `record_list`.

    Returns (X_spec, X_rr, y, record_ids) as arrays, all aligned beat-for-beat.
    `record_ids` is new relative to the original notebook -- needed for per-record metrics.
    """
    ensure_mitdb_cached(data_dir, record_list)
    rng = np.random.default_rng(seed)

    spec_list, rr_list, y_list, record_id_list = [], [], [], []

    for rec in record_list:
        try:
            record, annotation = load_record(rec, data_dir)
        except ValueError:
            continue

        fs = record.fs
        filtered = features.bandpass_filter(record.p_signal[:, 0], fs)

        specs, rrs, ys, rec_ids = features.extract_beats_from_record(
            filtered,
            fs,
            annotation.sample,
            annotation.symbol,
            rec,
            max_normal=max_normal_per_record,
            augment=augment,
            rng=rng,
        )
        spec_list.extend(specs)
        rr_list.extend(rrs)
        y_list.extend(ys)
        record_id_list.extend(rec_ids)

    X_spec = np.array(spec_list, dtype=np.float32)[..., np.newaxis]
    X_rr = np.array(rr_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    record_ids = np.array(record_id_list, dtype=object)

    return X_spec, X_rr, y, record_ids
