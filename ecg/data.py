"""MIT-BIH data acquisition/caching and dataset assembly."""
from __future__ import annotations

import os

import numpy as np
import wfdb

from . import features

DB_NAME = "mitdb"
# The lead the model is trained on. MIT-BIH usually stores it in channel 0, but not always (114 has V5 there).
LEAD = "MLII"


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


def lead_index(sig_names, lead=LEAD):
    """Channel index of `lead` in a record's signal names; ValueError if the record doesn't have it."""
    names = list(sig_names)
    if lead not in names:
        raise ValueError(f"lead {lead!r} not in {names}")
    return names.index(lead)


def build_dataset(record_list, data_dir, max_normal_per_record=None, augment=False, seed=None,
                  return_symbols=False, rr_from_all_annotations=False, lead=LEAD):
    """Load, filter, and feature-extract every usable beat across `record_list`.

    Returns (X_spec, X_rr, y, record_ids) as arrays, all aligned beat-for-beat.
    `record_ids` is new relative to the original notebook -- needed for per-record metrics.
    `return_symbols=True` appends each beat's annotation symbol as a fifth array.
    `rr_from_all_annotations=True` reproduces the pre-fix R-R features (see extract_beats_from_record).
    `lead` picks the channel by signal name; `lead=None` reproduces the pre-fix behavior of always reading
    channel 0, which for record 114 is V5 instead of MLII.
    """
    ensure_mitdb_cached(data_dir, record_list)
    rng = np.random.default_rng(seed)

    spec_list, rr_list, y_list, record_id_list, symbol_list = [], [], [], [], []

    for rec in record_list:
        try:
            record, annotation = load_record(rec, data_dir)
        except ValueError:
            continue

        fs = record.fs
        channel = 0 if lead is None else lead_index(record.sig_name, lead)
        filtered = features.bandpass_filter(record.p_signal[:, channel], fs)

        specs, rrs, ys, rec_ids, syms = features.extract_beats_from_record(
            filtered,
            fs,
            annotation.sample,
            annotation.symbol,
            rec,
            max_normal=max_normal_per_record,
            augment=augment,
            rng=rng,
            rr_from_all_annotations=rr_from_all_annotations,
        )
        spec_list.extend(specs)
        rr_list.extend(rrs)
        y_list.extend(ys)
        record_id_list.extend(rec_ids)
        symbol_list.extend(syms)

    X_spec = np.array(spec_list, dtype=np.float32)[..., np.newaxis]
    X_rr = np.array(rr_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    record_ids = np.array(record_id_list, dtype=object)

    if return_symbols:
        return X_spec, X_rr, y, record_ids, np.array(symbol_list, dtype=object)
    return X_spec, X_rr, y, record_ids
