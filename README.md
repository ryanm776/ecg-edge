# ECG Heartbeat Classifier

A dual-input (spectrogram + R-R timing) 2D CNN, trained with focal loss [3], that classifies
MIT-BIH [4, 5] heartbeats as Normal or Abnormal, quantized to INT8 using the integer-arithmetic
scheme of [1] and deployed on an ESP32-S3 via TensorFlow Lite Micro [2].

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

## Running the pipeline

```bash
python scripts/train.py
```

On first run this downloads the 44 MIT-BIH records used by the train/val/test splits into
`data/mitdb/` (cached for subsequent runs), trains the model, converts it to INT8, selects a
decision threshold from the **INT8 model's** validation predictions, and evaluates both the
float and INT8 models on the full test set.

Useful flags: `--epochs`, `--batch-size`, `--threshold-strategy {f1,recall_at_precision}`,
`--min-precision` (for `recall_at_precision`), `--seed`, `--variant {baseline,bn_folded,qat}`,
`--calib-samples` (PTQ calibration-set size, default 100), `--normalize-rr`,
`--no-firmware-export`. Run `python scripts/train.py --help` for the full list.

To compare variants (float vs. INT8, each in its own process, never touching
`deployment_model/`):

```bash
python scripts/compare_variants.py                          # baseline, bn_folded, qat
python scripts/compare_variants.py --variants bn_folded --calib-samples 100 500 1000 5000
python scripts/compare_variants.py --variants baseline bn_folded --seeds 42 43 44   # mean ± std over seeds
```

Each run writes to `artifacts/dechazal/variants/<run>/`; all runs are aggregated into
`artifacts/dechazal/variant_comparison.{csv,md}` (per-run rows, plus mean ± std over seeds, also
in `variant_comparison_seeds.csv`). Every variant, baseline included, is trained
from scratch. `artifacts/variants/` and `artifacts/variant_comparison.*` are kept as the
record of the old hand-picked split.

### Outputs

- `artifacts/ecg_multiscale_model.keras`, `artifacts/model_int8.tflite` — the trained model.
- `artifacts/threshold.json` — the chosen decision threshold, selection strategy, and the
  validation metrics at that threshold. This is the current operating threshold; the firmware's
  `deployment_model/threshold.h` is generated from the same value.
- `artifacts/metrics.json` — overall test-set precision/recall/F1/confusion matrix for both the
  float and INT8 models, plus the INT8 model size in KB.
- `artifacts/per_record_metrics.csv` — per-record breakdown (beats, TP/FP/FN/TN,
  precision/recall/F1) for both models.
- `artifacts/val_per_record_metrics.csv` — the same breakdown on validation, for each
  (model, threshold source) row of the comparison table, to see which validation records drive
  the errors and the threshold choice.
- `deployment_model/model_data.h`, `deployment_model/threshold.h` — what the firmware links against.

### Caveat: validation set changed

The original notebook capped normal beats on the validation set to 600/record. This pipeline
removes that cap so validation reflects the full MIT-BIH distribution — but `EarlyStopping`
(`monitor="val_loss"`) now sees a larger, differently-balanced validation set each epoch, so
training may stop at a different epoch and results won't exactly match the original notebook run.

### Data split

The split is the de Chazal inter-patient split [10]. It replaced an earlier hand-picked split
whose validation and test sets disagreed: validation abnormal beats were 94.4% supraventricular (S),
test abnormal beats were 91.9% ventricular, and train had no LBBB beats (see `artifacts/diagnosis.md`).

- **Test = DS2** (22 records), exactly as published, so results are comparable with the literature.
- **Train + validation = DS1** (22 records). Validation is carved out by patient:
  `115, 122, 207, 220, 223`. The other 17 DS1 records are train.
- The four paced records (102, 104, 107, 217) are excluded, as in [10].

Why these validation records:
- **One LBBB record per side:** 109 in train, 207 in validation. DS1 has only these two LBBB patients.
- **S and V beats:** 220 (94 A beats) and 223 (473 V, 73 A). Validation gets its S beats from these
  records rather than from 201, 209 or 118, which stay in train.
- **Normal volume:** 115 and 122 are normal-only, so validation false positives reflect a realistic
  share of normal beats.
- **201 stays in train:** 201 and 202 are the same patient, and 202 is in DS2. Keeping 201 out of
  validation means threshold selection never sees a patient who is also in test.

Usable beats per split (from `python scripts/analyze_splits.py --counts-only`):

| split | beats | normal | ventricular (V, E) | supraventricular (A, a, J, S) | S share of abnormal |
|---|---|---|---|---|---|
| train | 39,645 | 35,870 | 3,105 | 670 | 0.177 |
| val | 10,920 | 9,964 | 683 | 273 | 0.286 |
| test (DS2) | 49,288 | 44,232 | 3,220 | 1,836 | 0.363 |

Known limitations:
- **207 is noisy.** It has SVTA and ventricular-flutter episodes and may dominate validation errors or
  set the threshold. Check the per-record validation table before trusting a threshold.
- **Few S beats in train:** 670, of which 383 are from record 209.
- **Test S beats are mostly one patient's:** record 232 has 1,381 of DS2's 1,836 (75%).
- **201/202 same-patient exception.** The published split puts the same patient in DS1 (201) and DS2
  (202). It is kept for comparability, and documented in `scripts/train.py` and `tests/test_splits.py`.
  `scripts/train.py` also reports test metrics without 202 (`*_excl_202` in `metrics.json` and
  `variant_comparison.*`), alongside the full-DS2 numbers rather than instead of them.

## Testing

```bash
pytest
```

Unit tests cover feature extraction (spectrogram shape, R-R timing features), the C-header
exporters, and threshold selection — all with synthetic data, no network access or MIT-BIH
download required.

## Project layout

See `CLAUDE.md` for the full architecture writeup (data pipeline, model, firmware contract,
known deployment constraints).

## Future work

- Grouped cross-validation over DS1. With only two LBBB patients and 943 S beats in DS1, any fixed
  validation set is an estimate from one or two patients. Planned procedure:
  - patient-grouped 5-fold CV on DS1
  - threshold chosen from the pooled out-of-fold predictions; epoch count from the folds' best epochs
  - one final model trained on all 22 DS1 records with those settings, evaluated once on DS2
  - no ensembling, because of ESP32 compute and memory limits
- Pan-Tompkins R-peak detection on device, replacing ground-truth annotations [7] (see CLAUDE.md deployment constraints)
- Pruning for further model compression [9]
- Tiered cloud data pipeline (edge buffering → Redis → PostgreSQL → object storage) for streaming results, following [8]

## Related work

[6] runs a quantized 1D-CNN arrhythmia classifier on an 8-bit Arduino UNO, but evaluates with beat-wise splits; this project uses patient-independent splits.

## References

1. B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," Proc. IEEE/CVF CVPR, 2018, pp. 2704–2713. arXiv:1712.05877
2. R. David et al., "TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems," Proc. Machine Learning and Systems (MLSys), 2021. arXiv:2010.08678
3. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," Proc. IEEE Int. Conf. on Computer Vision (ICCV), 2017.
4. G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," IEEE Engineering in Medicine and Biology Magazine, vol. 20, no. 3, pp. 45–50, 2001.
5. A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals," Circulation, vol. 101, no. 23, pp. e215–e220, 2000.
6. M. Zambrano-de la Torre et al., "Real-Time Cardiac Arrhythmia Classification Using TinyML on Ultra-Low-Cost Microcontrollers: A Feasibility Study for Resource-Constrained Environments," Bioengineering, vol. 13, no. 5, p. 532, 2026. doi:10.3390/bioengineering13050532
7. J. Pan and W. J. Tompkins, "A Real-Time QRS Detection Algorithm," IEEE Transactions on Biomedical Engineering, vol. BME-32, no. 3, pp. 230–236, 1985.
8. S. Ghosh, R. Bhuvanakantham, P. Sindhujaa, P. B. Harishita, A. Mohan, B. Gulyás, D. Máthé, and P. Padmanabhan, "A Cloud-Aware Scalable Architecture for Distributed Edge-Enabled BCI Biosensor System," Biosensors, vol. 16, no. 3, p. 157, 2026. doi:10.3390/bios16030157
9. M. Zhu and S. Gupta, "To prune, or not to prune: exploring the efficacy of pruning for model compression," arXiv:1710.01878, 2017.
10. P. de Chazal, M. O'Dwyer, and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," IEEE Transactions on Biomedical Engineering, vol. 51, no. 7, pp. 1196–1206, 2004.
