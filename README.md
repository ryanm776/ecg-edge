# ECG heartbeat classifier

A small 2D CNN that classifies MIT-BIH [4, 5] heartbeats on an ESP32-S3. It takes two inputs per beat,
a spectrogram and five R-R timing features, and is trained with focal loss [3]. The model is quantized
to INT8 with the integer-arithmetic scheme of [1] and runs under TensorFlow Lite Micro [2].

> **Work in progress.** The model is still being developed under patient-grouped cross-validation on
> the de Chazal split [10], and no result here is final. The experiment log, with every result so far,
> is [`artifacts/diagnosis.md`](artifacts/diagnosis.md).

What works today:

- Training, INT8 conversion, threshold selection, and evaluation run locally from `scripts/`.
- The INT8 model runs on the ESP32-S3 with synthetic input, as an on-device smoke test.

What is pending:

- On-device evaluation with real ECG input.
- Firmware regenerated from the current model. The committed firmware headers come from an older model
  (see [Committed outputs are from the old split](#committed-outputs-are-from-the-old-split)).

## Set up

```bash
python -m venv .venv
.venv\Scripts\activate       # .venv/bin/activate on macOS and Linux
pip install -r requirements.txt
```

## Run a grouped cross-validation

This is how the model is developed now.

```bash
python scripts/train_cv.py --seeds 42 43 44 --artifacts-dir artifacts/dechazal/stage2/<step>
```

The first run downloads the 44 MIT-BIH records of the split into `data/mitdb/` and caches them.
For each seed, `train_cv.py` does three things:

1. **Trains 5 fold models** (`CV_FOLDS` in `scripts/train.py`). Each one trains on DS1 minus its fold,
   early-stops on the held-out fold, is converted to INT8, and scores the held-out fold.
2. **Selects the threshold** on the pooled out-of-fold INT8 scores from 21 DS1 records. The scores go
   through `evaluate.abnormal_score`, which is P(abnormal) for a binary model and 1 − P(N) for a
   multiclass model.
3. **Trains the final model** on all 22 DS1 records for the median of the folds' best epochs, with no
   early stopping and no ensembling, because the ESP32 runs one model. The final model is evaluated once
   on DS2 at that threshold.

The flags choose the configuration. The defaults give the binary model.

| Flag | Effect |
|---|---|
| `--classes nsv` | Trains a softmax over N, S, and V instead of the binary model. A beat is called N below the threshold, and the likelier of S and V above it. |
| `--loss-weights matched` | The default. Focal alpha 0.25 for binary, and N 0.75, S 0.25, V 0.25 for N/S/V. |
| `--loss-weights none` | Weights every beat 1. |
| `--loss-weights balanced` | N/S/V only. Weights each class by its inverse frequency in the model's own training data, normalized so that the frequency-weighted mean is 1. Fold models then early-stop on the unweighted validation loss. |
| `--rr-lookback N` | The number of R-R intervals averaged into `local_mean_rr`. The default is 10. |
| `--normalize-rr` | Standardizes the R-R features with the mean and standard deviation of each model's own training data. |

One `--artifacts-dir` holds one configuration, and `train_cv.py` refuses to mix configurations or to
overwrite a run unless you pass `--overwrite`. Run `python scripts/train_cv.py --help` for the full list.

**The classes are N, S, and V only.** The dataset drops F and Q beats (DS2 has 388 F beats), for the
binary model too. Results are therefore not directly comparable with papers that report all five AAMI
classes.

### Read the results

- `<artifacts-dir>/cv_comparison.{csv,md}` has each final model's DS2 results, with mean ± std over seeds.
- `<artifacts-dir>/cv/<variant>_seed<S>/` has each run's `metrics.json`, `folds.json`,
  `oof_per_record_metrics.csv`, and saved out-of-fold predictions. This directory is not committed.
- `folds.json` includes a DS2 evaluation of each fold model. It shows the fold-to-fold spread. It is a
  diagnostic and is never used for a choice.
- Each run records its loss weights and its early-stopping criterion, per fold in `folds.json` and for
  the final model in `metrics.json`.

The tables report AAMI sensitivity (`se_V`, `se_S`) and positive predictivity (`pp_V`, `pp_S`) for the
V class (V, E) and the S class (A, a, J, S) on DS2, next to the binary metrics. Positive predictivity is
undefined for a binary model, because an "abnormal" prediction names neither S nor V, and the tables show
`—` there.

To compare two configurations on their out-of-fold predictions, which is how each experiment is decided:

```bash
python scripts/compare_oof.py --runs <ref>=<ref-cv-dir> <new>=<new-cv-dir> --out-dir artifacts/dechazal/stage2/<step>
```

Each experiment's committed tables are in `artifacts/dechazal/stage2/<step>/`.
[`artifacts/diagnosis.md`](artifacts/diagnosis.md) explains each step and its decision.

## Run a single train, validation, and test split

```bash
python scripts/train.py --artifacts-dir <dir> --no-firmware-export
```

`train.py` trains one model on the fixed split below, converts it to INT8, selects a threshold from the
INT8 model's validation predictions, and evaluates the float and INT8 models on DS2.

Without those two flags, `train.py` writes to `artifacts/` and regenerates the firmware headers in
`deployment_model/`. Don't do that yet. See
[Committed outputs are from the old split](#committed-outputs-are-from-the-old-split).

Useful flags:

- `--variant {bn_folded,baseline,qat}`. The default is `bn_folded`. `baseline` is kept only to reproduce
  earlier results.
- `--threshold-strategy {f1,recall_at_precision}`, with `--min-precision` for `recall_at_precision`.
- `--calib-samples`, the size of the calibration set for post-training quantization. The default is 100.
- `--epochs`, `--batch-size`, `--seed`, `--normalize-rr`.

`train.py` writes these files to `--artifacts-dir`:

- `ecg_multiscale_model.keras` and `model_int8.tflite` are the trained model.
- `threshold.json` has the chosen threshold, the selection strategy, and the validation metrics at that
  threshold.
- `metrics.json` has precision, recall, F1, and the confusion matrix on the test set for both models, and
  the INT8 model size in KB.
- `per_record_metrics.csv` has the per-record breakdown on the test set for both models.
- `val_per_record_metrics.csv` has the same breakdown on validation. Use it to see which validation
  records drive the errors and the threshold.

To train several variants and compare float against INT8, each in its own process:

```bash
python scripts/compare_variants.py                          # bn_folded, the default
python scripts/compare_variants.py --variants bn_folded qat
python scripts/compare_variants.py --variants bn_folded --calib-samples 100 500 1000 5000
python scripts/compare_variants.py --variants baseline bn_folded --seeds 42 43 44   # mean ± std over seeds
```

`compare_variants.py` never writes to `deployment_model/`. Each run goes to
`artifacts/dechazal/variants/<run>/`, which is not committed, and all runs are aggregated into
`artifacts/dechazal/variant_comparison.{csv,md}`.

### Committed outputs are from the old split

The files directly under `artifacts/`, and `deployment_model/model_data.h` and `threshold.h`, come from
an earlier hand-picked split, not the de Chazal split. They are not a current operating point.

They come from a seed-42 run of the retired baseline architecture, which loses accuracy when quantized.
At threshold 0.35 on that split's test set, INT8 precision is 0.43 against 0.81 for the float model
(`artifacts/diagnosis.md`, "Float → INT8 gap"). `artifacts/variant_comparison.*` is the record of that
split. These files will be regenerated from the bn_folded model once the final-model procedure is fixed.

## Run the tests

```bash
pytest           # the fast tests, about 30 seconds
pytest -m slow   # the 7 tests that train real models, about 4 minutes
pytest -m ""     # all 200 tests
```

All tests use synthetic data or committed header fixtures. They need no network access and no MIT-BIH
download. Run the slow tests before you commit a change to `ecg/` or `scripts/`.

## Project layout

| Path | Contents |
|---|---|
| `ecg/` | The library: data loading, feature extraction, the models, INT8 export, evaluation, and CV helpers. |
| `scripts/` | Entry points: `train.py`, `train_cv.py`, `compare_variants.py`, `compare_oof.py`, and two diagnosis scripts. |
| `tests/` | The test suite. |
| `artifacts/` | Committed results, and `diagnosis.md`, the experiment log. |
| `deployment_model/` | The Arduino sketch for the ESP32-S3 and the headers it links against. |
| `train_pipeline.ipynb` | The original Colab notebook. It is superseded and kept for reference. |

`CLAUDE.md` describes the architecture in more detail: the data pipeline, the model, and what the
firmware expects from the exported model.

## Data split

The split is the de Chazal inter-patient split [10].

- **Test is DS2** (22 records), exactly as published, so that results are comparable with the literature.
- **Train and validation are DS1** (22 records). `train_cv.py` cross-validates over DS1. `train.py`
  uses a fixed validation set carved out by patient, `115, 122, 207, 220, 223`, and trains on the other
  17 records.
- The four paced records (102, 104, 107, 217) are excluded, as in [10].

This split replaced an earlier hand-picked split whose validation and test sets disagreed. Validation
abnormal beats were 94.4% supraventricular, test abnormal beats were 91.9% ventricular, and train had
no LBBB beats (see `artifacts/diagnosis.md`).

### The fixed validation set

Why these five records:

- **One LBBB record on each side.** 109 is in train and 207 is in validation. DS1 has only these two
  LBBB patients.
- **S and V beats.** 220 has 94 A beats, and 223 has 473 V and 73 A beats. Validation gets its S beats
  from these records, so 201, 209, and 118 stay in train.
- **Normal volume.** 115 and 122 are normal-only, so validation false positives reflect a realistic
  share of normal beats.
- **201 stays in train.** 201 and 202 are the same patient, and 202 is in DS2. Keeping 201 out of
  validation means that threshold selection never sees a patient who is also in test.

Usable beats per split (from `python scripts/analyze_splits.py --counts-only`):

| split | beats | normal | ventricular (V, E) | supraventricular (A, a, J, S) | S share of abnormal |
|---|---|---|---|---|---|
| train | 39,645 | 35,870 | 3,105 | 670 | 0.177 |
| val | 10,920 | 9,964 | 683 | 273 | 0.286 |
| test (DS2) | 49,288 | 44,232 | 3,220 | 1,836 | 0.363 |

Known limitations:

- **207 is noisy.** It has SVTA and ventricular-flutter episodes, and it may dominate validation errors
  or set the threshold. Check the per-record validation table before you trust a threshold.
- **Train has few S beats.** It has 670, and 383 of them are from record 209.
- **Test S beats are mostly from one patient.** Record 232 has 1,381 of DS2's 1,836 (75%).
- **201 and 202 are the same patient.** The published split puts 201 in DS1 and 202 in DS2. The split
  keeps this for comparability. `scripts/train.py` and `tests/test_splits.py` document it, and
  `train.py` also reports test metrics without 202 (`*_excl_202` in `metrics.json` and
  `variant_comparison.*`) next to the full-DS2 numbers.

### The cross-validation folds

A fixed 5-record validation set gives an unstable operating point. Across seeds 42–44, the bn_folded
model's DS2 false-positive count was 3,665 ± 3,630 (`artifacts/diagnosis.md`, "de Chazal split
results"). Grouped cross-validation picks the threshold and the epoch count from all of DS1 instead.

Folds (from `python scripts/analyze_splits.py --counts-only`):

| fold | records | beats | V (V, E) | S (A, a, J, S) | bundle branch block |
|---|---|---|---|---|---|
| 0 | 209, 203, 215 | 9,337 | 609 | 388 | — |
| 1 | 207, 106, 112, 122 | 8,895 | 730 | 108 | LBBB 207 |
| 2 | 118, 119, 116, 205 | 9,317 | 640 | 100 | RBBB 118 |
| 3 | 220, 208, 230, 115, 114 | 10,705 | 1,036 | 108 | — |
| 4 | 223, 124, 109, 108, 101 | 10,351 | 575 | 111 | LBBB 109, RBBB 124 |
| always train | 201 | 1,960 | 198 | 128 | — |

- **201 is never held out.** It trains in every fold and in the final model, so the pooled out-of-fold
  threshold never sees the 201 and 202 patient.
- The two LBBB records (109, 207) are in different folds, and so are the two RBBB records (118, 124).

Caveats:

- **Each fold model early-stops on the fold that it then scores,** so the out-of-fold estimate is
  optimistic. This is standard non-nested cross-validation.
- **The threshold moves from fold models to the final model.** It is chosen on fold models trained on
  about 80% of DS1 and applied to a final model trained on all of it, so the final model's score
  distribution may shift. The fold-model DS2 diagnostic in `folds.json` shows by how much.
- **Fold 0 holds record 209,** which has 41% of DS1's S beats. The fold 0 model therefore trains with
  few S beats: 555 of 943, including 201's 128.

## Differences from the original notebook

`train_pipeline.ipynb` is the superseded Colab version. Its results are not comparable with the results
of this pipeline.

| | Notebook (`train_pipeline.ipynb`) | This pipeline (`train.py`, `train_cv.py`) |
|---|---|---|
| Split | Hand-picked by record: 22 train, 8 validation, 8 test | de Chazal inter-patient split: train and validation (or grouped CV) within DS1, test = DS2; paced records excluded |
| Normal-beat cap | First 600 normal beats per record, on train, validation **and test** | First 600 normal beats per record, **training only** (`--max-normal-per-record-train`, default 600). Validation, CV out-of-fold and test predictions use every beat |
| Threshold | Swept on the test set's predictions (0.50–0.60, then 0.30–0.50) | Selected on validation (`train.py`) or pooled out-of-fold (`train_cv.py`) predictions, never on test |
| Evaluated models | Float Keras model only. The INT8 model is converted and exported but never evaluated | Float and INT8 models. The threshold is chosen on the INT8 model's predictions, since INT8 is what's deployed |

The notebook caps normal beats on its test set, so that test set under-represents normal beats. Its
precision and false-positive counts are not comparable with this pipeline's for that reason too.

## Future work

- **R-peak detection on the device** with Pan-Tompkins [7], in place of the ground-truth annotations.
- **Pruning** for further model compression [9].
- **More training data** from other PhysioNet databases [5]: the MIT-BIH Supraventricular Arrhythmia
  Database (78 recordings, rich in S beats, 128 Hz) and St. Petersburg INCART (75 recordings from 32
  patients, 12-lead, 257 Hz). DS1 has only 22 patients, 2 of them with LBBB, and record 209 holds about
  41% of its S beats.
  - The new data would go into training only. DS2 stays the untouched test set, and results stay
    comparable with papers that use the de Chazal split [10].
  - It needs resampling to 360 Hz, the lead closest to MLII (lead II for INCART), the annotation
    symbols mapped to the AAMI classes, and CV folds grouped by patient, because INCART has several
    recordings per patient.
- **A tiered cloud data pipeline** for streaming results (edge buffering, then Redis, PostgreSQL, and
  object storage), following [8].

## Related work

[6] runs a quantized 1D-CNN arrhythmia classifier on an 8-bit Arduino UNO, but evaluates it with
beat-wise splits. This project uses patient-independent splits.

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
