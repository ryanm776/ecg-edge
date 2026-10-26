# Float → INT8 gap: diagnosis

> **Old split.** Every result in this file except the final section, "de Chazal split results", and every committed file directly under `artifacts/` (model, `metrics.json`, `variant_comparison.*`, …), uses the old hand-picked train/val/test split. Results on the de Chazal split go in `artifacts/dechazal/` (see README.md, "Data split").

Baseline model (`ecg_multiscale_model.keras` / `model_int8.tflite`, seed 42) on test, threshold 0.35:

| Model | Precision | Recall | FP | FP rec 214 | FP excl. 214 |
|---|---|---|---|---|---|
| float | 0.810 | 0.959 | 282 | 79 | 203 |
| INT8 | 0.433 | 0.917 | 1,502 | 1,146 | 356 |

**Hypothesis tested:** INT8 input quantization (calibrated on 100 samples) clips the unnormalized R-R features.
**Verdict:** rejected. The gap comes from inside the network: Conv → ReLU6 → BatchNorm ordering leaves BN unfolded and tensors with ~4–5 INT8 levels.

All numbers below were reproduced with seed 42, so the PTQ calibration indices are identical to the ones `scripts/train.py` used.

## 1. Float32 TFLite matches Keras

Converting the same model to an unquantized float32 `.tflite` (17,466 test beats):

- max |Δprob| = 6.17e-6
- mean |Δprob| = 4.13e-7
- 0 decision flips at threshold 0.35

So the conversion pipeline isn't the cause.

## 2. Input quantization parameters

| Tensor | Scale | Zero point | Representable range |
|---|---|---|---|
| `rr_input` | 0.009004 | −128 | [0, 2.296] |
| `spectrogram_input` | 0.003922 (1/255) | −128 | [0, 1.000] |
| output | 0.003906 (1/256) | −128 | [0, 1.000] |

- The `rr_input` upper bound (2.296) is the calibration sample's max `pre_post_ratio`.
- Spectrograms are min-max normalized to [0, 1], so that input never clips.

## 3. R-R values outside the representable range

| Set | Beats | Values clipped | Beats with any clip | Per feature (pre_rr / post_rr / local_mean / rr_ratio / pre_post_ratio) |
|---|---|---|---|---|
| calibration (100) | 100 | 0.000% | 0.000% | 0 / 0 / 0 / 0 / 0 |
| train | 38,582 | 0.203% | 1.006% | 0.01 / 0.00 / 0.00 / 0.02 / 0.99 % |
| val | 15,996 | 0.620% | 2.313% | 0.56 / 0.57 / 0.00 / 0.49 / 1.48 % |
| test | 17,466 | 0.134% | 0.670% | 0.00 / 0.00 / 0.00 / 0.01 / 0.66 % |
| record 214 | 2,257 | 0.133% | 0.665% | 0.00 / 0.00 / 0.00 / 0.00 / 0.66 % |
| record 232 | 1,779 | 5.329% | 19.562% | 5.06 / 5.12 / 0.00 / 4.44 / 12.03 % |

Record 232's max values are pre/post_rr 5.87 s, rr_ratio 4.64 and pre_post_ratio 12.4. Train's max `pre_post_ratio` is 8.76.

## 4. Fake-quantizing the inputs doesn't reproduce the gap

The float Keras model is run with inputs passed through quantize→dequantize using the INT8 model's own input scale/zero point, which reproduces the clipping and rounding exactly. Test set, threshold 0.35:

| Variant | FP | FP rec 214 | FP excl. 214 | TP | Precision | Recall |
|---|---|---|---|---|---|---|
| float | 282 | 79 | 203 | 1,202 | 0.810 | 0.959 |
| float, R-R fake-quant | 276 | 77 | 199 | 1,202 | 0.813 | 0.959 |
| float, spectrogram fake-quant | 276 | 75 | 201 | 1,201 | 0.813 | 0.958 |
| float, both inputs fake-quant | 273 | 75 | 198 | 1,200 | 0.815 | 0.958 |
| **INT8 `.tflite`** | **1,502** | **1,146** | **356** | **1,149** | **0.433** | **0.917** |

Validation set:

| Model | Best F1 (threshold) | F1 @ 0.35 | Record 232 F1 @ 0.35 |
|---|---|---|---|
| float | 0.499 (0.15) | 0.425 | 0.665 |
| float, R-R fake-quant | 0.499 (0.15) | 0.425 | 0.663 |
| INT8 | 0.488 (0.35) | 0.488 | 0.917 |

Clipping record 232's R-R values (19.6% of its beats) changes its F1 by 0.002. The low validation F1 is a float-model property, not a quantization one.

Note: the threshold (0.35) is picked from INT8 val predictions, but float's own best val threshold is 0.15. So the baseline compares float at a threshold that wasn't chosen for it.

Per-record test FPs (float → INT8):

| Record | 100 | 105 | 113 | 121 | 123 | 202 | 214 | 233 |
|---|---|---|---|---|---|---|---|---|
| float | 0 | 124 | 3 | 8 | 3 | 30 | 79 | 35 |
| INT8 | 42 | 118 | 56 | 10 | 5 | 78 | 1,146 | 47 |

## 5. Root cause: BatchNorm after ReLU6 can't be folded

`ecg/model.py` uses Conv2D(relu6) → BatchNormalization in both conv stages. The parallel branches are concatenated first, then BN. BN after an activation can't be folded into the conv weights, so the converter emits it as standalone INT8 `MUL` + `ADD` ops.

Two things then combine:

- **Tiny pre-BN activations.** L2 regularization (`kernel_regularizer="l2"`, factor 0.01) has shrunk the conv kernels (conv2d_2 kernel |w| mean 6.3e-4), and BN `moving_variance` is ~1e-5. Each BN therefore applies a per-channel gain of roughly 100–300×. The INT8 BN multiplier tensors span [0, 45.96] and [0, 56.08].
- **Coarse quantization of those activations.** The pre-BN tensors span only a few INT8 steps:

| Tensor (pre-BN) | Float range (record 214 normals) | INT8 scale | INT8 levels covering the float range |
|---|---|---|---|
| `concatenate` (conv branches) | [0, 0.088] | 0.01773 | **~5.0** |
| `conv2d_2` | [0, 0.177] | 0.04746 | **~3.7** |

BN then multiplies that rounding error by its gain.

Accumulated INT8-vs-float error, measured end to end on 300 record-214 normal beats (dequantized INT8 intermediate tensor vs. Keras layer output):

| Tensor | Float range | RMSE | RMSE / float std |
|---|---|---|---|
| `concatenate` | [0, 0.088] | 0.0031 | 0.22 |
| `batch_normalization` | [−1.79, 3.19] | 0.120 | 0.59 |
| `max_pooling2d` | [−0.89, 3.19] | 0.123 | 0.56 |
| `conv2d_2` | [0, 0.177] | 0.0021 | 1.47 |
| `batch_normalization_1` | [0, 9.91] | 0.108 | **1.44** |
| `dense` (R-R branch) | [0, 6.00] | 0.033 | 0.02 |
| `dense_1` | [0, 0.425] | 0.067 | **1.12** |
| `output` (prob) | [0.001, 0.907] | 0.400 | **2.60** |

By `batch_normalization_1`, the error exceeds the signal's own spread, so the spectrogram branch is effectively noise. The R-R branch is quantized cleanly (0.02× std).

The per-op TFLite `QuantizationDebugger` (local error only, not accumulated) shows every op at ≤ 0.34 LSB RMSE except the R-R `dense` (0.98 LSB). The problem isn't any single op being badly calibrated. It's that a ~4-level tensor gets amplified by an unfolded BN.

## 6. Calibration size is a secondary effect

Re-running PTQ with larger shuffled calibration sets (test FP @ 0.35):

| Calibration samples | FP | FP rec 214 | TP | Precision |
|---|---|---|---|---|
| 100 (baseline) | 1,502 | 1,146 | 1,149 | 0.433 |
| 1,000 | 1,218 | 936 | 1,135 | 0.482 |
| 5,000 | 1,271 | 975 | 1,136 | 0.472 |

## 7. QAT compatibility (tfmot)

- Versions: TF 2.21.0, Keras 3.15.1, tf_keras 2.21.0, tensorflow-model-optimization 0.8.1. tfmot imports fine.
- Importing tfmot makes `tf.keras` resolve to `tf_keras` (Keras 2) for the whole process. QAT therefore has to run in its own process, separate from the Keras 3 training path.
- `tfmot.quantization.keras.quantize_model` on the current architecture fails both with and without `TF_USE_LEGACY_KERAS=1`: `Layer batch_normalization ... is not supported`. Neither BN placement (after Concatenate, or after Conv+ReLU6) is supported by the default 8-bit registry.

## 8. Unrelated firmware bug found along the way

The INT8 flatbuffer's subgraph inputs are `[rr_input (tensor 0), spectrogram_input (tensor 1)]`. `deployment_model.ino` bound `input(0)` as the spectrogram, so on device the two inputs were swapped, and the `min(bytes, 5)` bounds check hid it. This is fixed in commit f94682d: inputs are now bound by element count.

## Next step

Compare, on the same seed:

- (a) this baseline
- (d) PTQ with Conv → BN → ReLU6 reordering, so BN folds into the conv
- (e) QAT on the reordered architecture

Then add R-R normalization to the better of d/e as a separate row. Results will go in `variant_comparison.md`.

## QAT follow-up (parked)

- **QAT isn't needed for the current architecture.** BN folding alone closed the float→INT8 gap. In Run 1, bn_folded PTQ has float precision 0.853 vs INT8 0.852, FP 177 vs 179, and record 214 FP 6 vs 6. Its narrowest tensor spans 230 INT8 levels, up from the baseline's 5.
- **The existing QAT run isn't a fair test of QAT.** Its float model, trained from scratch under tf_keras with the same architecture, data and seed as bn_folded, underperformed before any QAT fine-tuning:
  - best val F1: 0.378 (threshold 0.15) vs bn_folded's 0.455 (threshold 0.40)
  - its INT8 model's val F1: 0.359, vs bn_folded INT8's 0.451

  QAT fine-tuning doesn't modify the float model (checked: no shared variables), so the deficit comes from the tf_keras float training itself. The cause wasn't investigated.
- **A fair future test** should initialize QAT from bn_folded's trained float weights rather than retraining under tf_keras. For example, build the same model with `keras_mod=tf_keras` and copy the weights over with `get_weights`/`set_weights`, then run `quantize_aware` and fine-tune.
- `ecg/qat.py` and `train.py --variant qat` stay in place for that test.

## Validation/test mismatch

Generated by `scripts/analyze_splits.py`; full tables in `artifacts/split_analysis/summary.md`, which is gitignored and can be regenerated. Models: baseline float, bn_folded float, and bn_folded INT8. Baseline INT8 is left out because the quantization bug dominates it. Test totals reproduce `variant_comparison.md`: bn_folded INT8 at 0.45 has 179 FP, and baseline float at 0.35 has 282 FP.

**Verdict:** the hypothesis holds for the abnormal beats, but it's only half the story.
- Validation's abnormal beats are dominated by one patient's supraventricular beats, while test's are ventricular. So the two splits do measure different tasks.
- Validation's errors split between two records:
  - **record 109 drives false positives.** Its normal beats are LBBB, a morphology the training set never contains.
  - **record 232 drives false negatives.** Its A beats are mostly missed.
- Separately, both val and test results are effectively decided by 2–3 patients each.

### 1. Abnormal composition: val is supraventricular, train and test are ventricular

Usable beats, using the same window rule as the dataset. Train counts are uncapped and unaugmented.

| Split | Normal | of which L (LBBB) | of which R (RBBB) | Ventricular (V, E) | Supraventricular (A, a, J, S) | S share of abnormal | F (dropped) | Q (dropped) |
|---|---|---|---|---|---|---|---|---|
| train (22 rec.) | 47,219 | **0** | 3,989 | 5,084 | 1,007 | 16.5% | 782 | 8 |
| val (8 rec.) | 14,398 | 4,612 | 3,180 | 89 | 1,509 | **94.4%** | 7 | 0 |
| test (8 rec.) | 16,213 | 2,001 | 0 | 1,151 | 102 | **8.1%** | 13 | 7 |

- **Val:** record 232 contributes 1,381 of the 1,598 abnormal beats (86%), all of them `A`.
- **Test:** records 233 (837) and 214 (256) contribute 1,093 of the 1,253 abnormal beats (87%), almost all `V`.
- **Train has no LBBB beats.** The only L records in the current split are 109 and 111 (val) and 214 (test).

### 2. Per-record validation: record 109 drives false positives, record 232 drives false negatives

| Model @ threshold | Val FP | From 109 (LBBB) | 2nd FP source | Val FN | From 232 | Record 232 A-beat recall |
|---|---|---|---|---|---|---|
| baseline float @ 0.35 | 1,528 | 1,281 (84%) | 230: 169 | 755 | 690 (91%) | 0.500 |
| baseline float @ 0.15 (own) | 2,831 | 2,401 (85%) | 230: 225 | 126 | 78 | 0.944 |
| bn_folded float @ 0.45 | 382 | 251 (66%) | 111: 74 | 1,020 | 912 (89%) | 0.340 |
| bn_folded float @ 0.40 (own) | 803 | 542 (67%) | 111: 140 | 891 | 800 (90%) | 0.421 |
| bn_folded INT8 @ 0.45 | 416 | 282 (68%) | 111: 80 | 1,011 | 904 (89%) | 0.345 |

- **Record 232 produces few false positives** (5–9 per model) but **most of validation's false negatives.** At the deployed INT8-val thresholds, its A-beat recall is only 0.34–0.50.
- Missed abnormal beats are the clinically more important failure.
- Getting record 232's recall up costs false positives elsewhere. Baseline float at 0.15 reaches 0.944 but flags 87% more normal beats on record 109 (2,401 vs 1,281).

### 3. Excluding record 232 doesn't fix validation

| Model | Val PR-AUC, all → excl. 232 | Best val F1, all → excl. 232 (threshold) | Test PR-AUC |
|---|---|---|---|
| baseline float | 0.513 → 0.463 | 0.499 (0.15) → 0.471 (0.70) | 0.961 |
| bn_folded float | 0.506 → 0.370 | 0.455 (0.40) → 0.426 (0.60) | 0.893 |
| bn_folded INT8 | 0.501 → 0.362 | 0.451 (0.45) → 0.424 (0.60) | 0.891 |

- Without record 232, only 217 abnormal beats remain against roughly 14k normals. Record 109's false positives then dominate.
- The best threshold jumps from 0.15 to 0.70 for baseline float. So **record 232 sets the threshold** (§5), and record 109 caps the F1.

### 4. By abnormal type: the models are mainly ventricular detectors

Recall at the threshold picked on INT8 validation:

| Model | V recall, val / test | S recall, val / test |
|---|---|---|
| baseline float @ 0.35 | 0.798 / 0.979 | 0.512 / 0.735 |
| bn_folded float @ 0.45 | 0.719 / 0.847 | 0.341 / 0.520 |
| bn_folded INT8 @ 0.45 | 0.730 / 0.847 | 0.346 / 0.529 |

Threshold-independent: each type against normal beats, with the other abnormal type excluded. ROC-AUC is the comparable metric across splits; PR-AUC depends on how common each type is.

| Model | V vs. normal ROC-AUC, val / test | S vs. normal ROC-AUC, val / test | V PR-AUC, test | S PR-AUC, test |
|---|---|---|---|---|
| baseline float | 0.932 / 0.998 | 0.892 / 0.985 | 0.970 | 0.295 |
| bn_folded float | 0.887 / 0.990 | 0.871 / 0.935 | 0.905 | 0.297 |
| bn_folded INT8 | 0.889 / 0.989 | 0.872 / 0.935 | 0.902 | 0.294 |

- **S detection is weak in every model.**
  - Record 124's 31 S beats (29 `J`) are missed by every model at these thresholds.
  - bn_folded detects 0 of the 33 `A` beats in record 100. Baseline float detects 14 at 0.35.
- Even within a single type, val ROC-AUC is lower than test, because val's normal beats include record 109.
- **Where the bn_folded recall drop comes from** (0.959 → 0.821 against baseline float): record 233's V beats (165 of the 224 FN at 0.45) and record 100's A beats (33 FN).

### 5. Normal-beat probabilities: val has a heavier upper tail

| Model | p90, val / test | p99, val / test | Fraction above the INT8-val threshold, val / test | Fraction above its own float-val threshold, val / test |
|---|---|---|---|---|
| baseline float | 0.359 / 0.146 | 0.635 / 0.420 | 10.6% / 1.7% (0.35) | 19.7% / 9.7% (0.15) |
| bn_folded float | 0.353 / 0.211 | 0.527 / 0.469 | 2.7% / 1.1% (0.45) | 5.6% / 1.5% (0.40) |
| bn_folded INT8 | 0.359 / 0.215 | 0.539 / 0.469 | 2.9% / 1.1% (0.45) | — |

- **The tail comes from record 109.** There, 51.5% of normal beats score above 0.35 for baseline float, and 11.3% score above 0.45 for bn_folded INT8. bn_folded is much less fooled by LBBB beats than the baseline is.
- **This is how the thresholds break.** Baseline float's own val threshold (0.15) is the point where record 232's A-beat recall reaches 0.944. On test, 0.15 flags 22.9% of record 202's normal beats, 23.5% of 214's and 15.8% of 233's, which gives 1,565 FP.

### 6. Few patients drive each split

- **Test:** 8 patients. Two records hold 87% of the abnormal beats (233, 214), and one record holds 80% of bn_folded INT8's false positives (record 105: 144 of 179).
- **Val:** one record holds 86% of the abnormal beats and about 90% of the false negatives (232), and one holds 66–85% of the false positives (109).
- Every metric and decision rule is therefore effectively a 2–3 patient estimate, independent of the seed-to-seed noise that the multi-seed check will measure.

### Proposal: switch to the de Chazal et al. inter-patient split (not implemented)

Reference: P. de Chazal, M. O'Dwyer, and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," IEEE Transactions on Biomedical Engineering, 2004.

**Recommendation: adopt it, on a new branch after this one merges.**

**DS1 (train + val) and DS2 (test), 22 records each.** Sums over the 19 records per set that are already cached:

| Set | Normal | L | R | Ventricular | Supraventricular | F | Q |
|---|---|---|---|---|---|---|---|
| DS1 | 39,517 | 2,490 | 3,694 | 3,469 | 836 | 414 | 8 |
| DS2 | 38,313 | 4,123 | 3,475 | 2,855 | 1,782 | 388 | 7 |

The missing records are 116, 122 and 207 (DS1), and 117, 228 and 234 (DS2). Rerun `python scripts/analyze_splits.py` without `--no-dechazal` once they've downloaded.

**What it fixes:**
- **LBBB in training.** DS1 contains 109 and 207, both LBBB, so train is no longer LBBB-free.
- **A test set with both types.** DS2 has ventricular and supraventricular beats in volume. It includes 232 and 233, so S detection gets tested properly instead of being invisible in the test set.
- **Lower variance.** 22 test patients instead of 8.
- **Comparability.** Results become comparable with the inter-patient literature.

**Validation carved from DS1, patient-wise:**
- **Exactly one LBBB record in validation, the other in train.** Leaning toward 109 → train and 207 → val, but confirm from the full DS1 counts once 116, 122 and 207 are downloaded.
- Validation must contain both V and S beats. Records with both kinds of beats can fill this in, e.g. 201 (198 V, 128 S) and 220 (94 A). Keep 209 (383 A) and 118 in train so train keeps most of DS1's S beats.
- Finalize the record list, about 20% of DS1 beats, from the full counts.
- Validation will still be a few-patient estimate. Per-type metrics in the comparison table make that visible.
- **Future upgrade: grouped cross-validation over DS1,** with patient-wise folds and pooled out-of-fold predictions used for threshold selection. Thresholds would then rest on all DS1 patients instead of a handful.

**What changes:**
- `TRAIN_RECORDS`, `VAL_RECORDS` and `TEST_RECORDS` in `scripts/train.py`. `analyze_splits.py` already has DS1 and DS2.
- 6 more records to download.
- Paced records stay excluded.
- F and Q beats are still dropped. DS1 holds 414 F beats and DS2 holds 388, which matters for the AAMI multiclass experiment.

**What goes stale:**
- All Run 1 numbers: `variant_comparison.*`, and this file's split-dependent figures (record 214 FP, the threshold of 0.35, §§1–6 above).
- The quantization root cause and the bn_folded fix don't depend on the split and stay valid.
- The firmware headers are unaffected until a model is adopted.

**What reruns it requires (user-run):**
1. baseline and bn_folded trained on the new split (2 training runs)
2. the multi-seed check, seeds 42–44 (4 more runs)
3. Run 2 (R-R normalization)
4. Run 3 (calibration sweep)
5. `analyze_splits.py` on the new split, to confirm val and test now agree on type composition

## de Chazal split results

Baseline and bn_folded trained on the de Chazal split (test = DS2, val = 115, 122, 207, 220, 223), seeds 42, 43, 44.
Full tables: `artifacts/dechazal/variant_comparison.md`. Mean ± std uses ddof=1. Thresholds are picked on INT8 validation.

| Variant | Float PR-AUC | INT8 PR-AUC | INT8 test F1 | INT8 FP | INT8 FN | INT8 V recall | INT8 S recall |
|---|---|---|---|---|---|---|---|
| baseline | 0.771 ± 0.111 | 0.676 ± 0.042 | 0.609 ± 0.022 | 2,156 ± 845 | 1,910 ± 221 | 0.825 ± 0.061 | 0.267 ± 0.125 |
| bn_folded | 0.771 ± 0.010 | 0.766 ± 0.007 | 0.636 ± 0.108 | 3,665 ± 3,630 | 1,208 ± 529 | 0.950 ± 0.044 | 0.430 ± 0.218 |

**bn_folded is the architecture going forward.**
- **The INT8 fix holds on DS2.** Going from float to INT8, bn_folded loses 0.005 PR-AUC; the baseline loses 0.095.
- **The old-split float gap was a single-seed artifact.** The old split had a float PR-AUC gap of 0.961 (baseline) vs 0.893 (bn_folded), measured only at seed 42. On DS2 both variants average 0.771. Baseline float is highly seed-sensitive (0.654, 0.876, 0.782), while bn_folded matches its average with about 10× less variance. Conv → BN → ReLU6 costs no float accuracy, makes training far more stable, and closes the INT8 gap.
- **Excluding record 202 is negligible.** It changes F1 by at most 0.007 in every row.

**Record 207 does not dominate validation.** Per-record validation, INT8 at each run's threshold:

| Run | Threshold | Val FP | From 207 | Val FN | From 207 | From 223 | Threshold without 207 |
|---|---|---|---|---|---|---|---|
| baseline 42 | 0.50 | 108 | 90 | 256 | 16 | 185 | 0.45 |
| baseline 43 | 0.70 | 455 | 426 | 278 | 80 | 164 | 0.65 |
| baseline 44 | 0.55 | 96 | 92 | 384 | 120 | 215 | 0.30 |
| bn_folded 42 | 0.15 | 108 | 54 | 124 | 14 | 99 | 0.15 |
| bn_folded 43 | 0.25 | 80 | 26 | 234 | 21 | 195 | 0.25 |
| bn_folded 44 | 0.30 | 103 | 84 | 224 | 13 | 187 | 0.25 |

- 207 is the largest validation false-positive source, but the totals are small, except in baseline seed 43 (426 of 455).
- For bn_folded, 207 is among the best-classified validation records (F1 0.86–0.93). Leaving it out moves the selected threshold by at most one 0.05 step.
- Validation false negatives come mostly from record 223's missed V beats.

**Threshold instability is the main remaining issue.**
- bn_folded's INT8 PR-AUC varies little across seeds (± 0.007), but its test F1 (± 0.108) and FP (± 3,630) vary a lot. Most of that spread comes from the selected threshold.
- Seed 42 picks 0.15, which gives 108 false positives on validation and 7,849 on test.
- Five validation records don't pin down an operating point that transfers to DS2. Grouped CV over DS1 is next.

**Test error sources (INT8):**
- **False negatives:** record 232 dominates every run, with 553–1,276 of its 1,381 abnormal beats missed. It holds 75% of DS2's S beats, so S recall (0.27–0.43) is largely record 232's recall. The next FN sources are 233, 228 and 200.
- **False positives:** 222 is among the top four FP sources in every run (551–1,103). The others are 212, 213, 111 and 214. In bn_folded seed 42, 212 contributes 1,780 and 213 contributes 1,643. Record 105 contributes about 130–220 per run.
- **Unverified hypothesis:** test contains normal morphologies absent from validation: 222, 212 (RBBB), 111 and 214 (LBBB). That may explain why validation false positives don't predict test false positives.

## Grouped CV results

bn_folded, patient-grouped 5-fold CV over DS1 (`scripts/train_cv.py`), seeds 42, 43, 44. The threshold is picked on pooled out-of-fold (OOF) INT8 scores. The final model trains on all 22 DS1 records for the median of the folds' best epochs and is evaluated once on DS2. Full tables: `artifacts/dechazal/cv_comparison.md`. Mean ± std uses ddof=1.

| Setup | INT8 PR-AUC | INT8 test F1 | INT8 FP | INT8 FN | se_V | se_S | Threshold |
|---|---|---|---|---|---|---|---|
| fixed val (above) | 0.766 ± 0.007 | 0.636 ± 0.108 | 3,665 ± 3,630 | 1,208 ± 529 | 0.950 ± 0.044 | 0.430 ± 0.218 | 0.15 / 0.25 / 0.30 |
| grouped CV | 0.748 ± 0.011 | 0.690 ± 0.036 | 1,366 ± 598 | 1,681 ± 177 | 0.881 ± 0.107 | 0.293 ± 0.101 | 0.50 / 0.50 / 0.50 |

**CV fixes the threshold instability.**
- All three seeds pick 0.50. It is a real peak, not a flat curve: pooled-OOF INT8 F1 is 0.737 / 0.716 / 0.741 at 0.50 and drops 0.001–0.017 one step away (0.45, 0.55). Only 1.8–2.0% of OOF scores fall in [0.45, 0.55].
- Test F1 std drops from 0.108 to 0.036 and FP std from 3,630 to 598. The cost is more FNs (1,681 vs 1,208) and lower V and S sensitivity.
- PR-AUC is 0.018 lower than the fixed-val runs. The final model has no early stopping on a validation set, and its epoch count varies (28 / 16 / 26). The fold best epochs range from 5 to 56. Seed 43 (16 epochs) has the lowest PR-AUC (0.740), but three seeds can't separate an epoch effect from seed noise.
- Excluding record 202 changes F1 by at most 0.002.

**DS2 results depend strongly on which patients are in training.** Each fold model is trained on about 4/5 of DS1. On DS2 (diagnostic only, never used for a choice), fold models span PR-AUC 0.548–0.870, F1 0.463–0.750, FP 74–862 and FN 1,749–3,510. The final models (0.740–0.761) sit inside that range.

**S beats from unseen patients are almost never detected.**
- Pooled OOF S recall over DS1 (815 S beats, 201 excluded) is 0.108 / 0.080 / 0.093 (88 / 65 / 76 detected).
- Record 209 (383 A beats) is the largest OOF FN source in every seed: 369 / 369 / 378 of its 384 abnormal beats are missed. 207, 223 and 215 follow.
- Per fold, OOF S recall is 0.000–0.310, with one outlier at 0.667 (seed 42, fold 3: 220/208/230/115/114).
- This weakens the record-232 hypothesis. If 232 were the only problem, held-out DS1 S beats would be detected. Record 232's DS2 A-beat recall varies by seed (710 / 247 / 158 of 1,381 detected), and so does the S recall on the remaining 455 DS2 S beats: 0.086 / 0.488 / 0.523. The excl-232 figures are derived as se_S × 1,836 minus 232's TP, assuming all 232 abnormal beats are S. The record-232 diagnosis should compute them directly.

**Test error sources (final INT8 model):**
- **False negatives:** 232 contributes 671 / 1,134 / 1,223 (36% / 75% / 73% of FNs). Next are 228 (276 of 365, seed 42 only), 233, 222 and 200 (80–104 of 856).
- **False positives:** 222 is a top-three source in every seed: 203 / 712 / 491 (8.9% / 31.3% / 21.6% of its normal beats). LBBB records also contribute: 214 (282 / 412 in seeds 42 and 43) and 111 (10 / 330 / 121). In seed 42, record 212 (RBBB) is the largest source at 684 (54%). Record 105 contributes 57–122.
- **OOF false positives** come from 203, 114, 230, 215 and 208. None of these are bundle-branch-block records, so OOF FPs still don't predict the DS2 sources (222, 212, 214, 111).

## Stage 1: S-beat and FP diagnosis

These results use no training. They come from the grouped-CV bn_folded runs (seeds 42, 43, 44) and `scripts/diagnose_s_beats.py`. Full tables are in `artifacts/dechazal/stage1/step*.md`. Every input to a Stage 2 decision is DS1 out-of-fold (OOF); the DS2 rows are descriptive. Before any analysis the script checked three things: the recomputed R-R features equal `build_dataset`'s `X_rr` beat for beat, the OOF rows line up with `oof_predictions.npz`, and DS2 per-record FP/FN match each run's `per_record_metrics.csv`.

### Step 0: the R-R features are computed across non-beat annotations

`extract_beats_from_record` takes R-R intervals between consecutive annotations of any kind. MIT-BIH's `.atr` files also contain rhythm changes (`+`), noise (`~`), artifacts (`|`, `x`), flutter waves (`!`) and comments (`"`). A beat counts as affected when any of its 5 R-R features differs from the beat-only computation.

| Split | N affected | S affected | V affected | Median rr_ratio error (N / S / V) | p90 rr_ratio error (N / S / V) |
|---|---|---|---|---|---|
| DS1 | 5,956 / 45,834 (13.0%) | 268 / 943 (28.4%) | 1,207 / 3,788 (31.9%) | 0.114 / 0.139 / 0.119 | 0.417 / 0.473 / 0.442 |
| DS2 | 6,008 / 44,232 (13.6%) | 377 / 1,836 (20.5%) | 1,071 / 3,220 (33.3%) | 0.126 / 0.103 / 0.128 | 0.389 / 0.342 / 0.440 |

Errors are relative, |all-annotation − beat-only| / beat-only, over affected beats.

- **Most of the error enters through the 10-interval lookback.** The median pre_rr error among affected beats is 0. Only 772 N, 28 S and 179 V beats in DS1 have a non-beat annotation directly before them.
- **Abnormal beats are hit about twice as often as normal ones.** Rhythm annotations cluster at arrhythmia episodes.
- **The worst records:** 200 (51.9% of beats), 230 (50.6%), 119 (44.3%) and 222 (43.6%). In 207, 472 ventricular-flutter wave annotations (`!`) give E beats an rr_ratio of 3.28 where the beat-only value is 0.21. In 231, 427 comment annotations affect 34% of beats.
- **Fixing it improves R-R-only S-vs-N separation** (step 2b, current 5 features). Logistic regression: pooled ROC-AUC 0.916 → 0.929, PR-AUC 0.166 → 0.240. Gradient boosting: PR-AUC 0.462 → 0.499, while ROC-AUC goes 0.934 → 0.929. Single-feature `rr_ratio`: pooled AUC 0.775 → 0.814.
- **Training and deployment don't match.** The firmware will compute R-R from Pan-Tompkins detections, which contain beats only. So the deployed model would get beat-only R-R features while being trained on all-annotation ones.

### Step 1: within most patients the model ranks S beats correctly, but their scores are too low

| OOF INT8 (mean ± std over seeds) | ROC-AUC | PR-AUC | Recall at t = 0.50 |
|---|---|---|---|
| S vs N, pooled | 0.771 ± 0.060 | 0.075 ± 0.028 | 0.094 ± 0.014 |
| V vs N, pooled | 0.966 ± 0.003 | 0.860 ± 0.010 | 0.772 ± 0.007 |

Float gives the same numbers (S vs N ROC-AUC 0.772), so quantization isn't the cause.

**Within each patient, S vs. that patient's own N beats (INT8):**

| Record | S beats | ROC-AUC | S recall at t | That patient's N above t |
|---|---|---|---|---|
| 207 | 106 | 0.982 ± 0.007 | 0.053 ± 0.068 | 0.009 |
| 220 | 94 | 0.982 ± 0.011 | 0.323 ± 0.358 | 0.001 |
| 223 | 73 | 0.965 ± 0.019 | 0.037 ± 0.063 | 0.000 |
| 118 | 96 | 0.922 ± 0.069 | 0.181 ± 0.158 | 0.002 |
| 209 | 383 | 0.806 ± 0.198 | 0.029 ± 0.014 | 0.001 |
| 124 | 31 | 0.572 ± 0.016 | 0.000 | 0.000 |

- In four of six patients, the model ranks S above that patient's own normal beats (AUC 0.92–0.98). It still detects almost none of them.
- **S scores sit in the normal tail.** S beats have a median score of 0.233 and a 90th percentile of 0.499. Normal beats have a 90th percentile of 0.273 and a 99th of 0.526. V beats have a median of 0.728.
- **S scores are too low in absolute terms and overlap other patients' normal tails.** This isn't the threshold: the threshold sits at a real F1 peak dominated by V, and no single threshold can separate S from N across patients when pooled PR-AUC is 0.075.
- **209 and 124 are different.** There the model's ranking is also weak: 209 is unstable across seeds (0.583 / 0.869 / 0.964 in step 2b's comparison), and 124 is near chance.

### Step 2: the S information is in the R-R inputs, but the CNN doesn't use it

These are R-R-only S-vs-N classifiers under the same grouped folds: 201 always trains, N is capped at 600 per record, V beats are dropped, and the same beats as the CNN are used. All rows are OOF ROC-AUC.

| Source | Pooled | 118 | 124 | 207 | 209 | 220 | 223 | Pooled PR-AUC |
|---|---|---|---|---|---|---|---|---|
| logreg, current 5 features | 0.916 | 0.982 | 0.812 | 0.988 | 0.981 | 0.988 | 0.980 | 0.166 |
| logreg, 5 beat-only features | 0.929 | 0.996 | 0.836 | 0.989 | 0.992 | 0.993 | 0.996 | 0.240 |
| hgb, current 5 features | 0.934 | 0.998 | 0.474 | 0.846 | 0.995 | 0.970 | 0.974 | 0.462 |
| CNN INT8, seeds 42 / 43 / 44 | 0.781 / 0.707 / 0.826 | 0.843 / 0.964 / 0.960 | 0.569 / 0.589 / 0.558 | 0.975 / 0.983 / 0.988 | 0.583 / 0.869 / 0.964 | 0.994 / 0.974 / 0.977 | 0.959 / 0.951 / 0.987 | 0.078 / 0.045 / 0.101 |

- **A logistic regression on the CNN's own 5 R-R inputs beats the CNN on S.** That holds even with the annotation bug: pooled ROC-AUC 0.916 vs. 0.707–0.826, PR-AUC 0.166 vs. 0.045–0.101. Gradient boosting reaches PR-AUC 0.462. On 209, logreg gets 0.981.
- **So for S, the limit is not missing input information.**
- **Caveat:** the baselines are trained on S vs. N with balanced class weights. The CNN is trained on abnormal vs. normal, where V is 82% of train abnormals and focal alpha = 0.25 weights abnormal beats at 0.25. This step therefore shows that the information exists. It doesn't yet separate the objective from the architecture (an underused R-R branch). Step 3 addresses that.

**The median-based ratio doesn't help.**
- In the 5-feature sets, swapping the mean for the median of the same 10-interval window changes pooled ROC-AUC by +0.002 (logreg) and −0.006 (hgb). Alone, the two are similar: 0.722 for the median vs. 0.727 for the mean.
- **The 90th percentile and the max are worse.** Pooled, alone: 0.699 and 0.679. Within 232: 0.912 and 0.897.
- **232 isn't a masking case (DS2, descriptive).**
  - Its A beats are already separated from its own normal beats by the mean-based `rr_ratio`: within-patient AUC 0.973, or 0.992 beat-only.
  - Their `rr_ratio` (median 0.745, IQR 0.666–0.839) is inside the DS1 S range (pooled DS1 S median 0.883).
  - What's unusual in 232 is its normal beats: median `rr_ratio` 1.767 and `pre_post_ratio` 2.384. These are the long pauses of sinus bradycardia (the record's only rhythm annotation is `(SBR`).
  - So 232's misses aren't explained by unusual R-R values on its A beats.
- **Masking is real in 209, and a longer window fixes it, not the median.**
  - 209's S beats come in SVTA runs (10 `(SVTA` rhythm annotations). The runs fill the 10-interval window, so their `rr_ratio` median is 0.950, barely premature.
  - With a 30-interval mean the median drops to 0.825, and 209's single-feature AUC rises from 0.756 to 0.897. The median of 10 gives 0.731, because a run longer than half the window defeats a median too.
  - Pooled single-feature AUC: mean of 30 gives 0.849, vs. 0.727 for the mean of 10. Inside the 5-feature set the gain is small (0.916 → 0.923).

### Step 3: the CNN uses R-R timing for S and shape for V

Each held-out fold is re-scored by its own INT8 fold model after shuffling, among beats of the same record, the R-R inputs, the spectrograms, or both. Results are averaged over 3 shuffle seeds, then mean ± std over model seeds. The unshuffled scores reproduce `oof_predictions.npz` exactly.

| Pooled OOF | Unshuffled | R-R shuffled | Spectrogram shuffled | Both shuffled |
|---|---|---|---|---|
| S vs N ROC-AUC | 0.771 ± 0.060 | 0.621 ± 0.067 | 0.652 ± 0.106 | 0.482 ± 0.080 |
| V vs N ROC-AUC | 0.966 ± 0.003 | 0.891 ± 0.022 | 0.769 ± 0.052 | 0.554 ± 0.046 |
| S mean score | 0.263 | 0.180 | 0.205 | 0.134 |
| V share above t | 0.772 | 0.556 | 0.249 | 0.170 |
| N share above t | 0.012 | 0.015 | 0.040 | 0.047 |

**Within patient, S vs. own N, change in ROC-AUC:**

| Record | Unshuffled | R-R shuffled | Spectrogram shuffled |
|---|---|---|---|
| 118 | 0.922 | −0.386 ± 0.103 | +0.018 ± 0.076 |
| 207 | 0.982 | −0.347 ± 0.250 | −0.109 ± 0.047 |
| 223 | 0.965 | −0.329 ± 0.072 | −0.102 ± 0.032 |
| 220 | 0.982 | −0.074 ± 0.048 | −0.070 ± 0.046 |
| 209 | 0.806 | −0.065 ± 0.028 | −0.117 ± 0.181 |
| 124 | 0.572 | −0.031 ± 0.041 | −0.043 ± 0.040 |

- **The R-R branch is not ignored.** Shuffling R-R costs S 0.151 ± 0.024 pooled AUC and lowers the mean S score from 0.263 to 0.180. In 118, 207 and 223, S ranking comes mostly from timing: shuffling R-R costs 0.33–0.39 AUC, while shuffling the spectrogram changes it by −0.11 to +0.02.
- **V detection is mostly shape.** Shuffling the spectrogram drops V recall at t from 0.772 to 0.249; shuffling R-R drops it to 0.556. Shape is also what keeps normal beats low: with the spectrogram shuffled, the N share above t goes from 0.012 to 0.040.
- **220 uses both cues redundantly.** Each shuffle alone costs about 0.07, both together 0.475.
- **209 leans on shape (−0.117) more than timing (−0.065).** This fits step 2: its SVTA runs mask prematurity in the 10-interval `rr_ratio`.
- **124 is at chance under every condition.** Its 31 S beats (29 `J`) are separable by neither cue in this model.
- **Shuffling both drops within-patient S-vs-N AUC to 0.500–0.522,** the sanity floor.

**What Stage 1 shows together:**
- The CNN reads the R-R inputs and uses them to rank S beats within a patient (step 3).
- The inputs carry enough information to separate S from N across patients (step 2: logreg pooled AUC 0.916).
- Yet the CNN's S scores stay low in absolute terms, inside other patients' normal tail (step 1: median 0.233).
- So the remaining gap points at how strongly training pushes S beats toward "abnormal": the training objective (a binary label where V is 82% of abnormals, plus focal alpha = 0.25). Missing input information and an ignored R-R branch are ruled out as the main cause.
- Two caveats:
  - The CNN's pooled S ranking (0.771) is still below the R-R-only baseline's. Part of the gap may be how the R-R features are represented (raw seconds and ratios, unnormalized, 10-interval window), not only the objective.
  - The R-R bug (step 0) affects all of these numbers. They are measured on the current, buggy inputs.

## Stage 2 plan

Fixes are ordered from most fundamental to least: input correctness → objective → input representation → architecture → error re-check. Each step's result sets the baseline for the next.

**Rules for every step:**
- One change at a time.
- Each change is retrained under grouped CV (`train_cv.py`) with seeds 42, 43, 44.
- Decisions use only OOF per-class (S vs N, V vs N) ROC-AUC and PR-AUC. DS2 results are reported and never used for a choice.
- A change counts only if its gain is larger than the measured seed-to-seed spread.
- The decision table is `scripts/compare_oof.py`, run with the previous step's runs as the reference. Its INT8 rows decide, since INT8 is what's deployed; float rows are shown alongside.

**Step 1: beat-only R-R intervals.**
- Change: `extract_beats_from_record` computes R-R between consecutive beat annotations only (step 0), with tests.
- Retrain the CV baseline: binary bn_folded, current focal loss (gamma 2, alpha 0.25), seeds 42–44.

**Step 1b: read MLII by signal name.**
- Added after step 1 (see "Stage 2 step 1 results"): record 114 stores V5 in channel 0, and `build_dataset` read channel 0.
- Change: `build_dataset` selects the `MLII` channel by name, and a test checks every DS1/DS2 record.
- Retrain the same CV baseline. That run becomes the baseline for step 2a.

**Step 2a: separate class structure from weighting.** Binary vs. N/S/V multiclass under two weightings, with focal gamma = 2 in every run:

| Weighting | Binary | Multiclass |
|---|---|---|
| unweighted | no alpha | no class weights |
| matched | alpha 0.25 (the step-1 baseline) | N 0.75 / S 0.25 / V 0.25 |

- Binary and multiclass are compared within each pair.
- **Reading the result:** if multiclass wins under both weightings, the benefit is structural. If it wins under only one, structure and weighting interact.
- **Multiclass is N/S/V only:** F and Q are still dropped. The README must say so wherever results are compared with the literature.
- The threshold runs on `abnormal_score` (1 − P(N)), so the CV pipeline is unchanged.

**Step 2b: balanced class weights.**
- Multiclass with inverse-frequency class weights, compared with the better 2a multiclass run.
- Once a multiclass model exists, +P for S and V is reported alongside Se.

**Step 3: R-R representation, only if a gap remains after step 2.**
- 3a: 30-interval lookback window.
- 3b: global standardization (`--normalize-rr`).
- The median-based ratio is dropped (step 2 of Stage 1).

**Step 4: global average pooling instead of Flatten.**
- Judged on per-class AUC and model size.
- Flatten → Dense(64) holds 197,184 of the model's ~202.2k parameters: 3,072 spectrogram features (16 × 6 × 32) plus 8 R-R features, times 64.

**Step 5: re-check false positives.**
- **Re-measure after steps 1–2:** AF/flutter normal-beat false positives (222, 219, 202 on DS2; 203 in DS1 OOF).
- **Known limitations:**
  - Wide-QRS false positives: there are only 4 bundle-branch-block patients in DS1.
  - Record 124's junctional (`J`) beats: neither cue separates them.

**Parked:**
- Pan-Tompkins R-peaks
- DS2 sensitivity analysis
- pruning
- firmware header regeneration
- calibration sweep
- fair QAT
- more training data from other PhysioNet databases (below)

**Parked idea: add training data from other PhysioNet databases.** Not implemented.
- **Why:** DS1 has only 22 patients, so no single validation group is representative, and fold models vary a lot with the patients they train on.
  - Only 2 of the 22 are LBBB patients (109, 207).
  - Record 209 holds 383 of DS1's 944 S beats (41%), counted over all annotations.
  - Fold-model DS2 PR-AUC (diagnostic only, `folds.json`) ranges from 0.548 to 0.870 in the pre-fix runs, and from 0.683 to 0.873 in step 1.
- **Candidates:**
  - MIT-BIH Supraventricular Arrhythmia Database (SVDB): 78 recordings, rich in S beats, 128 Hz.
  - St. Petersburg INCART: 75 recordings from 32 patients, 12-lead, 257 Hz.
- **Rule:** the new data goes into training only. DS2 stays the untouched test set, so results stay comparable with de Chazal-split papers.
- **Work needed:**
  - Resample to 360 Hz.
  - Pick the lead closest to MLII: SVDB's leads differ, and INCART would use lead II. Add a test, like the step 1b MLII check.
  - Check that the annotation symbols map to the AAMI classes correctly (`SYMBOL_MAP`, `AAMI_CLASSES`).
  - Group CV folds by patient, since INCART has several recordings per patient.
- **When:** after the Stage 2 experiments, so there is a DS1-only baseline to compare against.
- **Success test:** pooled OOF S-beat PR-AUC (S vs. N) improves by more than the seed spread (seeds 42–44).
  - Also recorded, but not pass/fail: S recall at the operating point that gives 0.90 overall abnormal recall on the pooled OOF scores. `compare_oof.py` doesn't report this yet; it reports S recall at the selected threshold.
  - The OOF pool stays the DS1 records; external recordings are only ever in training.
  - Decisions come from OOF results (`compare_oof.py`, INT8 rows), never DS2.

**Required before deployment (not yet fixed): the final-model procedure.**
- This covers the final model's epoch count (the median of the folds' best epochs, which range from 4 to 56 across the pre-fix and step 1 runs) and carrying the pooled-OOF threshold over to it.
- In step 1, seed 44's final model produced 12,917 DS2 FPs at its OOF threshold of 0.40, while seeds 42 and 43 produced 96 and 300 (see "Stage 2 step 1 results").
- OOF-based decisions are unaffected: they compare fold models scored on held-out patients. The deployable model, a single final model plus a transferred threshold, is not yet reliable.
- Stage 2 continues on OOF decisions. This must be fixed before any model is exported to firmware.

### Step 4: false positives on DS1 OOF are not wide-QRS; on DS2 they have two sources

- **OOF, INT8, at t = 0.50:**
  - L beats: 24.0 ± 13.9 FP of 3,947 (0.6%).
  - R beats: 4.7 ± 4.7 of 3,779 (0.1%).
  - N beats: 510.7 ± 89.9 of 36,452 (1.4%).
  - Every fold model trains on 1–2 LBBB patients (515–1,115 beats) and 2–3 RBBB patients (683–1,283 beats).
- **DS2, final models (descriptive).** Seeds 42 / 43 / 44, FP / normal beats of that symbol:

| Source | Record | Symbol | FP, seeds 42 / 43 / 44 | Total |
|---|---|---|---|---|
| Wide QRS | 212 | R | 684 / 78 / 2 | 1,825 |
| Wide QRS | 214 | L | 282 / 412 / 12 | 2,001 |
| Wide QRS | 111 | L | 10 / 330 / 121 | 2,122 |
| AF/flutter | 222 | N | 203 / 706 / 479 | 2,060 |
| AF/flutter | 219 | N | 0 / 190 / 4 | 2,081 |
| AF/flutter | 202 | N | 1 / 93 / 2 | 2,060 |
| Noise | 105 | N | 57 / 113 / 122 | 2,525 |

  Rhythm annotations: 222 has AFIB 24, AFL 42 and NOD 31; 219 has AFIB 10; 202 has AFIB 4 and AFL 1. 105 has 88 noise annotations (`~`).

- **Wide-QRS false positives depend heavily on the seed.** For example, 214 has 282 / 412 / 12, even though every fold model handles DS1's L and R beats. This looks like generalization variance across a few bundle-branch-block patients (4 in DS1), not missing coverage.
- **The AF/flutter normal beats are an R-R problem, not a shape problem.** An irregular rhythm makes normal beats look premature. 222 is also one of the records most affected by the annotation bug (43.6% of beats). In DS1, 203 (AFIB 21, AFL 2) is the top OOF FP source.

## Stage 2 step 1 results: beat-only R-R

The CV baseline was retrained with beat-only R-R intervals (`train_cv.py --seeds 42 43 44 --artifacts-dir artifacts/dechazal/stage2/step1_beat_rr`). Everything else is unchanged: binary bn_folded, focal gamma 2 / alpha 0.25. The pre-fix runs are `artifacts/dechazal/cv/`. The OOF pool shrinks from 48,605 to 48,596 beats: 9 normal beats at record edges are no longer usable.

**Decision metrics: pooled OOF, INT8, per class vs. N.** Values are seeds 42 / 43 / 44, then mean ± std.

| Metric | Pre-fix | Beat-only R-R | Gain | Exceeds the seed spread? |
|---|---|---|---|---|
| V ROC-AUC | 0.968 / 0.965 / 0.963 = 0.966 ± 0.003 | 0.976 / 0.972 / 0.972 = 0.973 ± 0.002 | +0.008 | Yes. Every new seed beats every pre-fix seed. |
| V PR-AUC | 0.865 / 0.849 / 0.868 = 0.860 ± 0.010 | 0.879 / 0.892 / 0.912 = 0.894 ± 0.016 | +0.034 | Yes. The lowest new seed (0.879) beats the highest pre-fix seed (0.868). |
| S ROC-AUC | 0.781 / 0.707 / 0.826 = 0.771 ± 0.060 | 0.795 / 0.782 / 0.793 = 0.790 ± 0.007 | +0.019 | No. The gain is smaller than the pre-fix spread (0.060), and pre-fix seed 44 (0.826) beats every new seed. The spread itself shrinks about 9×. |
| S PR-AUC | 0.078 / 0.045 / 0.101 = 0.075 ± 0.028 | 0.104 / 0.086 / 0.176 = 0.122 ± 0.048 | +0.047 | No. The gain is about the same as the new spread (0.048), and pre-fix seed 44 (0.101) beats new seed 43 (0.086). |

Full table: `artifacts/dechazal/stage2/step1_beat_rr/oof_comparison.md` (`compare_oof.py`). The float AUCs agree with INT8 to within 0.018; the largest gap is seed 43's S PR-AUC, 0.104 float vs. 0.086 INT8. That gap matters for one metric. On float, the S PR-AUC gain does exceed the spread: +0.053 vs. 0.040, with seeds separated, since the lowest new seed (0.104) beats the highest pre-fix seed (0.100). On INT8 it doesn't. The INT8 rows decide.

- **Verdict under the Stage 2 rules:**
  - The fix gives a real V gain.
  - S doesn't improve by more than the seed noise. The S gap is still open, so step 2 goes ahead as planned.
  - The fix stays in regardless, because it corrects an input bug. This run is the baseline for step 2a.
- **Per-record S ranking (OOF INT8 ROC-AUC, seeds 42 / 43 / 44):**
  - 118: 0.992 / 0.978 / 0.999 (pre-fix 0.843 / 0.964 / 0.960)
  - 207: 0.995 / 0.987 / 0.992
  - 223: 0.978 / 0.992 / 0.990
  - 220: 0.846 / 0.914 / 0.983 (pre-fix 0.994 / 0.974 / 0.977)
  - 209 is still unstable from seed to seed: 0.853 / 0.859 / 0.577 (pre-fix 0.583 / 0.869 / 0.964).
  - 124 is still near chance: 0.630 / 0.510 / 0.664.

**At the selected threshold (OOF INT8; descriptive, not decision metrics):**
- The threshold is no longer the same for every seed: it is 0.50 / 0.55 / 0.40 (pre-fix: 0.50 for all three).
- **FN side:**
  - S recall: 0.238 / 0.103 / 0.371 = 0.237 ± 0.134 (pre-fix 0.094 ± 0.014).
  - V recall: 0.853 ± 0.024 (pre-fix 0.772 ± 0.007).
  - Median S score: 0.279 ± 0.052 (pre-fix 0.233 ± 0.033).
- **FP side:**
  - FP rate on N beats: 0.016 ± 0.006 (pre-fix 0.012 ± 0.002).
  - Total OOF FPs: 985 / 447 / 687 (pre-fix 537 / 638 / 443).
  - Top sources: 203 (AF/flutter) with 493 / 366 / 411 (pre-fix 225 / 152 / 65), and 114 with 418 / ≤7 / 182 (pre-fix 105 / 149 / 286).

**DS2, final models (reported, never used for a choice):**
- INT8 PR-AUC 0.925 / 0.770 / 0.777 = 0.824 ± 0.087 (pre-fix 0.748 ± 0.011).
- F1 0.716 / 0.753 / 0.429 = 0.633 ± 0.177 (pre-fix 0.690 ± 0.036).
- S sensitivity 0.190 / 0.132 / 0.974; V sensitivity 0.784 / 0.929 / 0.969.
- **Seed 44 collapses at its OOF threshold.** It produces 12,917 FPs at t = 0.40:
  - wide QRS: 212 (R) 1,863, 214 (L) 1,753, 111 (L) 746
  - AF/flutter: 222 1,305, 219 844, 202 679
  - noise: 105 155
- **Seeds 42 and 43 have few FPs:** 96 and 300.
  - 222: 83 / 239
  - 105: 10 / 37
  - Wide QRS: 212, 214 and 111 together 2 / 13
- **DS2 variance went up, not down.** OOF-to-DS2 threshold transfer is fragile: a threshold 0.10 lower, plus a final model trained for a different epoch count (23), turned a ranking with ROC-AUC 0.953 into 12,917 FPs. This ties into the parked final-model epoch-count item. It's recorded here and not acted on, because DS2 is never used for a choice.

**New input finding: record 114's leads are swapped.**
- `build_dataset` uses `p_signal[:, 0]` as MLII. In record 114, channel 0 is V5 and channel 1 is MLII.
- It is the only such record among DS1 and DS2; 102 and 104 are the others in MIT-BIH, and they are excluded as paced.
- 114 is a top OOF FP source in both the pre-fix and the beat-only runs (counts above).
- Like the R-R bug, this is an input-correctness issue. It is not fixed in step 1 (one change per step); it becomes step 1b.
