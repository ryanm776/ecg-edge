# ECG classifier: investigation and experiment log

> **Current status:** Stage 2 is in progress. The latest results are in the last section, "Stage 2 step 2b results: balanced class weights": no weighting wins, so nsv_matched (N/S/V, weights N 0.75 / S 0.25 / V 0.25) carries forward. Step 3 (R-R representation) is designed in "Step 3 design" at the end of that section, before any step 3 run. The remaining steps are in "Stage 2 plan". No model has been exported to firmware since the old-split baseline.

## Contents

Investigations in the order they were run, each with its outcome:

- **Float → INT8 gap** (old split): the root cause is BatchNorm after ReLU6, which can't be folded into the conv. Fixed by the bn_folded architecture (Conv → BN → ReLU6). QAT is parked.
- **Validation/test mismatch** (old split): validation's abnormal beats were mostly one patient's S beats, test's were mostly V, and train had no LBBB. This led to the switch to the de Chazal split.
- **de Chazal split results:** bn_folded matches the baseline's float PR-AUC on DS2 with ~10× less seed variance and no INT8 gap, so it is the architecture going forward. A fixed 5-record validation set gives an unstable threshold.
- **Grouped CV results:** patient-grouped CV over DS1 stabilizes the threshold (0.50 for all three seeds). S beats from unseen patients are almost never detected (OOF S recall 0.08–0.11).
- **Stage 1: S-beat and FP diagnosis:** the S failure lies mainly in the training objective, plus an R-R input bug (intervals spanning non-beat annotations).
- **Stage 2 plan:** the fixes in order, one at a time, each decided on OOF per-class AUC against the seed-to-seed spread.
- **Stage 2 step 1 results: beat-only R-R:** a real V gain (INT8 OOF V PR-AUC 0.860 → 0.894); the S gain is within the seed spread.
- **Stage 2 step 1b results: MLII by signal name:** record 114's swapped leads are fixed, with no pooled gain beyond the seed spread. The S gap is still open.
- **Stage 2 step 2a results: binary vs. N/S/V:** no structural multiclass benefit; N/S/V matched shows no loss against binary and carries forward for AAMI comparability and a separate S weight.
- **Stage 2 step 2b results: balanced class weights:** no weighting wins on S PR-AUC. Balanced weights fail the V guard and mainly shift scores up rather than ranking S better. nsv_matched carries forward; step 3 is designed with 5 seeds.

> **Old split.** "Float → INT8 gap" and "Validation/test mismatch" use the old hand-picked train/val/test split, as does every committed file directly under `artifacts/` (model, `metrics.json`, `variant_comparison.*`, …). Everything from "de Chazal split results" on uses the de Chazal split; those results are in `artifacts/dechazal/` (see README.md, "Data split").

## Float → INT8 gap

Baseline model (`ecg_multiscale_model.keras` / `model_int8.tflite`, seed 42) on test, threshold 0.35:

| Model | Precision | Recall | FP | FP rec 214 | FP excl. 214 |
|---|---|---|---|---|---|
| float | 0.810 | 0.959 | 282 | 79 | 203 |
| INT8 | 0.433 | 0.917 | 1,502 | 1,146 | 356 |

**Hypothesis tested:** INT8 input quantization (calibrated on 100 samples) clips the unnormalized R-R features.
**Verdict:** rejected. The gap comes from inside the network: Conv → ReLU6 → BatchNorm ordering leaves BN unfolded and tensors with ~4–5 INT8 levels.

All numbers below were reproduced with seed 42, so the PTQ calibration indices are identical to the ones `scripts/train.py` used.

### 1. Float32 TFLite matches Keras

Converting the same model to an unquantized float32 `.tflite` (17,466 test beats):

- max |Δprob| = 6.17e-6
- mean |Δprob| = 4.13e-7
- 0 decision flips at threshold 0.35

So the conversion pipeline isn't the cause.

### 2. Input quantization parameters

| Tensor | Scale | Zero point | Representable range |
|---|---|---|---|
| `rr_input` | 0.009004 | −128 | [0, 2.296] |
| `spectrogram_input` | 0.003922 (1/255) | −128 | [0, 1.000] |
| output | 0.003906 (1/256) | −128 | [0, 1.000] |

- The `rr_input` upper bound (2.296) is the calibration sample's max `pre_post_ratio`.
- Spectrograms are min-max normalized to [0, 1], so that input never clips.

### 3. R-R values outside the representable range

| Set | Beats | Values clipped | Beats with any clip | Per feature (pre_rr / post_rr / local_mean / rr_ratio / pre_post_ratio) |
|---|---|---|---|---|
| calibration (100) | 100 | 0.000% | 0.000% | 0 / 0 / 0 / 0 / 0 |
| train | 38,582 | 0.203% | 1.006% | 0.01 / 0.00 / 0.00 / 0.02 / 0.99 % |
| val | 15,996 | 0.620% | 2.313% | 0.56 / 0.57 / 0.00 / 0.49 / 1.48 % |
| test | 17,466 | 0.134% | 0.670% | 0.00 / 0.00 / 0.00 / 0.01 / 0.66 % |
| record 214 | 2,257 | 0.133% | 0.665% | 0.00 / 0.00 / 0.00 / 0.00 / 0.66 % |
| record 232 | 1,779 | 5.329% | 19.562% | 5.06 / 5.12 / 0.00 / 4.44 / 12.03 % |

Record 232's max values are pre/post_rr 5.87 s, rr_ratio 4.64 and pre_post_ratio 12.4. Train's max `pre_post_ratio` is 8.76.

### 4. Fake-quantizing the inputs doesn't reproduce the gap

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

### 5. Root cause: BatchNorm after ReLU6 can't be folded

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

### 6. Calibration size is a secondary effect

Re-running PTQ with larger shuffled calibration sets (test FP @ 0.35):

| Calibration samples | FP | FP rec 214 | TP | Precision |
|---|---|---|---|---|
| 100 (baseline) | 1,502 | 1,146 | 1,149 | 0.433 |
| 1,000 | 1,218 | 936 | 1,135 | 0.482 |
| 5,000 | 1,271 | 975 | 1,136 | 0.472 |

### 7. QAT compatibility (tfmot)

- Versions: TF 2.21.0, Keras 3.15.1, tf_keras 2.21.0, tensorflow-model-optimization 0.8.1. tfmot imports fine.
- Importing tfmot makes `tf.keras` resolve to `tf_keras` (Keras 2) for the whole process. QAT therefore has to run in its own process, separate from the Keras 3 training path.
- `tfmot.quantization.keras.quantize_model` on the current architecture fails both with and without `TF_USE_LEGACY_KERAS=1`: `Layer batch_normalization ... is not supported`. Neither BN placement (after Concatenate, or after Conv+ReLU6) is supported by the default 8-bit registry.

### 8. Unrelated firmware bug found along the way

The INT8 flatbuffer's subgraph inputs are `[rr_input (tensor 0), spectrogram_input (tensor 1)]`. `deployment_model.ino` bound `input(0)` as the spectrogram, so on device the two inputs were swapped, and the `min(bytes, 5)` bounds check hid it. This is fixed in commit f94682d: inputs are now bound by element count.

### Next step (as planned at the time)

Compare, on the same seed:

- (a) this baseline
- (d) PTQ with Conv → BN → ReLU6 reordering, so BN folds into the conv
- (e) QAT on the reordered architecture

Then add R-R normalization to the better of d/e as a separate row. Results will go in `variant_comparison.md`.

### QAT follow-up (parked)

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
- Each change is retrained under grouped CV (`train_cv.py`) with seeds 42, 43, 44. From step 3 on, seeds 42–46, the reference included ("Step 3 design").
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
- **Implementation choices (agreed before the runs):**
  - The binary-matched cell reuses the step 1b runs. A test pins the binary model's initial weights at seed 42 to the pre-2a builder's.
  - "No alpha" means weight 1 on both classes, not alpha = 0.5. Alpha 0.5 would halve the data loss against the fixed L2 penalty.
  - Decisions use OOF per-class AUC on `abnormal_score` for both models. AUCs on P(S) and P(V) are descriptive only.
  - A multiclass beat is called N unless 1 − P(N) > threshold, otherwise the likelier of S and V. Se and +P count that call, so an S beat called V is an S miss. S→V and V→S counts are reported.
  - The INT8 − float gap of each decision metric is reported for the multiclass runs, as for binary.
- **Final-model calibration is deferred until after Stage 2.** Step 2a decides from OOF results only, so the final-model procedure ("Required before deployment" below) doesn't affect the decision.

**Step 2b: balanced class weights.**
- Multiclass with inverse-frequency class weights, compared with both 2a N/S/V runs. The full design, fixed before any 2b run, is in "Stage 2 step 2a results", "Step 2b design".
- Once a multiclass model exists, +P for S and V is reported alongside Se.
- **Normalize the weights** so that their mean, weighted by class frequency in the training data, is 1 (not implemented yet).
  - 2b compares weighted with unweighted losses. Raw inverse-frequency weights would change the data loss's overall scale, and with it the balance against L2.
  - Normalizing keeps that scale fixed, so only the class balance changes.

**Step 3: R-R representation, only if a gap remains after step 2.** A gap remains, so step 3 runs. The full design, fixed before any step 3 run, is in "Stage 2 step 2b results", "Step 3 design".
- 3a: 30-interval lookback window.
- 3b: global standardization (`--normalize-rr`), on top of the 3a winner or nsv_matched.
- Afterwards: decide whether the SVDB/INCART data work starts before steps 4 and 5.
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
- the training normal-beat cap: cap vs. no cap, and random vs. first-N sampling under the cap (below)
- nested validation, the fix for early stopping on the held-out fold (below)

**Known limitation: early stopping on the held-out fold.** This affects every OOF result, Stage 2 included.
- In grouped CV (`train_cv.py`), each fold model early-stops on its held-out fold, with best weights restored. Those same held-out beats become its OOF predictions.
- So each fold model's stopping epoch is chosen on the patients it is then scored on. That makes the OOF metrics optimistic, by an amount not measured.
- Comparisons between runs are fairer when the runs share a stopping criterion, since the bias then acts on all of them alike. That is why nsv_balanced stops on the unweighted val loss ("Step 2b design").
- **Fix, parked until after Stage 2: nested validation.** An inner, patient-grouped split inside each fold's training records would choose the stopping epoch. The held-out fold would then be used only for scoring.

**Parked idea: how the training cap samples normal beats.** Not changed in Stage 2.
- **Current behavior:** training keeps the first 600 N beats of each DS1 record in annotation order (`--max-normal-per-record-train`, `extract_beats_from_record`). Evaluation (OOF, DS2, val) is uncapped.
- **Every DS1 record hits the cap.** Each has 1,507–3,195 N-class annotations (`SYMBOL_MAP` class 0, before the window and R-R-neighbor rules), so the cap keeps about 19–40% of them. The 600th falls between minute 5.6 (215) and minute 12.5 (124) of the 30-minute recording, with a median of 8.2. So training normals come only from the start of each record.
- **Two questions, parked together:**
  - cap vs. no cap
  - under the cap, random vs. first-N sampling
- **What this affects:** training representativeness only. The evaluation numbers already cover every beat, so they're unaffected.

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
- **Success test:** pooled OOF S-beat PR-AUC (S vs. N) improves by more than the seed spread (seeds 42–44; 42–46 if the work follows step 3, see "Step 3 design").
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

## Stage 2 step 1b results: MLII by signal name

The CV baseline was retrained with record 114 read from its MLII channel (`train_cv.py --seeds 42 43 44 --artifacts-dir artifacts/dechazal/stage2/step1b_mlii`). Everything else matches step 1: beat-only R-R, binary bn_folded, focal gamma 2 / alpha 0.25. The OOF pool is the same 48,596 beats. Only record 114's 1,873 DS1 spectrograms changed (all of them).

**Decision metrics: pooled OOF, INT8, per class vs. N.** Values are seeds 42 / 43 / 44, then mean ± std. The full table is `artifacts/dechazal/stage2/step1b_mlii/oof_comparison.md`, with step 1 as the reference.

| Metric | Step 1 | Step 1b | Gain | Exceeds the seed spread? |
|---|---|---|---|---|
| S ROC-AUC | 0.795 / 0.782 / 0.793 = 0.790 ± 0.007 | 0.772 / 0.842 / 0.788 = 0.801 ± 0.037 | +0.011 | No (spread 0.037) |
| S PR-AUC | 0.104 / 0.086 / 0.176 = 0.122 ± 0.048 | 0.096 / 0.151 / 0.232 = 0.160 ± 0.068 | +0.038 | No (spread 0.068) |
| V ROC-AUC | 0.976 / 0.972 / 0.972 = 0.973 ± 0.002 | 0.975 / 0.973 / 0.984 = 0.977 ± 0.006 | +0.004 | No (spread 0.006) |
| V PR-AUC | 0.879 / 0.892 / 0.912 = 0.894 ± 0.016 | 0.889 / 0.914 / 0.939 = 0.914 ± 0.025 | +0.020 | No (spread 0.025) |

No float row exceeds the spread either.

- **Verdict under the Stage 2 rules:**
  - No pooled gain exceeds the seed spread. That is expected, since one record of 21 in the OOF pool changed.
  - The fix stays in regardless, because it corrects an input bug. This run is the baseline for step 2a.
  - The S gap is still open.
- **Record 114 is fixed** (OOF INT8, at each run's threshold, seeds 42 / 43 / 44):
  - **FP side:** normal-beat false positives in 114 fall from 418 / 0 / 182 to 0 / 0 / 0, out of 1,818.
  - **FN side:** for its 43 V beats, V-vs-N ROC-AUC rises from 0.857 / 0.542 / 0.730 to 0.978 / 1.000 / 0.966, and V recall from 0.814 / 0.000 / 0.465 to 0.674 / 1.000 / 0.744.
  - Under the swapped lead, 114's V beats were barely separable from its normal beats for seed 43.
- **Per-record S ranking (OOF INT8 ROC-AUC):** 118, 207 and 223 stay at 0.928–0.999, and 124 stays near chance (0.555–0.697). 209 is 0.809 / 0.814 / 0.708, against step 1's 0.853 / 0.859 / 0.577.

**At the selected threshold (OOF INT8; descriptive):**
- The threshold is 0.45 for every seed (step 1: 0.50 / 0.55 / 0.40).
- **FN side:**
  - S recall: 0.177 / 0.156 / 0.387 = 0.240 ± 0.128 (step 1: 0.237 ± 0.134).
  - V recall: 0.852 ± 0.034.
  - Median S score: 0.310 ± 0.067.
- **FP side:**
  - FP rate on N beats: 0.010 ± 0.004 (step 1: 0.016 ± 0.006).
  - Total OOF FPs: 603 / 235 / 529 (step 1: 985 / 447 / 687).
  - 203 (AF/flutter) is now the only large source: 487 / 170 / 330.

**DS2, final models (reported, never used for a choice):**
- INT8 PR-AUC 0.821 / 0.820 / 0.809 = 0.817 ± 0.007 (step 1: 0.824 ± 0.087).
- F1 0.736 / 0.780 / 0.747 = 0.754 ± 0.023 (step 1: 0.633 ± 0.177).
- FP / FN: 739 / 1,679, 485 / 1,513 and 1,051 / 1,415.
- S sensitivity 0.301 / 0.304 / 0.371; V sensitivity 0.877 / 0.927 / 0.919.
- **No seed collapses this time.**
  - AF/flutter normal beats are now the main FP source: 222 gives 551 / 322 / 545, 219 gives 25 / 4 / 193, and 202 gives 15 / 60 / 167.
  - Wide-QRS FPs (212, 214 and 111 combined) are 38 / 58 / 28.
  - Noise (105): 99 / 37 / 33.
- **The deployment requirement still stands.** One set of three seeds without a collapse doesn't show that the final-model procedure is reliable.
  - Final epochs were 13 / 15 / 26.
  - Fold best epochs now go as low as 1 (seed 43, fold 0: `[1, 22, 24, 15, 11]`).

## Stage 2 step 2a results: binary vs. N/S/V

Binary and N/S/V multiclass models were trained under two weightings, with focal gamma 2 throughout (`train_cv.py --seeds 42 43 44`). Everything else matches step 1b: beat-only R-R, MLII by name, bn_folded.

| Weighting | Binary | N/S/V |
|---|---|---|
| unweighted | `--loss-weights none`, in `stage2/step2a_binary_unweighted` | `--classes nsv --loss-weights none`, in `stage2/step2a_nsv_unweighted` |
| matched | alpha 0.25: the step 1b runs, `stage2/step1b_mlii` | N 0.75 / S 0.25 / V 0.25: `--classes nsv`, in `stage2/step2a_nsv_matched` |

- The binary-matched cell reuses step 1b. A test pins the binary model's seed-42 initial weights to the pre-2a builder's.
- The OOF pool is the same 48,596 beats: 44,191 N, 815 S, 3,590 V. 201 is never held out.
- The multiclass model covers N, S and V only. F and Q beats are dropped, as in the binary runs.

**Decision metrics: pooled OOF, INT8, per class vs. N.** Values are seeds 42 / 43 / 44, then mean ± std. "Exceeds" means |gain| > the larger seed std. "Separated" means every candidate seed is on the gain's side of every reference seed. Tables: `stage2/step2a_unweighted/oof_comparison.md` (reference binary_unweighted) and `stage2/step2a_matched/oof_comparison.md` (reference step 1b).

Unweighted row:

| Metric | Binary | N/S/V | Gain | Exceeds / separated |
|---|---|---|---|---|
| S ROC-AUC | 0.821 / 0.869 / 0.772 = 0.821 ± 0.049 | 0.857 / 0.832 / 0.782 = 0.824 ± 0.038 | +0.003 | no / no |
| S PR-AUC | 0.201 / 0.242 / 0.196 = 0.213 ± 0.025 | 0.213 / 0.104 / 0.115 = 0.144 ± 0.060 | −0.069 | yes / no: N/S/V seed 42 (0.213) beats binary seeds 42 and 44 |
| V ROC-AUC | 0.968 / 0.987 / 0.982 = 0.979 ± 0.010 | 0.969 / 0.971 / 0.968 = 0.969 ± 0.001 | −0.010 | yes, by under 0.001 / no |
| V PR-AUC | 0.895 / 0.936 / 0.932 = 0.921 ± 0.022 | 0.880 / 0.884 / 0.886 = 0.883 ± 0.003 | −0.038 | yes / yes: the highest N/S/V seed (0.886) is below the lowest binary seed (0.895) |

Matched row:

| Metric | Binary (step 1b) | N/S/V | Gain | Exceeds / separated |
|---|---|---|---|---|
| S ROC-AUC | 0.772 / 0.842 / 0.788 = 0.801 ± 0.037 | 0.874 / 0.828 / 0.883 = 0.862 ± 0.030 | +0.061 | yes / no: N/S/V seed 43 (0.828) is below step 1b seed 43 (0.842) |
| S PR-AUC | 0.096 / 0.151 / 0.232 = 0.160 ± 0.068 | 0.117 / 0.087 / 0.168 = 0.124 ± 0.041 | −0.036 | no / no |
| V ROC-AUC | 0.975 / 0.972 / 0.984 = 0.977 ± 0.006 | 0.973 / 0.976 / 0.982 = 0.977 ± 0.005 | −0.001 | no / no |
| V PR-AUC | 0.889 / 0.914 / 0.939 = 0.914 ± 0.025 | 0.904 / 0.899 / 0.934 = 0.912 ± 0.019 | −0.002 | no / no |

The float rows give the same verdicts. INT8 − float: the largest per-seed gap of any decision metric is 0.021 (binary unweighted, S PR-AUC). For both N/S/V runs it is at most 0.007, so the N/S/V head has no quantization gap.

- **Reading under the plan's rule:**
  - Unweighted: N/S/V is worse. The V PR-AUC loss is separated; the S PR-AUC loss exceeds the spread but isn't separated.
  - Matched: N/S/V shows no loss. Three metrics are within the spread. S ROC-AUC gains beyond the spread but isn't separated.
  - N/S/V doesn't win under both weightings, so it has no structural benefit. The two rows disagree, so class structure and weighting interact.
- **Decision: nsv_matched carries forward.** It is the reference for step 2b, and the default if 2b produces no winner. The grounds don't depend on the 2a numbers:
  - AAMI comparability. An N/S/V model gives per-class Se and +P, as reported in de Chazal-split papers; for a binary model, +P per class is undefined.
  - Only N/S/V lets the loss weight S separately from V, and S is the open gap.
  - The matched row is read as no loss against binary, not as a win.

**Descriptive (never decisive):**
- **For S, the class's own probability ranks worse than the abnormal score.** S vs. N ROC-AUC on P(S) is 0.752 ± 0.074 (unweighted) and 0.805 ± 0.074 (matched), against 0.824 and 0.862 on the abnormal score. For V, P(V) and the abnormal score agree (0.969–0.979).
- **S ↔ V mix-ups at the selected threshold:**
  - nsv_unweighted: S called V 193 / 109 / 31, V called S 88 / 75 / 153.
  - nsv_matched: S called V 44 / 13 / 101, V called S 258 / 211 / 374.
- **Per-record S ranking (OOF INT8 ROC-AUC, seeds 42 / 43 / 44):**
  - 209: nsv_matched 0.975 / 0.931 / 0.917 against step 1b's 0.809 / 0.814 / 0.708. Every N/S/V seed is above every step 1b seed.
  - 118: nsv_matched 0.384 / 0.898 / 0.984 (step 1b: 0.989 / 0.928 / 0.999); nsv_unweighted 0.956 / 0.905 / 0.564.
  - 207, 220 and 223 stay at 0.886–0.997 in every cell; 124 stays near chance (0.359–0.754).
- **At the selected threshold** (pooled-OOF INT8, F1-maximizing):
  - Thresholds: binary unweighted 0.65 / 0.60 / 0.50, nsv_unweighted 0.70 / 0.60 / 0.70, step 1b 0.45 for every seed, nsv_matched 0.65 / 0.60 / 0.50.
  - S recall (flagged abnormal): binary unweighted 0.307 ± 0.083, nsv_unweighted 0.273 ± 0.143, step 1b 0.240 ± 0.128, nsv_matched 0.188 ± 0.105.
  - FP rate on N beats: 0.010–0.013 in every cell.

**Operating points at fixed target recall** (pooled OOF, INT8, mean ± std over seeds; `oof_comparison.md`, "Operating points"). The target recall is abnormal recall (S and V beats with abnormal_score > threshold), and each threshold is the highest that reaches it. "Flagged" Se counts a beat flagged abnormal. The "call" columns (N/S/V only) use the call rule, so an S beat called V is an S miss and counts against +P_V.

| Cell | Target recall | Precision | FP | Se_S flagged | Se_V flagged | Se_S call | Se_V call | +P_S call | +P_V call |
|---|---|---|---|---|---|---|---|---|---|
| binary unweighted | 0.90 | 0.370 ± 0.111 | 7,540 ± 4,008 | 0.630 ± 0.039 | 0.963 ± 0.008 | — | — | — | — |
| nsv_unweighted | 0.90 | 0.384 ± 0.014 | 6,368 ± 383 | 0.678 ± 0.030 | 0.951 ± 0.007 | 0.426 ± 0.036 | 0.892 ± 0.015 | 0.210 ± 0.155 | 0.399 ± 0.049 |
| step 1b | 0.90 | 0.358 ± 0.077 | 7,448 ± 2,195 | 0.653 ± 0.080 | 0.958 ± 0.018 | — | — | — | — |
| nsv_matched | 0.90 | 0.468 ± 0.137 | 5,020 ± 2,693 | 0.665 ± 0.044 | 0.955 ± 0.010 | 0.463 ± 0.043 | 0.844 ± 0.006 | 0.141 ± 0.097 | 0.592 ± 0.029 |
| binary unweighted | 0.95 | 0.211 ± 0.077 | 17,417 ± 7,234 | 0.813 ± 0.041 | 0.984 ± 0.009 | — | — | — | — |
| nsv_unweighted | 0.95 | 0.191 ± 0.034 | 18,264 ± 4,433 | 0.853 ± 0.023 | 0.973 ± 0.005 | 0.593 ± 0.042 | 0.905 ± 0.017 | 0.048 ± 0.019 | 0.291 ± 0.033 |
| step 1b | 0.95 | 0.203 ± 0.046 | 17,123 ± 4,316 | 0.821 ± 0.052 | 0.981 ± 0.013 | — | — | — | — |
| nsv_matched | 0.95 | 0.270 ± 0.050 | 11,718 ± 3,295 | 0.853 ± 0.023 | 0.974 ± 0.005 | 0.640 ± 0.034 | 0.855 ± 0.003 | 0.067 ± 0.023 | 0.419 ± 0.024 |

- **FN side, at target recall 0.90:**
  - S recall is 0.63–0.68 flagged, so 32–37% of S beats are missed. Under the call rule it is 0.43–0.46, since some flagged S beats are called V.
  - V is 0.95–0.96 flagged. Under the call rule it drops to 0.84–0.89, since V beats called S count as V misses.
  - So S recall at the F1 threshold (0.19–0.31, above) understates what the models can reach, but reaching it costs precision.
- **FP side, at target recall 0.90:**
  - 5,020–7,540 N false positives, 11.4–17.1% of the 44,191 N beats, for abnormal precision 0.36–0.47.
  - +P_S is 0.14–0.21: most beats called S aren't S.
  - At target recall 0.95, FPs rise to 11,718–18,264 (26.5–41.3% of N) and +P_S falls to 0.05–0.07.
- **S is the gap on both sides,** recall and precision.
- nsv_matched has the fewest FPs and the highest precision at both target recalls, but its seed spread at 0.90 is wide (FP 2,561–7,898 by seed).

**DS2, final models (INT8, at each seed's pooled-OOF threshold; reported, never used for a choice):**

| Cell | PR-AUC | F1 | FP | FN | Se_S | +P_S | Se_V | +P_V |
|---|---|---|---|---|---|---|---|---|
| binary unweighted | 0.793 ± 0.060 | 0.678 ± 0.066 | 1,930 ± 1,327 | 1,416 ± 1,058 | 0.466 ± 0.391 | — | 0.865 ± 0.139 | — |
| nsv_unweighted | 0.789 ± 0.083 | 0.698 ± 0.097 | 1,962 ± 1,280 | 1,341 ± 388 | 0.225 ± 0.178 | 0.354 ± 0.137 | 0.887 ± 0.076 | 0.653 ± 0.161 |
| step 1b | 0.817 ± 0.007 | 0.754 ± 0.023 | 758 ± 283 | 1,536 ± 133 | 0.326 ± 0.040 | — | 0.908 ± 0.027 | — |
| nsv_matched | 0.752 ± 0.036 | 0.646 ± 0.053 | 2,022 ± 1,768 | 1,733 ± 478 | 0.108 ± 0.099 | 0.329 ± 0.053 | 0.888 ± 0.092 | 0.656 ± 0.208 |

- **nsv_matched calls 223 ± 7 of DS2's 1,836 S beats V.** Its DS2 S sensitivity is 0.108 ± 0.099.
- Every cell's DS2 numbers carry the unresolved final-model procedure (epoch count and threshold transfer; see "Required before deployment"). Final epochs in the three new 2a runs range from 9 to 28.

### Step 2b design

Fixed before any 2b run.

- **Cells:** a three-way comparison within N/S/V.
  - nsv_unweighted (from 2a)
  - nsv_matched, N 0.75 / S 0.25 / V 0.25 (from 2a)
  - nsv_balanced (new, seeds 42, 43, 44)
- **Balanced weights:**
  - Inverse class frequency in each fold's training view: after the 600-per-record normal cap, before augmentation (which duplicates every beat once and leaves the frequencies unchanged).
  - Normalized so the frequency-weighted mean weight is 1, i.e. w_c = 1 / (3 f_c). The final model uses the same rule on all of DS1.
  - Each run records its weights in `metrics.json`.
- **Decision:**
  - Primary metric: pooled-OOF INT8 S PR-AUC.
  - A cell wins only if its mean beats each other cell's by more than the larger of the two runs' seed std.
  - Guard: the winner's V PR-AUC must not fall below nsv_matched's by more than the larger seed std.
  - Step 1b is reported as a reference row, not a gate.
  - If no cell clears the rule, nsv_matched carries forward, and the result is read as evidence that loss weighting isn't the main lever for S.
  - No seeds are added after seeing the 2b results.
- **Loss scale:** the cells don't have the same data-loss scale against the fixed L2 penalty. Frequency-weighted mean weight:
  - unweighted and balanced: 1
  - matched: 0.627 / 0.618 / 0.615 / 0.620 / 0.608 in folds 0–4, and 0.618 on all of DS1

  So comparing balanced with matched changes two things at once: class balance, and loss scale (with it the relative L2 strength).
- **Early stopping: nsv_balanced stops on the unweighted validation loss.**
  - Fold models early-stop on their held-out fold, the same beats that become the OOF predictions (see "Known limitation: early stopping on the held-out fold" under "Parked").
  - With balanced weights in the val loss, S beats would dominate the stopping criterion (S weight 5.5–9.1). The balanced cell would then pick its stopping epoch by S performance on exactly the patients it is scored on, and S PR-AUC is the 2b primary metric.
  - So nsv_balanced keeps the balanced weights in its training loss, but early-stops on the unweighted focal loss plus L2 over the held-out fold. That is exactly the val_loss nsv_unweighted stopped on. `metrics.json` and `folds.json` record the monitor (`early_stopping_monitor`: `unweighted_val_loss`).
  - nsv_unweighted and nsv_matched are the existing 2a runs. Each stopped on its own val_loss: unweighted for nsv_unweighted, and N 0.75 / S 0.25 / V 0.25 for nsv_matched, which is N-dominated.
  - **Remaining asymmetry:** matched's criterion weights S at a third of N, while the unweighted criterion weights them equally. So nsv_balanced still stops on a slightly more S-sensitive criterion than nsv_matched. A 2b win also needs a win over nsv_unweighted, and that comparison uses an identical criterion, so the asymmetry alone can't produce a win.
- **Pre-declared follow-up, only if nsv_balanced wins:**
  - Train nsv_matched_normalized: the 0.75 / 0.25 / 0.25 ratios rescaled to a frequency-weighted mean of 1, seeds 42, 43, 44. It early-stops on the unweighted validation loss, like nsv_balanced.
  - Compare it with nsv_balanced under the same rule, to tell whether the gain came from class balance or from scale.
  - If nsv_balanced doesn't win, the follow-up is skipped.

Class counts in each training view (capped, before augmentation) and the balanced weights they give:

| Training view | N / S / V | Frequency | Balanced weights N / S / V |
|---|---|---|---|
| fold 0 (209, 203, 215 held out) | 11,400 / 555 / 3,179 | 0.753 / 0.037 / 0.210 | 0.443 / 9.089 / 1.587 |
| fold 1 | 10,800 / 835 / 3,058 | 0.735 / 0.057 / 0.208 | 0.453 / 5.865 / 1.602 |
| fold 2 | 10,800 / 843 / 3,148 | 0.730 / 0.057 / 0.213 | 0.457 / 5.849 / 1.566 |
| fold 3 | 10,200 / 835 / 2,752 | 0.740 / 0.061 / 0.200 | 0.451 / 5.504 / 1.670 |
| fold 4 | 10,200 / 832 / 3,213 | 0.716 / 0.058 / 0.226 | 0.466 / 5.707 / 1.478 |
| all DS1 (final model) | 13,200 / 943 / 3,788 | 0.736 / 0.053 / 0.211 | 0.453 / 6.338 / 1.578 |

Fold 0 holds out 209, 203 and 215, which carry 388 of the 943 S beats in this view, most of them from 209. So fold 0's S weight is the largest (9.1).

## Stage 2 step 2b results: balanced class weights

nsv_balanced was trained with seeds 42, 43 and 44 (`train_cv.py --classes nsv --loss-weights balanced`, in `stage2/step2b_nsv_balanced`). Each run's per-fold and final weights equal the "Step 2b design" table, its loss scale is 1, and every fold model early-stopped on the unweighted val loss. The OOF pool is the same 48,596 beats as in 2a.

**Decision metrics: pooled OOF, INT8.** Seeds 42 / 43 / 44, then mean ± std. Tables: `stage2/step2b/oof_comparison.md` (reference nsv_matched, with nsv_unweighted, nsv_balanced and step 1b) and `stage2/step2b/vs_unweighted/oof_comparison.md` (reference nsv_unweighted).

| Metric | nsv_matched | nsv_unweighted | nsv_balanced | step 1b (reference row) |
|---|---|---|---|---|
| **S PR-AUC (primary)** | 0.118 / 0.087 / 0.168 = 0.124 ± 0.041 | 0.213 / 0.104 / 0.115 = 0.144 ± 0.060 | 0.136 / 0.172 / 0.094 = 0.134 ± 0.039 | 0.096 / 0.151 / 0.232 = 0.160 ± 0.068 |
| S ROC-AUC | 0.862 ± 0.030 | 0.824 ± 0.038 | 0.808 / 0.872 / 0.783 = 0.821 ± 0.046 | 0.801 ± 0.037 |
| V PR-AUC (guard) | 0.904 / 0.899 / 0.934 = 0.912 ± 0.019 | 0.883 ± 0.003 | 0.826 / 0.879 / 0.856 = 0.853 ± 0.027 | 0.914 ± 0.025 |
| V ROC-AUC | 0.977 ± 0.005 | 0.969 ± 0.001 | 0.971 ± 0.006 | 0.977 ± 0.006 |

- **No cell clears the rule.** S PR-AUC gains, against the larger seed std of the pair:
  - balanced − matched: +0.010 (spread 0.041); balanced − unweighted: −0.010 (spread 0.060)
  - unweighted − matched: +0.020 (spread 0.060); unweighted − balanced: +0.010 (spread 0.060)
  - matched has the lowest mean, so it beats neither.
- **nsv_balanced also fails the V guard, and the loss is separated.** V PR-AUC is 0.059 below nsv_matched (spread 0.027); balanced's best seed (0.879) is below matched's worst (0.899). It is also 0.030 below nsv_unweighted (spread 0.027, separated).
- The float rows give the same verdicts. INT8 − float: the largest per-seed gap of any decision metric for nsv_balanced is 0.008 (V PR-AUC).
- **Decision (rule pre-declared in "Step 2b design"): nsv_matched carries forward.** The result is read as evidence that loss weighting isn't the main lever for S. The nsv_matched_normalized follow-up was conditional on a balanced win, so it is skipped.

**Balanced weights mainly shift scores up; they don't rank S better (descriptive).**
- Median OOF abnormal score of S beats: 0.624 ± 0.033 (balanced), against 0.337 ± 0.060 (matched) and 0.473 ± 0.135 (unweighted). The pooled-OOF thresholds move up with it: 0.75 / 0.80 / 0.85, against 0.65 / 0.60 / 0.50 for matched.
- S ranking doesn't improve: S ROC-AUC on the abnormal score is 0.821 ± 0.046, against 0.862 ± 0.030 (matched). On the class's own probability P(S) it is worse in every seed: 0.715 / 0.727 / 0.650 = 0.697 ± 0.041, against 0.805 ± 0.074 (matched) and 0.752 ± 0.074 (unweighted).
- **At each run's selected threshold:**
  - FN side: S recall (flagged abnormal) is 0.321 ± 0.086, against 0.188 ± 0.105 for matched.
  - FP side: the N false-positive rate rises to 0.023 ± 0.005, against 0.013 ± 0.004.
- **At target recall 0.90:**
  - FN side: S flagged 0.665 ± 0.037 (matched 0.665 ± 0.044), S under the call rule 0.455 ± 0.111 (matched 0.463 ± 0.043), V under the call rule 0.856 ± 0.027 (matched 0.844 ± 0.006).
  - FP side: 7,742 ± 1,941 N false positives (matched 5,020 ± 2,693), precision 0.345 ± 0.057 (matched 0.468 ± 0.137), +P_S 0.094 ± 0.012 (matched 0.141 ± 0.097), +P_V 0.421 ± 0.121 (matched 0.592 ± 0.029).
  - At target recall 0.95: 17,740 ± 5,772 FPs (matched 11,718 ± 3,295), +P_S 0.041 ± 0.014 (matched 0.067 ± 0.023).
- **S ↔ V mix-ups at the selected threshold:** S called V 175 / 14 / 107, V called S 177 / 296 / 226.
- **Per-record S ranking (OOF INT8 ROC-AUC, seeds 42 / 43 / 44):**
  - 124 falls to 0.375 / 0.566 / 0.252 (matched 0.754 / 0.512 / 0.724).
  - 209: 0.867 / 0.895 / 0.907, between matched (0.975 / 0.931 / 0.917) and step 1b (0.809 / 0.814 / 0.708).
  - 118 varies less across seeds than under matched: 0.972 / 0.818 / 0.930, against 0.384 / 0.898 / 0.984.
  - 207, 220 and 223: 0.814–0.991.

**DS2, final models (INT8, at each seed's pooled-OOF threshold; reported, never used for a choice):**

| Cell | PR-AUC | F1 | FP | FN | Se_S | +P_S | Se_V | +P_V |
|---|---|---|---|---|---|---|---|---|
| nsv_balanced | 0.786 ± 0.062 | 0.729 ± 0.003 | 1,545 ± 1,032 | 1,266 ± 611 | 0.164 ± 0.022 | 0.261 ± 0.045 | 0.915 ± 0.010 | 0.754 ± 0.198 |

- Seed 43 calls 1,012 of DS2's 1,836 S beats V (seeds 42 and 44: 182 and 105). It also has 2,736 FPs, against 931 and 967.
- Final epochs are 14 / 26 / 14. The unresolved final-model procedure applies, as in every cell.

**Where Stage 2 stands after step 2.** S PR-AUC hasn't moved beyond the seed spread in steps 1, 1b, 2a or 2b: every cell is between 0.12 and 0.16, with seed std 0.04–0.07. The gap after step 2 is large, so step 3 runs. Its design is below, fixed before any step 3 run.

### Step 3 design

Fixed before any step 3 run.

- **Cells, all N/S/V with matched weights (N 0.75 / S 0.25 / V 0.25), focal gamma 2, early stopping on their own val_loss:**
  - **nsv_matched**, the reference. Seeds 42–44 are the 2a runs. Seeds 45 and 46 are new, trained with the same configuration.
  - Directories: the reference's runs are in `stage2/step3_nsv_matched`. Its seeds 42–44 are byte-identical copies of the `stage2/step2a_nsv_matched` run dirs, so 2a's committed tables stay the 3-seed record. 3a goes to `stage2/step3a_rr30` and 3b to `stage2/step3b_rr_norm`.
  - **3a, nsv_rr30:** nsv_matched with a 30-interval R-R lookback instead of 10 (`train_cv.py --rr-lookback 30`).
  - **3b, rr_norm:** `--normalize-rr` on top of the 3a winner, or on top of nsv_matched if 3a has no winner.
- **Seeds: 42, 43, 44, 45 and 46 in every step 3 cell, the reference included.**
  - This is declared before any step 3 result, in response to the 2b observation that with 3 seeds only gains of about 0.04–0.07 in S PR-AUC can count.
  - What 5 seeds buy: the means and the spread estimate are more reliable. A real gain larger than the spread is less likely to be masked by one unlucky seed, and a 3-seed std that is small by chance (V PR-AUC for nsv_unweighted: 0.003) can't set the bar.
  - What they don't buy: the rule's bar is the seed std itself, not a standard error, so it doesn't shrink with more seeds. The smallest gain that can count stays about one seed std.
  - The added nsv_matched seeds are used only in step 3. The 2a and 2b decisions stand as made on seeds 42–44 ("No seeds are added after seeing the 2b results").
  - No seeds are added after seeing step 3 results.
- **Decision, for 3a and again for 3b:**
  - Primary metric: pooled-OOF INT8 S PR-AUC.
  - The candidate wins only if its mean beats the reference's by more than the larger of the two runs' seed std (over seeds 42–46).
  - Guard: its V PR-AUC must not fall below the reference's by more than the larger seed std.
  - `separated`, S and V ROC-AUC, the float rows, P(S) / P(V) AUCs, operating points and per-record AUCs are reported, not decisive.
  - 3a's reference is nsv_matched. 3b's reference is the cell it builds on: the 3a winner, or nsv_matched.
  - If neither 3a nor 3b wins, nsv_matched carries forward. The result is then read as evidence that R-R representation isn't the main lever for S either.
- **What 3a changes.** `local_mean_rr` becomes the mean of the 30 intervals before the beat (the window includes `pre_rr`, as with 10), and `rr_ratio` = `pre_rr` / that mean. `pre_rr`, `post_rr` and `pre_post_ratio` are unchanged.
  - The beat set doesn't change: a beat is usable if it has a beat annotation on each side, whatever the lookback. So the OOF pool stays the same 48,596 beats.
  - Near the start of a record, fewer than 30 intervals exist, and the window uses the ones available, as the 10-interval window does now.
  - Motivation: Stage 1, step 2. 209's SVTA runs fill the 10-interval window and mask prematurity. A 30-interval mean drops their median `rr_ratio` from 0.950 to 0.825 and raises pooled single-feature S-vs-N AUC from 0.727 to 0.849, but only 0.916 → 0.923 inside the 5-feature logistic regression.
- **What 3b changes.** Each R-R feature is standardized, (x − mean) / std, with statistics fitted on the training view of the model that uses them:
  - a fold model's: DS1 minus the held-out fold, after the 600-per-record normal cap and augmentation, exactly the arrays it trains on
  - the final model's: the same view of all DS1
  - The same statistics are applied to that model's held-out fold, its DS2 inputs and its INT8 calibration sample. They are never fitted on held-out or DS2 beats. Each run records them per fold and for the final model in `folds.json` / `metrics.json`.
  - A per-feature affine change could in principle be absorbed by the first Dense layer. Any effect therefore comes through optimization (initialization scale, the fixed L2 penalty, Adam step sizes) or through INT8: the R-R input is one tensor with one scale and zero point, shared by features with very different ranges (intervals in seconds, and ratios whose maximum was 8.76 in the old split's training set; see "3. R-R values outside the representable range"). The INT8 − float table is reported for 3b as for every step.

**Leakage checks for the normalization.**
- **Fitted per model, on its own training view.** Fitting once on all of DS1 would put each held-out fold's R-R distribution into the statistics of the fold model scored on it. A test checks that each fold's recorded statistics equal those fitted on its training view alone, and that the saved fold and final INT8 models, re-scored on inputs normalized with those statistics, reproduce the OOF scores and the DS2 row.
- **Global, not per-recording.** 3b does not standardize each record by its own statistics. Per-recording statistics over a 30-minute recording use beats after the one being classified, so they can't run on a device in real time. They also carry the patient's rhythm, so a record dense with S beats would shift its own mean. A causal, running per-patient version is a different experiment (`local_mean_rr` / `rr_ratio` already are one), and it isn't in step 3.
- **The statistics come from the capped view.** Training keeps 600 N per record (73% N, against 91% in the uncapped OOF pool), so the means and stds are those of the training mix, not of any patient population. That is correct for the model trained on them, but it means the firmware's statistics must be the final model's own, not recomputed from other data.
- **Not changed by step 3, recorded here for deployment:**
  - `post_rr` and `pre_post_ratio` need the next beat, so on a device every classification waits one beat. That is a latency, not a leak.
  - The band-pass filter uses `filtfilt` (zero-phase, non-causal) over the whole recording. The firmware will need a causal filter; that preprocessing mismatch is a separate, existing issue.

**What the firmware would need.**
- **If 3a wins:** a ring buffer of the last 30 beat-to-beat intervals from the on-device R-peak detector (120 bytes as floats). Until it fills, the mean runs over the intervals available, which matches the truncated window at the start of a training record.
- **If 3b wins:**
  - `rr_norm.h` (per-feature mean and std, `export.write_rr_norm_header`) from the final model's training view, generated in the same run as `model_data.h` and `threshold.h`. They are a pair: a stale `rr_norm.h` would silently shift every R-R input.
  - In `deployment_model.ino`: z = (x − mean) / std for each of the 5 features, then `quantize_float_to_int8` with the R-R input tensor's scale and zero point. Those change with the normalization and are regenerated with the model.
  - `train.py` refuses `--normalize-rr` with firmware export until the firmware applies it. That guard lifts in the same change.
- Firmware export stays blocked until the final-model procedure is fixed ("Required before deployment").

**Decision point after step 3.** Once step 3 is decided, and before step 4, decide whether to start the SVDB/INCART training-data work ("Parked idea: add training data from other PhysioNet databases") ahead of Stage 2's remaining steps 4 (global average pooling) and 5 (false-positive re-check). The inputs:
- whether S PR-AUC moved beyond the seed spread in any of steps 1–3
- how much of the S result rests on few patients: 209 holds 383 of DS1's 944 S beats, 124's S ranking is near chance in every cell, and 118's swings by seed
- the step 3 per-record S AUCs

The data work's success test would then use the step 3 reference with seeds 42–46.
