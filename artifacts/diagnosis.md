# Float → INT8 gap: diagnosis

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
