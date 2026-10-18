# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An ECG heartbeat classifier (Normal vs. Abnormal) trained on the MIT-BIH Arrhythmia Database, quantized to INT8, and deployed for on-device inference on an ESP32-S3 via TensorFlow Lite Micro. The project has three stages that live in three different places:

1. **Training** — the `ecg/` package + `scripts/train.py`, runs locally. (`train_pipeline.ipynb` is the original Colab notebook this was extracted from; it's superseded and kept only as a Colab-runnable reference — see the note at the top of the notebook.)
2. **Generated artifacts** — `artifacts/ecg_multiscale_model.keras`, `artifacts/model_int8.tflite`, `artifacts/threshold.json`, `artifacts/metrics.json`, `artifacts/per_record_metrics.csv`, `deployment_model/model_data.h`, `deployment_model/threshold.h` — all produced by `scripts/train.py`. Not sources of truth; regenerate rather than hand-edit.
3. **Firmware** — `deployment_model/deployment_model.ino`, an Arduino sketch for the ESP32-S3 that loads `model_data.h` + `threshold.h` and runs inference.

## Commands

```bash
pip install -r requirements.txt   # once, inside .venv
python scripts/train.py           # train + evaluate + export everything (see README.md for flags)
pytest                            # unit tests (features, export, threshold selection) — no network/data needed
```

## Working with the `ecg` package

- **`ecg/data.py`**: `ensure_mitdb_cached` / `load_record` download and cache MIT-BIH records under `<data_dir>/mitdb` (only the records actually needed, not the full 48-record database); `build_dataset` orchestrates loading + feature extraction across a record list and returns `X_spec, X_rr, y, record_ids`. Train/val/test splits are fixed, hardcoded MIT-BIH record-ID lists in `scripts/train.py` (patient-level split, not random, to avoid leakage). `max_normal_per_record` caps normal beats for train (class-balance control); val/test pass `None` (no cap) so reported metrics reflect the real class distribution.
- **`ecg/features.py`**: pure signal-processing functions, no I/O — `SYMBOL_MAP` (binary relabeling: 0 = Normal `N,L,R,e,j`; 1 = Abnormal `V,E,A,a,J,S`), `bandpass_filter` (0.5–40 Hz Butterworth), `compute_spectrogram` (STFT, `nperseg=64, noverlap=48`, normalized), `compute_rr_timing_features` (`pre_rr, post_rr, local_mean_rr, rr_ratio, pre_post_ratio`), `extract_beats_from_record` (windowing + R-peak-jitter augmentation, seeded via an injected `rng`). Kept I/O-free specifically so these are unit-testable with synthetic signals (see `tests/test_features.py`).
- **`ecg/model.py`**: `build_multiscale_ptq_model` + `focal_loss` — ported verbatim from the notebook, architecture/hyperparameters unchanged. Dual-input: two parallel Conv2D branches over the spectrogram (temporal `(1,5)` and spectral `(5,1)` kernels) concatenated, plus a Dense branch over the 5 R-R features, fused into a sigmoid output. `relu6` + L2 regularization specifically because the model targets INT8 PTQ — bounded activations quantize more cleanly. Focal loss (`gamma=2.0, alpha=0.25`) instead of plain BCE because Abnormal beats are the minority class.
- **`ecg/export.py`**: `convert_to_int8_tflite` (INT8 PTQ via `TFLiteConverter`, calibration set is a **shuffled** sample of training data, not the first N in order); `write_c_header` (writes the TFLite flatbuffer as `deployment_model/model_data.h`); `write_threshold_header` (writes the chosen threshold as `deployment_model/threshold.h`, a `constexpr float kAbnormalThreshold`); `quantize_value`/`dequantize_value` (shared with `ecg/evaluate.py`, mirror the firmware's `quantize_float_to_int8`/`dequantize_int8_to_float`).
- **`ecg/evaluate.py`**: `select_threshold` sweeps thresholds on **validation** predictions from the **INT8** model (not the float model, not the test set — the INT8 model is what's actually deployed) and picks one by `strategy="f1"` (maximize Abnormal F1) or `"recall_at_precision"` (lowest threshold meeting a precision floor); `predict_tflite` runs the `.tflite` model via `tf.lite.Interpreter`, binding input/output tensors **by name substring**, not by index (unlike the firmware — see below); `evaluate_predictions` returns overall + per-record metrics (`zero_division=0`, with `n_abnormal` so a genuine 0 can be told apart from an undefined metric).

## Working with the firmware (`deployment_model/deployment_model.ino`)

- Targets ESP32-S3, built against `TensorFlowLite_ESP32` (TFLite Micro). `kTensorArenaSize` is fixed at 16 KB — if the model architecture grows, this must be increased or `AllocateTensors()` fails at runtime.
- Expects two input tensors in a fixed order: `input(0)` = spectrogram, `input(1)` = the 5 R-R features; `output(0)` is the single sigmoid probability. This ordering must match the Keras model's `inputs=[spec_input, rr_input]` order from `ecg/model.py` — TFLite Micro binds by index, not by name (contrast with `ecg/evaluate.py`'s name-based binding, which is more robust but not an option on-device).
- All I/O is INT8; `quantize_float_to_int8`/`dequantize_int8_to_float` do the float↔int8 conversion using the per-tensor `scale`/`zero_point` baked into the converted model, which change any time the model is re-trained/re-quantized.
- `loop()` currently feeds hardcoded dummy sensor data (`dummy_rr_features`, constant `0.5f` spectrogram) — real sensor acquisition/preprocessing to match `ecg/features.py`'s filtering + STFT pipeline is not yet implemented on-device.
- Alert threshold (`probability > kAbnormalThreshold`, from `threshold.h`) is generated by `scripts/train.py` from the same value recorded in `artifacts/threshold.json` — don't hardcode a threshold here; regenerate `threshold.h` instead.

## Deployment constraints

- The deployed firmware will detect R-peaks itself using Pan-Tompkins, not ground-truth annotations. Feature extraction should accept R-peak locations from any source (annotation-derived or detector-derived), so the model can later be evaluated using Pan-Tompkins detections on MIT-BIH instead of the expert annotations — not yet implemented.

## Git history note

An earlier QAT (quantization-aware training) pipeline and its model output were removed in favor of the current PTQ (post-training quantization) approach (see commit `7e74dbb`). Don't reintroduce QAT scaffolding unless that decision is being deliberately revisited.
