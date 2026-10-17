# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An ECG heartbeat classifier (Normal vs. Abnormal) trained on the MIT-BIH Arrhythmia Database, quantized to INT8, and deployed for on-device inference on an ESP32-S3 via TensorFlow Lite Micro. The project has three stages that live in three different places:

1. **Training** — `train_pipeline.ipynb`, designed to run in Google Colab (not locally).
2. **Converted artifacts** — `ecg_multiscale_model.keras`, `model_int8.tflite`, `deployment_model/model_data.h`, all produced by the notebook.
3. **Firmware** — `deployment_model/deployment_model.ino`, an Arduino sketch for the ESP32-S3 that loads `model_data.h` and runs inference.

There is no local build/lint/test tooling — this is a research/notebook + firmware project, not a packaged application.

## Working with the notebook (`train_pipeline.ipynb`)

The notebook is written for **Google Colab**, not local execution: it does `!pip install wfdb`, imports `google.colab.files`, and uses `files.download(...)` to push artifacts to the user's machine. Running it locally requires stripping/replacing the Colab-specific cells (installs and `files.download` calls).

The notebook has 4 cells; nearly everything is in cell 0:

- **Data acquisition (`load_dual_input_spectrogram_data`)**: streams records directly from PhysioNet's `mitdb` via `wfdb.rdrecord`/`rdann` (no local dataset directory — `data/` is gitignored and not checked in). Train/val/test splits are fixed, hardcoded MIT-BIH record-ID lists at the top of the cell (patient-level split, not random, to avoid leakage). Beats are relabeled into a binary `symbol_map` (0 = Normal: `N,L,R,e,j`; 1 = Abnormal: `V,E,A,a,J,S`).
- **Feature extraction**: for each beat, a 250-sample window around the R-peak is bandpass-filtered (0.5–40 Hz Butterworth) and converted to a normalized STFT spectrogram (`nperseg=64, noverlap=48`). A parallel 5-feature R-R timing vector (`pre_rr, post_rr, local_mean_rr, rr_ratio, pre_post_ratio`) is extracted alongside it — the model is dual-input. `augment=True` (train split only) adds R-peak jitter (±8 samples) to simulate R-peak detection error.
- **Model (`build_multiscale_ptq_model`)**: a dual-input Keras functional model — two parallel Conv2D branches over the spectrogram (temporal `(1,5)` kernel and spectral `(5,1)` kernel) concatenated, plus a separate Dense branch over the 5 R-R features, fused and passed through a Dense head to a single sigmoid output. Uses `relu6` activations and L2 regularization specifically because the model is PTQ (post-training quantization) INT8 target — bounded activations quantize more cleanly.
- **Loss**: custom focal loss (`gamma=2.0, alpha=0.25`), not plain BCE, because Abnormal beats are the minority class.
- **Threshold selection**: the decision threshold is *not* the Keras default — cells 1–2 sweep thresholds and settle on **0.30** (lower than 0.5) to bias toward Abnormal recall over false alarms, since missing an abnormal beat is costlier than a false positive. The firmware's `probability > 0.30f` check in `deployment_model.ino` must stay consistent with whatever threshold the notebook settles on — if you change one, change the other.
- **Export (section 5)**: converts the trained Keras model to INT8 via `TFLiteConverter` with a representative dataset generator (first 100 train samples), then hand-writes a C header (`model_data.h`) containing the TFLite flatbuffer as a `g_model[]` byte array — this is the file the ESP32 firmware links against. `ecg_multiscale_model.keras`, `model_int8.tflite`, and `model_data.h` at the repo root/`deployment_model/` are checked-in outputs of this export step, not sources of truth — regenerate them from the notebook rather than hand-editing.

Local Python environment: `requirements.txt` pins exact versions (TensorFlow 2.21, Keras 3.15, wfdb 4.3.1, etc.) for the `.venv`. There's no local training entry point — the pinned env is for inspecting/post-processing artifacts, not for running the Colab-only acquisition/training cells as-is.

## Working with the firmware (`deployment_model/deployment_model.ino`)

- Targets ESP32-S3, built against `TensorFlowLite_ESP32` (TFLite Micro). `kTensorArenaSize` is fixed at 16 KB — if the model architecture grows, this must be increased or `AllocateTensors()` fails at runtime.
- Expects two input tensors in a fixed order: `input(0)` = spectrogram, `input(1)` = the 5 R-R features; `output(0)` is the single sigmoid probability. This ordering must match the Keras model's `inputs=[spec_input, rr_input]` order from the notebook — TFLite Micro binds by index, not by name.
- All I/O is INT8; `quantize_float_to_int8`/`dequantize_int8_to_float` do the float↔int8 conversion using the per-tensor `scale`/`zero_point` baked into the converted model, which change any time the model is re-trained/re-quantized.
- `loop()` currently feeds hardcoded dummy sensor data (`dummy_rr_features`, constant `0.5f` spectrogram) — real sensor acquisition/preprocessing to match the notebook's filtering + STFT pipeline is not yet implemented on-device.
- Alert threshold (`probability > 0.30f`) mirrors the notebook's threshold-sweep conclusion; see note above.

## Git history note

An earlier QAT (quantization-aware training) pipeline and its model output were removed in favor of the current PTQ (post-training quantization) approach (see commit `7e74dbb`). Don't reintroduce QAT scaffolding unless that decision is being deliberately revisited.
