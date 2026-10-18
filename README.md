# ECG Heartbeat Classifier

A dual-input (spectrogram + R-R timing) 2D CNN that classifies MIT-BIH heartbeats as Normal or
Abnormal, quantized to INT8 and deployed on an ESP32-S3 via TensorFlow Lite Micro.

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

On first run this downloads the 38 MIT-BIH records used by the train/val/test splits into
`data/mitdb/` (cached for subsequent runs), trains the model, converts it to INT8, selects a
decision threshold from the **INT8 model's** validation predictions, and evaluates both the
float and INT8 models on the full test set.

Useful flags: `--epochs`, `--batch-size`, `--threshold-strategy {f1,recall_at_precision}`,
`--min-precision` (for `recall_at_precision`), `--seed`. Run `python scripts/train.py --help`
for the full list.

### Outputs

- `artifacts/ecg_multiscale_model.keras`, `artifacts/model_int8.tflite` — the trained model.
- `artifacts/threshold.json` — the chosen decision threshold, selection strategy, and the
  validation metrics at that threshold. This is the current operating threshold; the firmware's
  `deployment_model/threshold.h` is generated from the same value.
- `artifacts/metrics.json` — overall test-set precision/recall/F1/confusion matrix for both the
  float and INT8 models, plus the INT8 model size in KB.
- `artifacts/per_record_metrics.csv` — per-record breakdown (beats, TP/FP/FN/TN,
  precision/recall/F1) for both models.
- `deployment_model/model_data.h`, `deployment_model/threshold.h` — what the firmware links against.

### Caveat: validation set changed

The original notebook capped normal beats on the validation set to 600/record. This pipeline
removes that cap so validation reflects the full MIT-BIH distribution — but `EarlyStopping`
(`monitor="val_loss"`) now sees a larger, differently-balanced validation set each epoch, so
training may stop at a different epoch and results won't exactly match the original notebook run.

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
