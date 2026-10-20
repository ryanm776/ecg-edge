"""INT8 post-training quantization export: TFLite conversion and C header writers."""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def quantize_value(value, scale, zero_point):
    """Float -> INT8, matching the firmware's quantize_float_to_int8. Works on scalars or arrays."""
    arr = np.asarray(value, dtype=np.float64)
    if scale <= 0:
        q = np.zeros_like(arr) + zero_point
    else:
        q = np.round(arr / scale) + zero_point
    q = np.clip(q, -128, 127).astype(np.int8)
    return int(q) if arr.ndim == 0 else q


def dequantize_value(value, scale, zero_point):
    """INT8 -> float, matching the firmware's dequantize_int8_to_float. Works on scalars or arrays."""
    arr = np.asarray(value, dtype=np.float64)
    result = (arr - zero_point) * scale
    return float(result) if arr.ndim == 0 else result


def convert_to_int8_tflite(model, X_train_spec, X_train_rr, num_samples=100, seed=None):
    """Same INT8 PTQ settings as the notebook, but the calibration set is a shuffled
    sample of the training set rather than the first `num_samples` examples in order.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_train_spec))[:num_samples]

    def representative_data_gen():
        for i in indices:
            yield {
                "spectrogram_input": X_train_spec[i:i + 1].astype(np.float32),
                "rr_input": X_train_rr[i:i + 1].astype(np.float32),
            }

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def write_c_header(tflite_bytes, path, array_name="g_model"):
    """Write a TFLite flatbuffer as a C byte array header, e.g. for deployment_model/model_data.h."""
    guard = f"{array_name.upper()}_DATA_H_"
    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(tflite_bytes)};\n")
        f.write(f"const unsigned char {array_name}[] __attribute__((aligned(4))) = {{\n  ")
        for i, byte in enumerate(tflite_bytes):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")
        f.write(f"\n}};\n\n#endif  // {guard}\n")


def write_rr_norm_header(mean, std, path):
    """Write the R-R normalizer stats as C++ constexpr arrays; the firmware applies
    (x - kRrMean[i]) / kRrStd[i] before quantizing the R-R input."""
    guard = "RR_NORM_H_"

    def fmt(values):
        return ", ".join(f"{float(v):.6f}f" for v in values)

    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"constexpr float kRrMean[{len(mean)}] = {{{fmt(mean)}}};\n")
        f.write(f"constexpr float kRrStd[{len(std)}] = {{{fmt(std)}}};\n\n")
        f.write(f"#endif  // {guard}\n")


def write_threshold_header(threshold, path, const_name="kAbnormalThreshold"):
    """Write the decision threshold as a C++ constexpr, so firmware stays in sync with training."""
    guard = f"{const_name.upper()}_H_"
    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"constexpr float {const_name} = {threshold:.4f}f;\n\n")
        f.write(f"#endif  // {guard}\n")
