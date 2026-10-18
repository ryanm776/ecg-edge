"""Quantization-aware training (Jacob et al., CVPR 2018) via tensorflow-model-optimization.

tfmot only supports Keras 2, so this module must run in a process started with
TF_USE_LEGACY_KERAS=1 and on a model built with `keras_mod=tf_keras` (see
`ecg.model.build_multiscale_bn_folded_model`). Importing tfmot also switches `tf.keras`
to tf_keras for the whole process, so don't import this module from the Keras 3
(PTQ) training path. The baseline architecture (`build_multiscale_ptq_model`) is not
supported: tfmot rejects BatchNormalization after an activation / after Concatenate.
"""
from __future__ import annotations

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras


def quantize_aware(model):
    """Wrap a trained float tf_keras model with fake-quant ops for QAT fine-tuning."""
    if not isinstance(model, tf_keras.Model):
        raise TypeError(
            "QAT needs a tf_keras model (build it with keras_mod=tf_keras under "
            f"TF_USE_LEGACY_KERAS=1); got {type(model).__module__}.{type(model).__name__}"
        )
    return tfmot.quantization.keras.quantize_model(model)


def quantize_scope():
    """Context manager needed to load a saved QAT model."""
    return tfmot.quantization.keras.quantize_scope()


def convert_qat_to_int8_tflite(qat_model):
    """INT8 TFLite from a QAT model. Ranges come from the trained fake-quant ops, so no
    representative dataset is needed. I/O is INT8 to match the PTQ export and firmware.

    `supported_ops = [TFLITE_BUILTINS_INT8]` is deliberately not set: it makes the
    converter demand a representative dataset even for QAT models. The output is still
    fully integer (tests/test_qat.py asserts there's no DEQUANTIZE op).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()
