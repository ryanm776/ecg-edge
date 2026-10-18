import numpy as np
import tensorflow as tf

from ecg.export import convert_to_int8_tflite
from ecg.model import build_multiscale_bn_folded_model, build_multiscale_ptq_model

SPEC_SHAPE = (33, 12, 1)


def _op_names(tflite_bytes):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    return [op["op_name"] for op in interpreter._get_ops_details()]


def _float_tflite(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def _random_data(n, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.random((n, *SPEC_SHAPE), dtype=np.float32),
        rng.uniform(0.2, 2.0, size=(n, 5)).astype(np.float32),
    )


def test_bn_folded_model_matches_baseline_interface():
    baseline = build_multiscale_ptq_model(SPEC_SHAPE)
    folded = build_multiscale_bn_folded_model(SPEC_SHAPE)

    assert [i.name for i in folded.inputs] == [i.name for i in baseline.inputs]
    assert folded.output.shape == baseline.output.shape


def test_baseline_model_leaves_standalone_batchnorm_ops():
    # Documents the root cause in artifacts/diagnosis.md: BN after ReLU6 can't be folded.
    # (With untrained BN params beta=mean=0 the converter drops the ADD half, so check MUL only.)
    ops = _op_names(_float_tflite(build_multiscale_ptq_model(SPEC_SHAPE)))
    assert ops.count("MUL") == 2


def test_bn_folded_model_has_no_standalone_batchnorm_ops():
    ops = _op_names(_float_tflite(build_multiscale_bn_folded_model(SPEC_SHAPE)))
    assert "MUL" not in ops
    assert "ADD" not in ops


def test_bn_folded_int8_conversion_has_no_standalone_batchnorm_ops():
    X_spec, X_rr = _random_data(20)
    model = build_multiscale_bn_folded_model(SPEC_SHAPE)
    ops = _op_names(convert_to_int8_tflite(model, X_spec, X_rr, num_samples=20, seed=0))
    assert "MUL" not in ops
    assert "ADD" not in ops
