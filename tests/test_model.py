import hashlib

import numpy as np
import tensorflow as tf

from ecg.evaluate import min_int8_levels
from ecg.export import convert_to_int8_tflite
from ecg.model import (
    build_multiscale_bn_folded_model,
    build_multiscale_ptq_model,
    categorical_focal_loss,
    focal_loss,
)

SPEC_SHAPE = (33, 12, 1)

# Labels and sigmoid outputs for the loss tests, kept away from the 1e-7 clip.
Y_BINARY = np.array([0, 1, 1, 0, 1, 0], dtype=np.float32)
P_BINARY = np.array([0.1, 0.8, 0.3, 0.6, 0.95, 0.02], dtype=np.float32)
Y_NSV = np.array([0, 1, 2, 0, 2, 1])
P_NSV = np.array([
    [0.7, 0.2, 0.1],
    [0.5, 0.3, 0.2],
    [0.1, 0.1, 0.8],
    [0.2, 0.6, 0.2],
    [0.3, 0.3, 0.4],
    [0.05, 0.9, 0.05],
], dtype=np.float32)


def _loss(fn, y, p):
    return float(fn(tf.constant(y), tf.constant(p)))


def _binary_focal_terms(y, p, gamma=2.0):
    p_t = np.where(y == 1, p, 1 - p)
    return (1 - p_t) ** gamma * -np.log(p_t)


def test_focal_loss_default_is_alpha_025_gamma_2():
    # The step 1b runs are the binary-matched cell of step 2a, so the default must stay this formula.
    weights = np.where(Y_BINARY == 1, 0.25, 0.75)
    expected = np.mean(weights * _binary_focal_terms(Y_BINARY, P_BINARY))
    assert np.isclose(_loss(focal_loss(), Y_BINARY, P_BINARY), expected, rtol=1e-6)


def test_focal_loss_alpha_none_weights_both_classes_one():
    expected = np.mean(_binary_focal_terms(Y_BINARY, P_BINARY))
    assert np.isclose(_loss(focal_loss(alpha=None), Y_BINARY, P_BINARY), expected, rtol=1e-6)
    # Not alpha=0.5, which halves it.
    assert np.isclose(_loss(focal_loss(alpha=0.5), Y_BINARY, P_BINARY), expected / 2, rtol=1e-6)


def test_categorical_focal_loss_matches_hand_computed():
    p_t = P_NSV[np.arange(len(Y_NSV)), Y_NSV]
    terms = (1 - p_t) ** 2 * -np.log(p_t)
    weights = np.array([0.75, 0.25, 0.25])[Y_NSV]

    assert np.isclose(_loss(categorical_focal_loss(), Y_NSV, P_NSV), np.mean(terms), rtol=1e-6)
    assert np.isclose(_loss(categorical_focal_loss(class_weights=(0.75, 0.25, 0.25)), Y_NSV, P_NSV),
                      np.mean(weights * terms), rtol=1e-6)


def test_categorical_focal_loss_accepts_column_labels():
    # Keras may pass sparse labels as (n,) or (n, 1).
    fn = categorical_focal_loss(class_weights=(0.75, 0.25, 0.25))
    assert np.isclose(_loss(fn, Y_NSV.reshape(-1, 1), P_NSV), _loss(fn, Y_NSV, P_NSV), rtol=1e-6)


def test_two_class_categorical_focal_loss_equals_binary_focal_loss():
    p_two = np.stack([1 - P_BINARY, P_BINARY], axis=1)
    y_int = Y_BINARY.astype(int)
    assert np.isclose(_loss(categorical_focal_loss(class_weights=(0.75, 0.25)), y_int, p_two),
                      _loss(focal_loss(alpha=0.25), Y_BINARY, P_BINARY), rtol=1e-6)
    assert np.isclose(_loss(categorical_focal_loss(), y_int, p_two),
                      _loss(focal_loss(alpha=None), Y_BINARY, P_BINARY), rtol=1e-6)


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


def test_min_int8_levels_reports_an_op_output_tensor(tmp_path):
    X_spec, X_rr = _random_data(20)
    model = build_multiscale_bn_folded_model(SPEC_SHAPE)
    path = tmp_path / "model.tflite"
    path.write_bytes(convert_to_int8_tflite(model, X_spec, X_rr, num_samples=20, seed=0))

    levels, name = min_int8_levels(path, X_spec[:5], X_rr[:5])

    assert 0 <= levels <= 255
    interpreter = tf.lite.Interpreter(model_path=str(path))
    input_names = {d["name"] for d in interpreter.get_input_details()}
    assert name not in input_names  # model inputs aren't op outputs


# sha256 of the bn_folded binary model's initial weights (float32 bytes, in get_weights() order) at seed 42,
# recorded from the builder as of 33cc699, before `n_classes` existed. Step 1b ran on that builder and is the
# binary-matched cell of step 2a, so the binary model must still start from the same weights. Recorded with
# TF 2.21.0 / Keras 3.15.1; a TF or Keras upgrade can change initializer output and would need a new digest.
BN_FOLDED_SEED42_WEIGHTS_SHA256 = "791398219cad694ddf4d31afed923091a6285ad4f23ef2170cc937ba80d7186b"


def _seeded_weights(seed=42, **kwargs):
    tf.keras.utils.set_random_seed(seed)
    return build_multiscale_bn_folded_model(SPEC_SHAPE, **kwargs).get_weights()


def _weights_sha256(weights):
    h = hashlib.sha256()
    for w in weights:
        h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()


def test_binary_bn_folded_initial_weights_unchanged_by_n_classes():
    assert _weights_sha256(_seeded_weights()) == BN_FOLDED_SEED42_WEIGHTS_SHA256
    assert _weights_sha256(_seeded_weights(n_classes=1)) == BN_FOLDED_SEED42_WEIGHTS_SHA256


def test_multiclass_bn_folded_differs_from_binary_only_in_output_layer():
    binary = build_multiscale_bn_folded_model(SPEC_SHAPE)
    multi = build_multiscale_bn_folded_model(SPEC_SHAPE, n_classes=3)

    assert binary.output.shape[-1] == 1 and multi.output.shape[-1] == 3
    assert multi.get_layer("output").get_config()["activation"] == "softmax"
    assert multi.count_params() - binary.count_params() == 2 * 65  # Dense(64 -> 3) vs. Dense(64 -> 1)

    # Same seed: every weight before the output layer starts identical.
    w_binary, w_multi = _seeded_weights(), _seeded_weights(n_classes=3)
    assert len(w_binary) == len(w_multi)
    for a, b in zip(w_binary[:-2], w_multi[:-2]):
        np.testing.assert_array_equal(a, b)


def test_multiclass_bn_folded_int8_conversion_has_no_standalone_batchnorm_ops():
    X_spec, X_rr = _random_data(20)
    model = build_multiscale_bn_folded_model(SPEC_SHAPE, n_classes=3)
    ops = _op_names(convert_to_int8_tflite(model, X_spec, X_rr, num_samples=20, seed=0))
    assert "MUL" not in ops
    assert "ADD" not in ops
    assert "SOFTMAX" in ops
