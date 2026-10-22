"""Model architecture and loss function -- ported verbatim from the training notebook."""
from __future__ import annotations

import tensorflow as tf


def build_multiscale_ptq_model(spec_shape):
    spec_input = tf.keras.layers.Input(shape=spec_shape, name="spectrogram_input")

    # Parallel Multi-Scale Kernel Branches
    branch_temporal = tf.keras.layers.Conv2D(
        filters=8, kernel_size=(1, 5), padding="same", activation="relu6", kernel_regularizer="l2"
    )(spec_input)
    branch_spectral = tf.keras.layers.Conv2D(
        filters=8, kernel_size=(5, 1), padding="same", activation="relu6", kernel_regularizer="l2"
    )(spec_input)

    x = tf.keras.layers.concatenate([branch_temporal, branch_spectral])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), padding="same", activation="relu6", kernel_regularizer="l2"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)

    # 5-Feature R-R Timing Branch
    rr_input = tf.keras.layers.Input(shape=(5,), name="rr_input")
    y = tf.keras.layers.Dense(8, activation="relu6")(rr_input)

    # Dense Fusion Network
    combined = tf.keras.layers.concatenate([x, y])
    z = tf.keras.layers.Dense(64, activation="relu6", kernel_regularizer="l2")(combined)
    z = tf.keras.layers.Dropout(0.40)(z)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(z)

    return tf.keras.models.Model(inputs=[spec_input, rr_input], outputs=output)


def build_multiscale_bn_folded_model(spec_shape, keras_mod=tf.keras):
    """Same layers/hyperparameters as build_multiscale_ptq_model, but each conv is
    Conv2D (linear) -> BatchNormalization -> ReLU6, with BN applied per branch *before*
    the concatenate. BN can then be folded into the conv weights at conversion, instead
    of surviving as a standalone INT8 MUL/ADD that amplifies a tensor quantized to only
    a handful of levels (see artifacts/diagnosis.md). This is also the Conv+BN+ReLU
    pattern tfmot's QAT folds. `keras_mod` lets the QAT path build it with tf_keras.

    ReLU6 is always a separate `ReLU(max_value=6)` layer, never `activation="relu6"`:
    tfmot's QuantizeAwareActivation rejects relu6 as a built-in layer activation. The
    converter still fuses it into the preceding CONV_2D / FULLY_CONNECTED op.
    """
    layers = keras_mod.layers

    def conv_bn_relu6(x, filters, kernel_size):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same", kernel_regularizer="l2")(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU(max_value=6.0)(x)

    spec_input = layers.Input(shape=spec_shape, name="spectrogram_input")

    # Parallel Multi-Scale Kernel Branches
    branch_temporal = conv_bn_relu6(spec_input, 8, (1, 5))
    branch_spectral = conv_bn_relu6(spec_input, 8, (5, 1))

    x = layers.concatenate([branch_temporal, branch_spectral])
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = conv_bn_relu6(x, 32, (3, 3))
    x = layers.Flatten()(x)

    # 5-Feature R-R Timing Branch
    rr_input = layers.Input(shape=(5,), name="rr_input")
    y = layers.ReLU(max_value=6.0)(layers.Dense(8)(rr_input))

    # Dense Fusion Network
    combined = layers.concatenate([x, y])
    z = layers.ReLU(max_value=6.0)(layers.Dense(64, kernel_regularizer="l2")(combined))
    z = layers.Dropout(0.40)(z)
    output = layers.Dense(1, activation="sigmoid", name="output")(z)

    return keras_mod.models.Model(inputs=[spec_input, rr_input], outputs=output)


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_factor * focal_factor * bce)
    return loss
