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
