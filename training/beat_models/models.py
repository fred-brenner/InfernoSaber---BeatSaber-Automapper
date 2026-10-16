"""Small serializable stage models and losses; importing never starts training."""

import tensorflow as tf

from .contracts import (CLASSES, CORE, FEATURES, FRAME_SECONDS, HALO, HISTORY,
                        MELS, PAD, PATCH, TIMING_WINDOW)

keras = tf.keras


@keras.utils.register_keras_serializable(package="InfernoSaber")
class ApplyMask(keras.layers.Layer):
    def call(self, inputs):
        values, mask = inputs
        return values * tf.cast(mask[..., None], values.dtype)


@keras.utils.register_keras_serializable(package="InfernoSaber")
class DurationSoftmax(keras.layers.Layer):
    """Turn log densities into whole-song mass, excluding padded bins."""

    def call(self, inputs):
        logits, durations, mask = inputs
        logits = tf.squeeze(logits, -1)
        mask = tf.cast(mask, tf.bool)
        scores = logits + tf.math.log(tf.maximum(durations, keras.backend.epsilon()))
        scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        mass = tf.exp(scores - tf.reduce_max(scores, axis=1, keepdims=True))
        mass *= tf.cast(mask, mass.dtype)
        return tf.math.divide_no_nan(mass, tf.reduce_sum(mass, axis=1, keepdims=True))


def _temporal_stack(values, mask, channels, dilations, prefix):
    values = keras.layers.Conv1D(channels, 1, name=f"{prefix}_projection")(values)
    values = ApplyMask(name=f"{prefix}_input_mask")([values, mask])
    for dilation in dilations:
        residual = values
        for index in range(2):
            name = f"{prefix}_d{dilation}_{index}"
            values = keras.layers.Conv1D(channels, 3, padding="same", dilation_rate=dilation,
                                         name=f"{name}_conv")(values)
            values = keras.layers.LayerNormalization(name=f"{name}_norm")(values)
            values = keras.layers.Activation("relu", name=f"{name}_relu")(values)
            values = keras.layers.Dropout(0.1, name=f"{name}_dropout")(values)
            values = ApplyMask(name=f"{name}_mask")([values, mask])
        values = keras.layers.Add(name=f"{prefix}_d{dilation}_residual")([residual, values])
        values = ApplyMask(name=f"{prefix}_d{dilation}_mask")([values, mask])
    return values


def build_intensity_model():
    inputs = {"audio": keras.Input((None, FEATURES), name="audio"),
              "durations": keras.Input((None,), name="durations"),
              "mask": keras.Input((None,), dtype="bool", name="mask")}
    values = ApplyMask()([inputs["audio"], inputs["mask"]])
    values = _temporal_stack(values, inputs["mask"], 48, (1, 2, 4, 8), "intensity")
    logits = keras.layers.Conv1D(1, 1, name="log_density")(values)
    shares = DurationSoftmax(name="shares")([logits, inputs["durations"], inputs["mask"]])
    return keras.Model(inputs, shares, name="intensity")


def build_timing_model():
    inputs = {"audio": keras.Input((TIMING_WINDOW, FEATURES), name="audio"),
              "conditions": keras.Input((TIMING_WINDOW, 4), name="conditions"),
              "mask": keras.Input((TIMING_WINDOW,), dtype="bool", name="mask")}
    values = keras.layers.Concatenate()([inputs["audio"], inputs["conditions"]])
    values = ApplyMask()([values, inputs["mask"]])
    values = _temporal_stack(values, inputs["mask"], 64, (1, 2, 4, 8, 16, 32, 64), "timing")
    values = keras.layers.Cropping1D((HALO, HALO), name="valid_core")(values)
    probabilities = keras.layers.Conv1D(3, 1, activation="softmax", name="counts")(values)
    return keras.Model(inputs, probabilities, name="timing")


def build_notes_model():
    inputs = {"audio": keras.Input((PATCH, MELS, 1), name="audio"),
              "audio_mask": keras.Input((PATCH,), dtype="bool", name="audio_mask"),
              "history": keras.Input((HISTORY, 2), dtype="int32", name="history"),
              "history_age": keras.Input((HISTORY, 1), name="history_age"),
              "history_mask": keras.Input((HISTORY,), dtype="bool", name="history_mask"),
              "conditions": keras.Input((6,), name="conditions")}
    audio_mask = keras.layers.Reshape((PATCH, 1))(inputs["audio_mask"])
    audio = ApplyMask(name="audio_valid_frames")([inputs["audio"], audio_mask])
    audio = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(audio)
    audio = keras.layers.MaxPooling2D((4, 4))(audio)
    audio = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(audio)
    audio = keras.layers.MaxPooling2D((5, 4))(audio)
    audio = keras.layers.Flatten()(audio)
    audio = keras.layers.Dense(64, activation="relu", name="audio_features")(audio)
    history = keras.layers.Embedding(PAD + 1, 16, name="hand_embedding")(inputs["history"])
    history = keras.layers.Reshape((HISTORY, 32))(history)
    history = keras.layers.Dropout(0.1, name="history_dropout")(history)
    history = keras.layers.Concatenate()([history, inputs["history_age"]])
    history = keras.layers.GRU(64, name="history_gru")(history, mask=inputs["history_mask"])
    joined = keras.layers.Concatenate()([audio, history, inputs["conditions"]])
    joined = keras.layers.Dense(128, activation="relu", name="shared_features")(joined)
    outputs = {hand: keras.layers.Dense(CLASSES, activation="softmax", name=hand)(joined)
               for hand in ("red", "blue")}
    return keras.Model(inputs, outputs, name="notes")


BUILDERS = {"intensity": build_intensity_model, "timing": build_timing_model,
            "notes": build_notes_model}


def build_model(stage):
    if stage not in BUILDERS:
        raise ValueError("Unknown model stage")
    model = BUILDERS[stage]()
    if model.count_params() >= 1_000_000:
        raise ValueError("Stage model exceeds the one-million-parameter limit")
    return model


def _masked_mean(values, mask):
    mask = tf.cast(mask, values.dtype)
    return tf.math.divide_no_nan(tf.reduce_sum(values * mask), tf.reduce_sum(mask))


def intensity_loss(target, predicted, mask=None, target_valid=None):
    target = tf.cast(target, predicted.dtype)
    terms = -target * tf.math.log(tf.clip_by_value(predicted, keras.backend.epsilon(), 1.0))
    if mask is not None:
        terms *= tf.cast(mask, terms.dtype)
    per_song = tf.reduce_sum(terms, axis=-1)
    valid = tf.reduce_sum(target, axis=-1) > 0 if target_valid is None else target_valid
    return _masked_mean(per_song, valid)


def timing_loss_components(target, predicted, mask):
    """Return CE/frame and rate-Huber/chunk numerators and denominators."""
    target = tf.cast(target, tf.int32)
    mask = tf.cast(mask, predicted.dtype)
    cross_entropy = keras.losses.sparse_categorical_crossentropy(target, predicted)
    expected = tf.reduce_sum(predicted * tf.cast([0, 1, 2], predicted.dtype), axis=-1)
    frames = tf.reduce_sum(mask, axis=-1)
    error = tf.math.divide_no_nan(tf.reduce_sum((expected - tf.cast(target, predicted.dtype)) * mask,
                                                axis=-1), 10 * FRAME_SECONDS * frames)
    absolute = tf.abs(error)
    huber = tf.where(absolute <= 1, 0.5 * tf.square(error), absolute - 0.5)
    valid_chunks = tf.cast(frames > 0, predicted.dtype)
    return (tf.reduce_sum(cross_entropy * mask), tf.reduce_sum(mask),
            tf.reduce_sum(huber * valid_chunks), tf.reduce_sum(valid_chunks))


def timing_loss(target, predicted, mask):
    ce_sum, frames, huber_sum, chunks = timing_loss_components(target, predicted, mask)
    return tf.math.divide_no_nan(ce_sum, frames) + 0.1 * tf.math.divide_no_nan(huber_sum, chunks)


def notes_loss(target, predicted, mask=None):
    target = tf.cast(target, tf.int32)
    losses = [keras.losses.sparse_categorical_crossentropy(target[:, hand], predicted[name])
              for hand, name in enumerate(("red", "blue"))]
    per_event = (losses[0] + losses[1]) / 2
    return tf.reduce_mean(per_event) if mask is None else _masked_mean(per_event, mask)


@keras.utils.register_keras_serializable(package="InfernoSaber")
def intensity_objective(target, predicted):
    return intensity_loss(target, predicted)


@keras.utils.register_keras_serializable(package="InfernoSaber")
def timing_objective(target_and_mask, predicted):
    return timing_loss(target_and_mask[..., 0], predicted, target_and_mask[..., 1])
